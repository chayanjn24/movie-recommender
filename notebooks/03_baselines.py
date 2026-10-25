# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Baselines
# MAGIC Three simpler recommenders that ALS has to beat:
# MAGIC
# MAGIC - **Popularity:** the most-rated movies the user hasn't rated yet. Same list for everyone.
# MAGIC - **Genre popularity:** popular movies re-weighted by how much of the user's history is in each of the movie's genres. Score = mean genre share × log(1 + rating count).
# MAGIC - **Item-kNN:** "people who rated this also rated". Cosine similarity between movies on who rated them; a movie's score = sum of its similarity to everything in the user's history. The neighbourhood size is tuned on val.
# MAGIC
# MAGIC Popularity and genre popularity pick from the 1,000 most-rated movies. Item-kNN uses movies with at least 5 training ratings; its self-join grows with the square of each user's history, so it runs on the small dataset only and is skipped at 25M. Scored on val (fit on train) and on test (fit on train + val). Results go to `eval_results`.

# COMMAND ----------

# Dataset dropdown. Run this cell, pick small or 25m at the top, then run the rest.
try:
    dbutils.widgets.dropdown("dataset", "small", ["small", "25m"], "Dataset")
except NameError:  # outside Databricks
    pass

# COMMAND ----------

# MAGIC %run ./00_common

# COMMAND ----------

CANDIDATES = 1000
s = load_split()
movie_genres = spark.table(tbl("movies_bronze")).select("movieId", F.explode(F.split("genres", r"\|")).alias("genre"))


def top_popular(train):
    return train.groupBy("movieId").agg(F.count("*").alias("cnt")).orderBy(F.desc("cnt"), "movieId").limit(CANDIDATES)


def popularity_scores(train, users):
    return users.crossJoin(top_popular(train).withColumnRenamed("cnt", "score"))


def genre_popularity_scores(train, users):
    cand = top_popular(train).join(movie_genres, "movieId").withColumn(
        "n_genres", F.count("*").over(Window.partitionBy("movieId"))
    )
    user_genre = (
        train.join(users, "userId").join(movie_genres, "movieId")
        .groupBy("userId", "genre").count()
        .withColumn("share", F.col("count") / F.sum("count").over(Window.partitionBy("userId")))
    )
    return (
        user_genre.join(cand, "genre")
        .groupBy("userId", "movieId")
        .agg((F.sum("share") / F.first("n_genres") * F.log1p(F.first("cnt"))).alias("score"))
    )


MIN_SUPPORT, NEIGHBOR_GRID = 5, [20, 50, 100]


def item_similarities(train):
    """Cosine similarity between movies on who rated them (binary), keeping the top max(NEIGHBOR_GRID) per movie.

    Written to a Delta table because serverless can't cache, and every neighbourhood size reuses it.
    """
    counts = train.groupBy("movieId").agg(F.count("*").alias("n")).filter(F.col("n") >= MIN_SUPPORT)
    inter = train.select("userId", "movieId").join(counts.select("movieId"), "movieId")
    co = (
        inter.alias("a").join(inter.alias("b"), "userId")
        .filter(F.col("a.movieId") != F.col("b.movieId"))
        .groupBy(F.col("a.movieId").alias("i"), F.col("b.movieId").alias("j")).count()
    )
    sim = (
        co.join(counts.toDF("i", "n_i"), "i").join(counts.toDF("j", "n_j"), "j")
        .withColumn("sim", F.col("count") / F.sqrt(F.col("n_i") * F.col("n_j")))
        .withColumn("nbr_rank", F.row_number().over(Window.partitionBy("i").orderBy(F.desc("sim"), "j")))
        .filter(F.col("nbr_rank") <= max(NEIGHBOR_GRID))
        .select("i", "j", "sim", "nbr_rank")
    )
    sim.write.mode("overwrite").saveAsTable(tbl("item_sims"))
    return spark.table(tbl("item_sims"))


def item_knn_scores(train, users, sims, n_neighbors):
    history = train.join(users, "userId").select("userId", F.col("movieId").alias("i"))
    return (
        history.join(sims.filter(F.col("nbr_rank") <= n_neighbors), "i")
        .groupBy("userId", F.col("j").alias("movieId"))
        .agg(F.sum("sim").alias("score"))
    )

# COMMAND ----------

results = []
for split, train, truth in [("val", s["train"], s["val"]), ("test", s["train_full"], s["test"])]:
    users = truth.select("userId").distinct()
    for name, score_fn in [("popularity", popularity_scores), ("genre_popularity", genre_popularity_scores)]:
        recs = topn_from_scores(score_fn(train, users), train)
        results.append(evaluate(recs, truth, train, model_name=name, split=split, params={"candidates": CANDIDATES}))
        print(f"{split} | {name}: NDCG@{K} = {results[-1]['ndcg']:.4f}")

    if DATASET != "small":
        continue  # item-kNN's self-join is too large at 25M
    sims = item_similarities(train)
    if split == "val":
        knn_val = {}
        for n in NEIGHBOR_GRID:
            recs = topn_from_scores(item_knn_scores(train, users, sims, n), train)
            knn_val[n] = evaluate(recs, truth, train, "item_knn", split, params={"neighbors": n, "min_support": MIN_SUPPORT})
            print(f"val | item_knn, {n} neighbours: NDCG@{K} = {knn_val[n]['ndcg']:.4f}")
        best_n = max(knn_val, key=lambda n: knn_val[n]["ndcg"])
        results.append(knn_val[best_n])
    else:
        recs = topn_from_scores(item_knn_scores(train, users, sims, best_n), train)
        results.append(evaluate(recs, truth, train, "item_knn", split, params={"neighbors": best_n, "min_support": MIN_SUPPORT}))
        print(f"test | item_knn, {best_n} neighbours: NDCG@{K} = {results[-1]['ndcg']:.4f}")

save_results(results)
display(pd.DataFrame(results)[["model", "split", "users", "precision", "recall", "ndcg", "map", "coverage", "pop_pct"]])

# COMMAND ----------

# MAGIC %md ## Sanity check: one user's genre-popularity recommendations

# COMMAND ----------

titles = spark.table(tbl("movies_bronze")).select("movieId", "title", "genres")
user = s["test"].select("userId").orderBy("userId").first()[0]
print(f"user {user}: highest-rated movies in training")
display(s["train_full"].filter(F.col("userId") == user).join(titles, "movieId").orderBy(F.desc("rating"), F.desc("timestamp")).limit(10))

recs = topn_from_scores(genre_popularity_scores(s["train_full"], spark.createDataFrame([(user,)], "userId INT")), s["train_full"])
hits = s["test"].filter(F.col("userId") == user).select("movieId", F.lit(True).alias("in_test"))
print("top 10 recommendations (in_test = the user actually rated it next)")
display(recs.join(titles, "movieId").join(hits, "movieId", "left").orderBy("rank"))
