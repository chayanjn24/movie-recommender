# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Temporal per-user split
# MAGIC Each user's **last 5** ratings are the test set and the **5 before those** are validation. Everything earlier is train. This mirrors the real question, "what will this user watch next?", and avoids the leakage of a random split, where a model can train on a user's future.
# MAGIC
# MAGIC - Tune on train → val. Fit the final models on train + val (`train_full`) → score on test.
# MAGIC - Users with fewer than 15 ratings stay in train only. (Every MovieLens user has at least 20, so none are dropped.)
# MAGIC - Ties on timestamp are broken by movieId so the split is deterministic.
# MAGIC
# MAGIC Writes one table, `ratings_split`, with a `part` column (train / val / test).

# COMMAND ----------

# Dataset dropdown. Run this cell, pick small or 25m at the top, then run the rest.
try:
    dbutils.widgets.dropdown("dataset", "small", ["small", "25m"], "Dataset")
except NameError:  # outside Databricks
    pass

# COMMAND ----------

# MAGIC %run ./00_common

# COMMAND ----------

N_TEST, N_VAL, MIN_TRAIN = 5, 5, 5

ratings = spark.table(tbl("ratings_bronze")).select("userId", "movieId", "rating", "timestamp")
newest_first = Window.partitionBy("userId").orderBy(F.desc("timestamp"), F.desc("movieId"))
eligible = F.col("n") >= N_TEST + N_VAL + MIN_TRAIN

split = (
    ratings.withColumn("rn", F.row_number().over(newest_first))
    .withColumn("n", F.count("*").over(Window.partitionBy("userId")))
    .withColumn(
        "part",
        F.when(eligible & (F.col("rn") <= N_TEST), "test")
        .when(eligible & (F.col("rn") <= N_TEST + N_VAL), "val")
        .otherwise("train"),
    )
    .drop("rn", "n")
)
split.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(tbl("ratings_split"))
split = spark.table(tbl("ratings_split"))

# COMMAND ----------

# MAGIC %md ## Checks

# COMMAND ----------

parts = {r["part"]: r["count"] for r in split.groupBy("part").count().collect()}
print(parts)
assert sum(parts.values()) == ratings.count(), "split rows do not tie to bronze"

# Per user: newest train <= oldest val, newest val <= oldest test
b = split.groupBy("userId").pivot("part", ["train", "val", "test"]).agg(
    F.min("timestamp").alias("min"), F.max("timestamp").alias("max")
)
violations = b.filter((F.col("train_max") > F.col("val_min")) | (F.col("val_max") > F.col("test_min"))).count()
print(f"users whose held-out ratings predate their training ratings: {violations}")
assert violations == 0

held_out_users = split.filter("part = 'test'").select("userId").distinct().count()
print(f"users with a test set: {held_out_users:,} of {split.select('userId').distinct().count():,}")

# COMMAND ----------

# MAGIC %md ## Cold-start movies
# MAGIC Held-out movies the model never saw in training can't be recommended by any collaborative model. This caps achievable recall, so report it.

# COMMAND ----------

s = load_split()
for name, train, truth in [("val", s["train"], s["val"]), ("test", s["train_full"], s["test"])]:
    unseen = truth.join(train.select("movieId").distinct(), "movieId", "left_anti").count()
    print(f"{name}: {unseen} of {truth.count()} held-out ratings are for movies absent from training ({unseen / truth.count():.1%})")

# COMMAND ----------

# MAGIC %md ## Date ranges per part
# MAGIC The split is per user, not global, so one user's test ratings can be older than another user's training ratings. That's standard for per-user holdout and noted in the README's limitations.

# COMMAND ----------

display(
    split.groupBy("part").agg(
        F.from_unixtime(F.min("timestamp")).alias("first"),
        F.from_unixtime(F.max("timestamp")).alias("last"),
        F.count("*").alias("ratings"),
    )
)
