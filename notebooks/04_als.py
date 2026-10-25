# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · ALS: tune, then compare with the baselines
# MAGIC Two ALS variants from Spark MLlib:
# MAGIC
# MAGIC - **Explicit:** predicts the star rating. Tuned on validation RMSE.
# MAGIC - **Implicit:** treats each rating as a "watched" signal with confidence = 1 + alpha × rating. Tuned on validation NDCG@10.
# MAGIC
# MAGIC The best of each is refit on train + val and scored on test, alongside the baselines from `03_baselines`. The point to check: a model with the best RMSE is not necessarily the best ranker.
# MAGIC
# MAGIC At 25M each fit takes minutes on Free Edition serverless, so the grid only covers the neighbourhood of the small-data winners. The last section times the chosen model on growing samples of the data.

# COMMAND ----------

# Dataset dropdown. Run this cell, pick small or 25m at the top, then run the rest.
try:
    dbutils.widgets.dropdown("dataset", "small", ["small", "25m"], "Dataset")
except NameError:  # outside Databricks
    pass

# COMMAND ----------

# MAGIC %run ./00_common

# COMMAND ----------

import itertools
import time

from pyspark.ml.recommendation import ALS

MAX_ITER = 10
if DATASET == "small":
    EXPLICIT_GRID = list(itertools.product([10, 30, 50], [0.05, 0.1, 0.2]))
    # A first pass (rank 10/50, reg 0.01/0.1, alpha 1/10/40) peaked at the grid edge: rank 50, reg 0.1, alpha 1.
    # Alpha 40 was worst everywhere, so it is dropped and the grid extends past each edge instead.
    IMPLICIT_GRID = list(itertools.product([10, 50, 100], [0.01, 0.1, 0.3], [0.5, 1, 10]))
else:
    EXPLICIT_GRID = [(10, 0.1), (50, 0.1)]
    IMPLICIT_GRID = [(50, 0.1, 1), (100, 0.1, 1), (100, 0.1, 0.5)]

s = load_split()
val_users = s["val"].select("userId").distinct()
test_users = s["test"].select("userId").distinct()


def fit_als(train, implicit, rank, reg, alpha=1.0):
    return ALS(
        userCol="userId", itemCol="movieId", ratingCol="rating", implicitPrefs=implicit,
        rank=rank, regParam=reg, alpha=alpha, maxIter=MAX_ITER, coldStartStrategy="drop", seed=SEED,
    ).fit(train)


def rmse(model, truth):
    """RMSE on held-out ratings. coldStartStrategy='drop' skips unseen users/movies, so report how many were dropped."""
    r = model.transform(truth).agg(
        F.sqrt(F.avg((F.col("prediction") - F.col("rating")) ** 2)).alias("rmse"), F.count("*").alias("scored")
    ).first()
    return r["rmse"], truth.count() - r["scored"]

# COMMAND ----------

# MAGIC %md ## Explicit ALS: tune on validation RMSE

# COMMAND ----------

explicit_runs = []
for rank, reg in EXPLICIT_GRID:
    t0 = time.time()
    val_rmse, dropped = rmse(fit_als(s["train"], False, rank, reg), s["val"])
    explicit_runs.append({"rank": rank, "reg": reg, "val_rmse": val_rmse, "dropped": dropped, "secs": round(time.time() - t0, 1)})
    print(explicit_runs[-1])

best_exp = min(explicit_runs, key=lambda r: r["val_rmse"])
display(pd.DataFrame(explicit_runs).sort_values("val_rmse"))

# COMMAND ----------

# MAGIC %md ## Implicit ALS: tune on validation NDCG@10

# COMMAND ----------

implicit_runs = []
for rank, reg, alpha in IMPLICIT_GRID:
    t0 = time.time()
    model = fit_als(s["train"], True, rank, reg, alpha)
    m = evaluate(als_topn(model, val_users, s["train"]), s["val"], s["train"], "als_implicit", "val")
    implicit_runs.append({"rank": rank, "reg": reg, "alpha": alpha, "val_ndcg": m["ndcg"], "val_recall": m["recall"],
                          "coverage": m["coverage"], "secs": round(time.time() - t0, 1)})
    print(implicit_runs[-1])

best_imp = max(implicit_runs, key=lambda r: r["val_ndcg"])
display(pd.DataFrame(implicit_runs).sort_values("val_ndcg", ascending=False))

# COMMAND ----------

# MAGIC %md ## Validation results for the chosen settings

# COMMAND ----------

exp_params = {"rank": best_exp["rank"], "reg": best_exp["reg"], "max_iter": MAX_ITER}
imp_params = {"rank": best_imp["rank"], "reg": best_imp["reg"], "alpha": best_imp["alpha"], "max_iter": MAX_ITER}

rows = []
exp_model = fit_als(s["train"], False, best_exp["rank"], best_exp["reg"])
r = evaluate(als_topn(exp_model, val_users, s["train"]), s["val"], s["train"], "als_explicit", "val", params=exp_params)
r["rmse"] = best_exp["val_rmse"]
rows.append(r)

imp_model = fit_als(s["train"], True, best_imp["rank"], best_imp["reg"], best_imp["alpha"])
rows.append(evaluate(als_topn(imp_model, val_users, s["train"]), s["val"], s["train"], "als_implicit", "val", params=imp_params))

# COMMAND ----------

# MAGIC %md ## Final: refit on train + val, score on test

# COMMAND ----------

exp_model = fit_als(s["train_full"], False, best_exp["rank"], best_exp["reg"])
test_rmse, dropped = rmse(exp_model, s["test"])
print(f"explicit ALS test RMSE {test_rmse:.4f} ({dropped} held-out ratings dropped as cold start)")
r = evaluate(als_topn(exp_model, test_users, s["train_full"]), s["test"], s["train_full"], "als_explicit", "test", params=exp_params)
r["rmse"] = test_rmse
rows.append(r)

t0 = time.time()
imp_model = fit_als(s["train_full"], True, best_imp["rank"], best_imp["reg"], best_imp["alpha"])
full_fit_secs = round(time.time() - t0, 1)
rows.append(evaluate(als_topn(imp_model, test_users, s["train_full"]), s["test"], s["train_full"], "als_implicit", "test", params=imp_params))
save_results(rows)

# COMMAND ----------

# MAGIC %md ## Comparison on test (all models)

# COMMAND ----------

display(
    spark.table(tbl("eval_results")).filter("split = 'test'")
    .select("model", "users", "precision", "recall", "ndcg", "map", "coverage", "pop_pct", "rmse", "params")
    .orderBy(F.desc("ndcg"))
)

# COMMAND ----------

# MAGIC %md ## Runtime vs data size
# MAGIC Fit time of the chosen implicit model on random samples of train + val. Both datasets write to one shared table, `movielens.als_runtimes`, so small and 25M sit side by side. Times include serverless start-up overhead, which dominates the small fits.

# COMMAND ----------

RUNTIME_COLS = "dataset STRING, fraction DOUBLE, ratings BIGINT, model STRING, params STRING, fit_secs DOUBLE, run_ts TIMESTAMP"
FRACTIONS = {"small": [], "25m": [0.01, 0.1, 0.5]}[DATASET]

runtimes = [{"dataset": DATASET, "fraction": 1.0, "ratings": s["train_full"].count(), "model": "als_implicit",
             "params": json.dumps(imp_params), "fit_secs": full_fit_secs}]
for frac in FRACTIONS:
    sample = s["train_full"].sample(fraction=frac, seed=SEED)
    t0 = time.time()
    fit_als(sample, True, best_imp["rank"], best_imp["reg"], best_imp["alpha"])
    runtimes.append({"dataset": DATASET, "fraction": frac, "ratings": sample.count(), "model": "als_implicit",
                     "params": json.dumps(imp_params), "fit_secs": round(time.time() - t0, 1)})
    print(runtimes[-1])

upsert(f"{CATALOG}.movielens.als_runtimes", runtimes, RUNTIME_COLS, ["dataset", "fraction"])
display(spark.table(f"{CATALOG}.movielens.als_runtimes").orderBy("ratings"))

# COMMAND ----------

# MAGIC %md ## Sanity check: one user's implicit-ALS recommendations

# COMMAND ----------

titles = spark.table(tbl("movies_bronze")).select("movieId", "title", "genres")
user = test_users.orderBy("userId").first()[0]
one_user = spark.createDataFrame([(user,)], "userId INT")
hits = s["test"].filter(F.col("userId") == user).select("movieId", F.lit(True).alias("in_test"))
print(f"user {user}: top 10 (in_test = the user actually rated it next)")
display(als_topn(imp_model, one_user, s["train_full"]).join(titles, "movieId").join(hits, "movieId", "left").orderBy("rank"))
