# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Shared config and evaluation helpers
# MAGIC Not run on its own. Every other notebook loads it with `%run ./00_common`, so keep all notebooks in the same folder.
# MAGIC
# MAGIC - **Config:** the dataset dropdown (`small` / `25m`), catalog, schema, Volume path, K, seed, scored-user sample
# MAGIC - **`load_split()`:** the train / val / test tables written by `02_split`
# MAGIC - **`topn_from_scores()`:** turns (user, movie, score) rows into a top-K list, excluding movies the user already rated
# MAGIC - **`als_topn()`:** top-K from an ALS model's factors (`recommendForAllUsers` is blocked on serverless + Unity Catalog)
# MAGIC - **`evaluate()`:** Precision@K, Recall@K, NDCG@K, MAP@K, catalogue coverage, popularity bias
# MAGIC - **`save_results()`:** writes metric rows to the `eval_results` Delta table

# COMMAND ----------

import json
import math

import numpy as np
import pandas as pd
from pyspark.sql import Window
from pyspark.sql import functions as F

# Dataset dropdown, shown at the top of whichever notebook runs this one.
# small = ml-latest-small (100K ratings), 25m = ml-25m (25M ratings). Each lives in its own schema.
try:
    dbutils.widgets.dropdown("dataset", "small", ["small", "25m"], "Dataset")
    DATASET = dbutils.widgets.get("dataset")
except NameError:  # outside Databricks
    DATASET = "small"

CATALOG = "workspace"
SCHEMA = {"small": "movielens", "25m": "movielens_25m"}[DATASET]
VOL = f"/Volumes/{CATALOG}/{SCHEMA}/raw"
K = 10
SEED = 42
# At 25M, models train on everything but are scored on a fixed sample of users to keep scoring affordable.
EVAL_USERS = {"small": None, "25m": 10_000}[DATASET]
print(f"dataset: {DATASET} | schema: {CATALOG}.{SCHEMA} | scored users: {EVAL_USERS or 'all'}")


def tbl(name):
    return f"{CATALOG}.{SCHEMA}.{name}"

# COMMAND ----------

def load_split():
    """train / val / test from 02_split. train_full = train + val, used to fit the final models.

    With EVAL_USERS set, val and test keep only a fixed hash-ordered sample of users; train stays complete.
    """
    df = spark.table(tbl("ratings_split"))
    s = {
        "train": df.filter("part = 'train'"),
        "val": df.filter("part = 'val'"),
        "test": df.filter("part = 'test'"),
        "train_full": df.filter("part != 'test'"),
    }
    if EVAL_USERS:
        users = (s["test"].select("userId").distinct()
                 .orderBy(F.xxhash64("userId", F.lit(SEED)), "userId").limit(EVAL_USERS))
        s["val"] = s["val"].join(users, "userId")
        s["test"] = s["test"].join(users, "userId")
    return s


def topn_from_scores(scores, seen, k=K):
    """scores: userId, movieId, score. Drops movies in `seen`, keeps the k best per user (ties broken by movieId)."""
    w = Window.partitionBy("userId").orderBy(F.desc("score"), F.asc("movieId"))
    return (
        scores.join(seen.select("userId", "movieId"), ["userId", "movieId"], "left_anti")
        .withColumn("rank", F.row_number().over(w))
        .filter(F.col("rank") <= k)
        .select("userId", "movieId", "rank")
    )


def als_topn(model, users, seen, k=K, max_cells=20_000_000):
    """Top-k unseen movies per user from ALS factors, scored on the driver with numpy.

    recommendForAllUsers fails on serverless (UC_COMMAND_NOT_SUPPORTED), and a Spark join of
    users x items x rank rows is too large, so the dot products run in numpy chunks instead.
    Each chunk's score matrix holds at most max_cells floats (about 80 MB).
    """
    users = users.select("userId").distinct()
    uf = model.userFactors.join(users.withColumnRenamed("userId", "id"), "id").toPandas()
    itf = model.itemFactors.toPandas()
    U = np.vstack(uf["features"].values).astype(np.float32)
    I = np.vstack(itf["features"].values).astype(np.float32)
    uid, iid = uf["id"].to_numpy(), itf["id"].to_numpy()
    user_pos = pd.Series(np.arange(len(uid)), index=uid)
    item_pos = pd.Series(np.arange(len(iid)), index=iid)
    chunk = max(1, max_cells // len(iid))

    s = seen.join(users, "userId").select("userId", "movieId").toPandas()
    s = s[s["userId"].isin(user_pos.index) & s["movieId"].isin(item_pos.index)]
    seen_u = user_pos[s["userId"]].to_numpy()
    seen_i = item_pos[s["movieId"]].to_numpy()

    out = []
    for start in range(0, len(uid), chunk):
        S = U[start:start + chunk] @ I.T
        m = (seen_u >= start) & (seen_u < start + chunk)
        S[seen_u[m] - start, seen_i[m]] = -np.inf
        top = np.argpartition(-S, k, axis=1)[:, :k]
        top = np.take_along_axis(top, np.argsort(-np.take_along_axis(S, top, 1), axis=1), 1)
        out.append(pd.DataFrame({
            "userId": np.repeat(uid[start:start + chunk], k).astype("int64"),
            "movieId": iid[top].ravel().astype("int64"),
            "rank": np.tile(np.arange(1, k + 1), top.shape[0]).astype("int64"),
        }))
    return spark.createDataFrame(pd.concat(out, ignore_index=True))

# COMMAND ----------

def evaluate(recs, truth, train, model_name, split, k=K, params=None):
    """Score top-k recs against held-out movies. Every held-out movie counts as relevant ("watched next").

    Users in `truth` with no recs score 0. Returns one dict of averaged metrics.
    """
    recs = recs.filter(F.col("rank") <= k)
    rel = truth.select("userId", "movieId").distinct().withColumn("hit", F.lit(1))
    n_rel = rel.groupBy("userId").agg(F.count("*").alias("n_rel"))

    w = Window.partitionBy("userId").orderBy("rank").rowsBetween(Window.unboundedPreceding, 0)
    per_user_recs = (
        recs.join(rel, ["userId", "movieId"], "left").fillna(0, ["hit"])
        .withColumn("cum_hits", F.sum("hit").over(w))
        .groupBy("userId")
        .agg(
            F.sum("hit").alias("hits"),
            F.sum(F.col("hit") / F.log2(F.col("rank") + 1)).alias("dcg"),
            F.sum(F.col("hit") * F.col("cum_hits") / F.col("rank")).alias("ap_sum"),
        )
    )
    idcg = spark.createDataFrame(
        [(n, sum(1 / math.log2(i + 1) for i in range(1, n + 1))) for n in range(1, k + 1)], "n_cap INT, idcg DOUBLE"
    )
    per_user = (
        n_rel.join(per_user_recs, "userId", "left").fillna(0, ["hits", "dcg", "ap_sum"])
        .withColumn("n_cap", F.least(F.col("n_rel"), F.lit(k)).cast("int"))
        .join(F.broadcast(idcg), "n_cap")
        .select(
            (F.col("hits") / k).alias("precision"),
            (F.col("hits") / F.col("n_rel")).alias("recall"),
            (F.col("dcg") / F.col("idcg")).alias("ndcg"),
            (F.col("ap_sum") / F.col("n_cap")).alias("map"),
        )
    )
    m = per_user.agg(
        F.count("*").alias("users"), *[F.avg(c).alias(c) for c in ["precision", "recall", "ndcg", "map"]]
    ).first().asDict()

    # Coverage: share of trainable movies that appear in anyone's top-k.
    # Popularity bias: mean popularity percentile of recommended movies (1.0 = most-rated movie).
    pop = train.groupBy("movieId").count().withColumn("pop_pct", F.percent_rank().over(Window.orderBy("count")))
    m["coverage"] = recs.select("movieId").distinct().count() / pop.count()
    m["pop_pct"] = recs.join(pop, "movieId", "left").agg(F.avg("pop_pct")).first()[0]
    m.update(model=model_name, split=split, k=k, params=json.dumps(params or {}))
    return m

# COMMAND ----------

import datetime

RESULT_COLS = "model STRING, split STRING, k INT, users BIGINT, precision DOUBLE, recall DOUBLE, ndcg DOUBLE, " \
              "map DOUBLE, coverage DOUBLE, pop_pct DOUBLE, rmse DOUBLE, params STRING, run_ts TIMESTAMP"


def upsert(table, rows, schema, keys):
    """Replace rows sharing `keys` with `rows` and keep the rest. For small result tables only: it collects to the driver."""
    cols = [c.split()[0] for c in schema.split(", ")]
    now = datetime.datetime.now()
    new_keys = {tuple(r[k] for k in keys) for r in rows}
    kept = []
    if spark.catalog.tableExists(table):
        kept = [tuple(r[c] for c in cols) for r in spark.table(table).select(*cols).collect()
                if tuple(r[k] for k in keys) not in new_keys]
    new = [tuple(now if c == "run_ts" else r.get(c) for c in cols) for r in rows]
    (spark.createDataFrame(kept + new, schema)
     .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table))


def save_results(rows):
    """Upsert metric rows into eval_results, one row per (model, split). Rerunning a notebook replaces its rows."""
    upsert(tbl("eval_results"), rows, RESULT_COLS, ["model", "split"])
