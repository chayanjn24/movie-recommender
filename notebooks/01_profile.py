# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Load and profile MovieLens
# MAGIC Gets the raw CSVs into the Volume, loads them into bronze Delta tables, then profiles the ratings: size, sparsity, data quality, time range, user activity, and the long tail of movie popularity.
# MAGIC
# MAGIC Pick the dataset in the dropdown at the top. The first cell creates the schema and Volume if needed, downloads the MovieLens zip from GroupLens, and unzips the 4 CSVs. If the download is blocked, upload the zip to the Volume by hand and rerun.

# COMMAND ----------

# Dataset dropdown. Run this cell, pick small or 25m at the top, then run the rest.
try:
    dbutils.widgets.dropdown("dataset", "small", ["small", "25m"], "Dataset")
except NameError:  # outside Databricks
    pass

# COMMAND ----------

# MAGIC %run ./00_common

# COMMAND ----------

# MAGIC %md ## Get the CSVs into the Volume

# COMMAND ----------

import os
import shutil
import urllib.request
import zipfile

ZIP = {"small": "ml-latest-small.zip", "25m": "ml-25m.zip"}[DATASET]
CSVS = ["ratings.csv", "movies.csv", "tags.csv", "links.csv"]

if VOL.startswith("/Volumes/"):
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.raw")

if not all(os.path.exists(f"{VOL}/{c}") for c in CSVS):
    if not os.path.exists(f"{VOL}/{ZIP}"):
        url = f"https://files.grouplens.org/datasets/movielens/{ZIP}"
        print(f"downloading {url} ...")
        try:
            with urllib.request.urlopen(url, timeout=600) as src, open(f"{VOL}/{ZIP}", "wb") as dst:
                shutil.copyfileobj(src, dst)
        except Exception as e:
            raise RuntimeError(f"Download failed ({e}). Upload {ZIP} to {VOL} by hand, then rerun this cell.")
    # Stream each CSV out of the zip; ratings.csv in ml-25m is ~650 MB, too big to read into memory at once
    with zipfile.ZipFile(f"{VOL}/{ZIP}") as zf:
        for name in zf.namelist():
            if os.path.basename(name) in CSVS:
                with zf.open(name) as src, open(f"{VOL}/{os.path.basename(name)}", "wb") as dst:
                    shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)

print({c: f"{os.path.getsize(f'{VOL}/{c}') / 1e6:.1f} MB" for c in CSVS})

# COMMAND ----------

# MAGIC %md ## Load bronze tables

# COMMAND ----------

for name in ["ratings", "movies", "tags", "links"]:
    # escape='"' because titles like "Shawshank Redemption, The" are quoted and inner quotes are doubled
    df = spark.read.csv(f"{VOL}/{name}.csv", header=True, inferSchema=True, escape='"')
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(tbl(f"{name}_bronze"))
    print(f"{name}_bronze: {spark.table(tbl(f'{name}_bronze')).count():,} rows")

ratings = spark.table(tbl("ratings_bronze"))
movies = spark.table(tbl("movies_bronze"))

# COMMAND ----------

# MAGIC %md ## Size and sparsity

# COMMAND ----------

n = ratings.count()
n_users = ratings.select("userId").distinct().count()
n_items = ratings.select("movieId").distinct().count()
sparsity = 1 - n / (n_users * n_items)
print(f"ratings: {n:,} | users: {n_users:,} | rated movies: {n_items:,} of {movies.count():,} in catalogue")
print(f"sparsity: {sparsity:.2%} of the user x movie matrix is empty")

# COMMAND ----------

# MAGIC %md ## Data quality

# COMMAND ----------

ratings.select([F.sum(F.col(c).isNull().cast("int")).alias(f"null_{c}") for c in ratings.columns]).show()
dupes = n - ratings.select("userId", "movieId").distinct().count()
orphans = ratings.join(movies, "movieId", "left_anti").count()
print(f"duplicate (user, movie) pairs: {dupes} | ratings for movies missing from movies.csv: {orphans}")
display(ratings.groupBy("rating").count().orderBy("rating"))

# COMMAND ----------

# MAGIC %md ## Time range

# COMMAND ----------

ratings.agg(
    F.from_unixtime(F.min("timestamp")).alias("first_rating"),
    F.from_unixtime(F.max("timestamp")).alias("last_rating"),
).show()
display(ratings.groupBy(F.year(F.from_unixtime("timestamp")).alias("year")).count().orderBy("year"))

# COMMAND ----------

# MAGIC %md ## Activity: ratings per user and per movie

# COMMAND ----------

per_user = ratings.groupBy("userId").count()
per_item = ratings.groupBy("movieId").count()
print("ratings per user")
per_user.select("count").summary("min", "25%", "50%", "75%", "90%", "max").show()
print("ratings per movie")
per_item.select("count").summary("min", "25%", "50%", "75%", "90%", "max").show()

# COMMAND ----------

# MAGIC %md ## Long tail
# MAGIC How concentrated are ratings on a few popular movies? This is why popularity is a hard baseline to beat, and why coverage matters.

# COMMAND ----------

import matplotlib.pyplot as plt

counts = per_item.toPandas()["count"].sort_values(ascending=False).to_numpy()
cum_share = counts.cumsum() / counts.sum()
top10_share = cum_share[int(len(counts) * 0.1) - 1]
rare_share = (counts < 5).mean()
print(f"top 10% of movies hold {top10_share:.1%} of ratings | {rare_share:.1%} of movies have fewer than 5 ratings")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(np.arange(1, len(counts) + 1) / len(counts), cum_share)
ax.axvline(0.1, linestyle="--", linewidth=1)
ax.set_xlabel("share of movies (most rated first)")
ax.set_ylabel("cumulative share of ratings")
ax.set_title("Long tail of movie popularity")
plt.show()

# COMMAND ----------

# MAGIC %md ## Genres

# COMMAND ----------

genres = movies.select("movieId", F.explode(F.split("genres", r"\|")).alias("genre"))
display(ratings.join(genres, "movieId").groupBy("genre").count().orderBy(F.desc("count")))

# COMMAND ----------

# MAGIC %md ## Summary (copy into `docs/data_dictionary.md`)

# COMMAND ----------

print(f"""ratings {n:,}; users {n_users:,}; rated movies {n_items:,}; sparsity {sparsity:.2%}
duplicates {dupes}; orphan ratings {orphans}
top 10% of movies = {top10_share:.1%} of ratings; {rare_share:.1%} of movies have < 5 ratings""")
