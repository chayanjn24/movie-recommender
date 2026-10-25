# Data dictionary

Source: [MovieLens](https://grouplens.org/datasets/movielens/) by GroupLens Research, University of Minnesota. The data is not stored in this repo; `01_profile` downloads it.

| Dataset | Download | Ratings | Users | Movies rated | Date range | Sparsity | Top 10% of movies hold |
|---|---|---|---|---|---|---|---|
| `ml-latest-small` (dev) | [ml-latest-small.zip](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip) (978 KB) | 100,836 | 610 | 9,724 | 1996-03-29 → 2018-09-24 | 98.30% | 60.0% of ratings |
| `ml-25m` (scale) | [ml-25m.zip](https://files.grouplens.org/datasets/movielens/ml-25m.zip) (262 MB) | 25,000,095 | 162,541 | 59,047 of 62,423 | 1995-01-09 → 2019-11-21 | 99.74% | 94.0% of ratings |

Both have no nulls, no duplicate (user, movie) pairs and no ratings for movies missing from `movies.csv`. Every user has at least 20 ratings.

## Raw files (same columns in both datasets)

| File | Columns | Notes |
|---|---|---|
| `ratings.csv` | `userId`, `movieId`, `rating`, `timestamp` | Rating 0.5–5.0 in half stars. Timestamp in Unix seconds (UTC). |
| `movies.csv` | `movieId`, `title`, `genres` | Title includes the year, e.g. `Toy Story (1995)`. Genres are pipe-separated. |
| `tags.csv` | `userId`, `movieId`, `tag`, `timestamp` | Free-text user tags. Loaded, not used by the models. |
| `links.csv` | `movieId`, `imdbId`, `tmdbId` | IDs on IMDb and TMDB. Loaded, not used by the models. |

## Tables written on Databricks

Catalog `workspace`. Schema `movielens` for the small dataset, `movielens_25m` for 25M. The raw files sit in the Volume `/Volumes/workspace/<schema>/raw`.

| Table | Written by | Contents |
|---|---|---|
| `ratings_bronze`, `movies_bronze`, `tags_bronze`, `links_bronze` | `01_profile` | The raw CSVs as Delta tables (inferred column types). |
| `ratings_split` | `02_split` | `ratings_bronze` plus `part` = `train` / `val` / `test`. Test = each user's last 5 ratings, val = the 5 before. |
| `item_sims` | `03_baselines` | Item-kNN cosine similarities (small dataset only). |
| `eval_results` | `03_baselines`, `04_als` | One row per (model, split): precision, recall, NDCG, MAP at 10, coverage, popularity bias, RMSE, params. |
| `als_runtimes` | `04_als` (25M run) | Implicit ALS fit time on random 1%, 10%, 50% and 100% samples of train + val. Stored in `movielens`. |
