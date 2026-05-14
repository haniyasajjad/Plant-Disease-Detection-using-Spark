"""
Phase 3: Data Processing  (MongoDB → Spark)

Loads image metadata from MongoDB, joins with Parquet features,
handles class imbalance, splits, scales, and writes train/test Parquet.

Why the join approach:
  MongoDB holds metadata (path, label) – stored in Phase 1.
  Parquet holds feature vectors – stored in Phase 2.
  We load labels from Mongo, join on path, then proceed.

"""

import json
import os
from config import create_spark
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, StandardScaler
from pyspark.storagelevel import StorageLevel

PARQUET_FEATURES = "data/features.parquet"
TRAIN_OUT        = "data/train.parquet"
TEST_OUT         = "data/test.parquet"
TRAIN_RATIO      = 0.8
SEED             = 42


def undersample(df, label_col="label", seed=SEED):
    """Cap majority classes at 2× minority count to preserve more data."""
    counts = {r[label_col]: r["count"]
              for r in df.groupBy(label_col).count().collect()}
    if not counts:
        return df

    min_count = min(counts.values())
    cap = min_count * 2

    parts = []
    for cls, cnt in counts.items():
        cls_df = df.filter(F.col(label_col) == cls)
        if cnt > cap:
            cls_df = cls_df.sample(withReplacement=False,
                                   fraction=cap / cnt, seed=seed)
        parts.append(cls_df)

    result = parts[0]
    for p in parts[1:]:
        result = result.union(p)
    return result


def run():
    spark = create_spark()
    spark.sparkContext.setLogLevel("WARN")

    # ── 1. Load metadata FROM MongoDB (Phase 1 stored this) ────────────────
    print("[Phase 3] Loading metadata from MongoDB …")
    mongo_df = (
        spark.read
        .format("com.mongodb.spark.sql.DefaultSource")
        .option("uri", "mongodb://127.0.0.1:27017/plant_disease.images")
        .load()
        .select("path", "label")
        .filter(F.col("label").isNotNull())
    )

    print(f"[Phase 3] Records from MongoDB: {mongo_df.count()}")
    mongo_df.show(5, truncate=80)

    # ── 2. Load feature vectors from Parquet (computed in Phase 2) ─────────
    print("[Phase 3] Loading feature vectors from Parquet …")
    features_df = spark.read.parquet(PARQUET_FEATURES).select("path", "features")

    # ── 3. Join on path to get (label + features) in one DataFrame ─────────
    # MongoDB stores full file:// URIs; Parquet may too – normalise to basename
    strip_path = F.regexp_extract(F.col("path"), r"[^/]+$", 0)

    mongo_df    = mongo_df   .withColumn("filename", strip_path)
    features_df = features_df.withColumn("filename", strip_path)

    df = mongo_df.join(features_df, on="filename", how="inner") \
                 .select(
                     mongo_df["path"].alias("path"),
                     F.col("label"),
                     F.col("features")
                 )

    joined_count = df.count()
    print(f"[Phase 3] Rows after join: {joined_count}")

    if joined_count == 0:
        print("[Phase 3] WARNING: join produced 0 rows.")
        print("  This usually means Phase 2 has not been run yet.")
        print("  Run: python phase2_features.py  then retry.")
        spark.stop()
        return

    # ── 4. Handle class imbalance ───────────────────────────────────────────
    print("[Phase 3] Class distribution BEFORE balancing:")
    df.groupBy("label").count().orderBy("count", ascending=False).show(20, False)

    print("[Phase 3] Undersampling majority classes …")
    df = undersample(df)

    print("[Phase 3] Class distribution AFTER balancing:")
    df.groupBy("label").count().orderBy("count", ascending=False).show(20, False)

    # ── 5. StringIndexer – label → numeric index ────────────────────────────
    print("[Phase 3] Encoding labels …")
    indexer = StringIndexer(inputCol="label", outputCol="label_index",
                            handleInvalid="skip")
    indexer_model = indexer.fit(df)
    df = indexer_model.transform(df)

    # Save label map for frontend decoding
    os.makedirs("data", exist_ok=True)
    label_map = {str(i): lbl for i, lbl in enumerate(indexer_model.labels)}
    with open("data/label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"[Phase 3] Label map saved → data/label_map.json ({len(label_map)} classes)")

    # ── 6. Feature scaling ──────────────────────────────────────────────────
    print("[Phase 3] Applying StandardScaler …")
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features",
                            withMean=False, withStd=True)
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df) \
                     .drop("features") \
                     .withColumnRenamed("scaled_features", "features")

    # ── 7. Persist before split ─────────────────────────────────────────────
    df = df.persist(StorageLevel.MEMORY_AND_DISK)
    df.count()  # materialise

    # ── 8. Train / test split ───────────────────────────────────────────────
    print(f"[Phase 3] Splitting {TRAIN_RATIO*100:.0f}% / {(1-TRAIN_RATIO)*100:.0f}% …")
    train_df, test_df = df.randomSplit([TRAIN_RATIO, 1 - TRAIN_RATIO], seed=SEED)

    # ── 9. Write to Parquet ─────────────────────────────────────────────────
    print(f"[Phase 3] Writing train → {TRAIN_OUT}")
    train_df.write.mode("overwrite").parquet(TRAIN_OUT)

    print(f"[Phase 3] Writing test  → {TEST_OUT}")
    test_df.write.mode("overwrite").parquet(TEST_OUT)

    df.unpersist()

    t = spark.read.parquet(TRAIN_OUT).count()
    v = spark.read.parquet(TEST_OUT).count()
    print(f"[Phase 3] Train: {t} rows  |  Test: {v} rows")
    print("[Phase 3] Complete ✓")
    spark.stop()


if __name__ == "__main__":
    run()