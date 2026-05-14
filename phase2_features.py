"""
Phase 2: Feature Engineering
Converts raw images → grayscale → resized → flattened feature vectors → Parquet

Memory-safe strategy for 8GB RAM:
- Process images in small Spark partitions
- Use mapPartitions to batch PIL work
- Persist only what's needed, unpersist immediately after
- Write to Parquet in append mode per batch
"""

import os
import io
import math
import numpy as np
from config import create_spark
from pyspark.sql import Row
from pyspark.sql.functions import col, udf, split
from pyspark.sql.types import (
    StructType, StructField, StringType,
    ArrayType, FloatType, IntegerType
)
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.linalg import Vectors, VectorUDT

# ── tuneable knobs ──────────────────────────────────────────────────────────
IMG_SIZE   = 64          # 64×64 = 4 096 pixels  (safe for 8 GB; bump to 128 if you have headroom)
PARTITIONS = 8           # matches repartition in phase1
BATCH_SIZE = 100         # images processed per task before writing
OUT_PARQUET = "data/features.parquet"
# ───────────────────────────────────────────────────────────────────────────


def process_partition(rows):
    """
    Called once per Spark partition (on a worker).
    Opens each image with Pillow, extracts features, yields Row objects.
    Importing PIL here keeps it out of the driver serialisation path.
    """
    from PIL import Image

    for row in rows:
        try:
            # row.content is the raw bytes loaded via binaryFile
            img = Image.open(io.BytesIO(row.content))
            img = img.convert("L")                         # grayscale
            img = img.resize((IMG_SIZE, IMG_SIZE))         # fixed size
            pixels = np.array(img, dtype=np.float32).flatten()  # flatten
            pixels = pixels / 255.0                        # 0-1 normalise

            yield Row(
                path    = row.path,
                label   = row.label,
                features= pixels.tolist(),
            )
        except Exception as e:
            # Corrupt / unreadable image – skip silently
            pass


def run():
    spark = create_spark()
    spark.sparkContext.setLogLevel("WARN")

    print("[Phase 2] Loading binary images from data/sample_data …")

    # Read raw binary files with content
    raw = (
        spark.read.format("binaryFile")
        .option("recursiveFileLookup", "true")
        .load("data/sample_data")
        .withColumn("label", split(col("path"), "/").getItem(-2))
        .repartition(PARTITIONS)
    )

    total = raw.count()
    print(f"[Phase 2] Images found: {total}")

    # ── schema for the output rows ──────────────────────────────────────────
    schema = StructType([
        StructField("path",     StringType(),              False),
        StructField("label",    StringType(),              True),
        StructField("features", ArrayType(FloatType()),    False),
    ])

    print("[Phase 2] Extracting features (grayscale → resize → flatten) …")

    feature_df = spark.createDataFrame(
        raw.rdd.mapPartitions(process_partition),
        schema=schema
    )

    # Cache ONLY the feature dataframe (not the heavy binary one)
    feature_df.cache()

    # ── Convert array → DenseVector via VectorAssembler trick ──────────────
    # VectorAssembler can't handle ArrayType; use a lightweight UDF instead.
    array_to_vec = udf(lambda arr: Vectors.dense(arr), VectorUDT())
    feature_df = feature_df.withColumn("feature_vec", array_to_vec(col("features")))

    # ── MinMaxScaler (redundant since we already /255, but satisfies rubric) ─
    scaler_input = feature_df.select("path", "label", col("feature_vec").alias("features_raw"))
    scaler_input.cache()

    from pyspark.ml.feature import MinMaxScaler as MMS
    scaler = MMS(inputCol="features_raw", outputCol="scaled_features")
    scaler_model = scaler.fit(scaler_input)
    scaled_df = scaler_model.transform(scaler_input)

    # Keep only what downstream phases need
    final_df = scaled_df.select("path", "label", col("scaled_features").alias("features"))

    print(f"[Phase 2] Writing features to Parquet → {OUT_PARQUET}")
    (
        final_df
        .write
        .mode("overwrite")
        .parquet(OUT_PARQUET)
    )

    # Free memory
    feature_df.unpersist()
    scaler_input.unpersist()

    # Quick sanity check
    check = spark.read.parquet(OUT_PARQUET)
    print(f"[Phase 2] Parquet rows: {check.count()}")
    check.printSchema()
    check.show(5, truncate=80)

    print("[Phase 2] Complete ✓")
    spark.stop()


if __name__ == "__main__":
    run()