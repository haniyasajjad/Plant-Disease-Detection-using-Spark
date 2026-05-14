"""
config.py  –  Memory-safe Spark session for 8 GB MacBook

Key settings explained:
  driver.memory 3g         – leaves ~5 GB for macOS + MongoDB + overhead
  executor.memory 3g       – same JVM process in local mode, so totals ~3 GB heap
  memory.fraction 0.6      – 60% of heap for execution+storage (default 0.6)
  memory.storageFraction   – of that 60%, only 40% pinned for caching
  sql.shuffle.partitions 8 – 8 is sensible for a ~2k-image sample
  kryo serialization       – much faster and smaller than Java serialisation
  off-heap disabled        – keeps memory accounting simple
"""

from pyspark.sql import SparkSession


def create_spark(app_name="PlantDiseaseSpark"):
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")

        # MongoDB connector
        .config("spark.jars.packages",
                "org.mongodb.spark:mongo-spark-connector_2.12:2.4.0")

        # ── Memory ─────────────────────────────────────────────────────────
        .config("spark.driver.memory",              "3g")
        .config("spark.executor.memory",            "3g")
        .config("spark.memory.fraction",            "0.6")
        .config("spark.memory.storageFraction",     "0.4")

        # ── Serialisation ──────────────────────────────────────────────────
        .config("spark.serializer",
                "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max",  "512m")

        # ── Shuffle / partitions ───────────────────────────────────────────
        .config("spark.sql.shuffle.partitions",     "8")
        .config("spark.default.parallelism",        "8")

        # ── Misc stability ────────────────────────────────────────────────
        .config("spark.network.timeout",            "300s")
        .config("spark.sql.files.maxPartitionBytes","67108864")   # 64 MB
        .config("spark.driver.maxResultSize",       "1g")

        # ── UI on an alternate port (avoids conflict if 4040 is taken) ────
        .config("spark.ui.port", "4041")

        .getOrCreate()
    )