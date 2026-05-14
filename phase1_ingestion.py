"""
Phase 1: Data Ingestion  (Spark → MongoDB)

Extracts label from the filename

PlantVillage filename pattern:
    <uuid>___<CLASS_CODE> <number>.JPG
    e.g. fc5c5672-d1e5-4374-bd99-608d95609a7f___RS_HL 0474.JPG
                                                 ^^^^^
                                                 label

The regex  ___([^_ ][^ ]+)  captures everything between ___ and the
first space, giving e.g. "RS_HL".

"""

from config import create_spark
from pyspark.sql.functions import regexp_extract, col


LABEL_REGEX = r"___([A-Za-z0-9]+(?:_[A-Za-z0-9]+)*)"

spark = create_spark()

print("Loading images...")
df = spark.read.format("binaryFile") \
    .option("recursiveFileLookup", "true") \
    .load("data/sample_data")

print(f"Total images: {df.count()}")

# Extract label from filename
df = df.withColumn(
    "label",
    regexp_extract(col("path"), LABEL_REGEX, 1)
)

# Replace empty string (no match) with None so NULLs are honest
from pyspark.sql.functions import when, lit
df = df.withColumn(
    "label",
    when(col("label") == "", lit(None)).otherwise(col("label"))
)

# Keep only metadata
df = df.select("path", "label", "length", "modificationTime")

# Sanity check – show label distribution
print("\nLabel distribution:")
df.groupBy("label").count().orderBy("count", ascending=False).show(20, False)

df.show(10, False)

# Repartition for parallel processing
df = df.repartition(8)

print("Writing metadata to MongoDB...")
df.write \
    .format("com.mongodb.spark.sql.DefaultSource") \
    .mode("overwrite") \
    .option("uri", "mongodb://127.0.0.1:27017/plant_disease.images") \
    .save()

print("Phase 1 complete: metadata stored in MongoDB.")
spark.stop()