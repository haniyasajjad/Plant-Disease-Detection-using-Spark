"""
partition_analysis.py  –  Advanced requirement: analyse impact of partitioning

Reads the already-scaled train/test Parquet splits (from Phase 3) and
measures how repartitioning affects shuffle and training time.

"""

import time
import json
from config import create_spark
from pyspark.storagelevel import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

TRAIN_PARQUET = "data/train.parquet"
TEST_PARQUET  = "data/test.parquet"
RESULTS       = "data/partition_analysis.json"

# Test these partition counts – keep modest for 8 GB
PARTITION_COUNTS = [2, 4, 8, 16]


def run_experiment(spark, n_partitions: int) -> dict:
    """
    Load the pre-scaled splits, repartition to n_partitions,
    train a LogisticRegression, return timing + accuracy.
    """
    # Load pre-scaled data (features already normalised in Phase 3)
    train_df = (
        spark.read.parquet(TRAIN_PARQUET)
        .repartition(n_partitions)
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    test_df = (
        spark.read.parquet(TEST_PARQUET)
        .repartition(n_partitions)
        .persist(StorageLevel.MEMORY_AND_DISK)
    )

    train_count = train_df.count()
    test_count  = test_df.count()

    # Guard: too few rows to train safely
    if train_count < 10:
        train_df.unpersist()
        test_df.unpersist()
        return {
            "n_partitions"  : n_partitions,
            "train_rows"    : train_count,
            "test_rows"     : test_count,
            "train_time_sec": None,
            "accuracy"      : None,
            "skipped"       : True,
            "reason"        : f"Only {train_count} training rows – too few",
        }

    # Simple LR – no scaler needed, data already scaled
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label_index",
        maxIter=20,
        regParam=0.1,
        family="multinomial",
    )
    pipeline = Pipeline(stages=[lr])

    t0 = time.time()
    model = pipeline.fit(train_df)
    train_time = round(time.time() - t0, 2)

    preds = model.transform(test_df)
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label_index",
        predictionCol="prediction",
        metricName="accuracy",
    )
    accuracy = round(evaluator.evaluate(preds), 4)

    train_df.unpersist()
    test_df.unpersist()

    return {
        "n_partitions"  : n_partitions,
        "train_rows"    : train_count,
        "test_rows"     : test_count,
        "train_time_sec": train_time,
        "accuracy"      : accuracy,
        "skipped"       : False,
    }


def run():
    spark = create_spark("PartitionAnalysis")
    spark.sparkContext.setLogLevel("WARN")

    print("[Partition Analysis] Using pre-scaled Phase 3 splits (no re-scaling needed)\n")

    results = []
    for n in PARTITION_COUNTS:
        print(f"[Partition Analysis] Testing n_partitions={n} …")
        r = run_experiment(spark, n)
        results.append(r)
        if r["skipped"]:
            print(f"  SKIPPED – {r['reason']}")
        else:
            print(f"  train_rows={r['train_rows']}  "
                  f"train_time={r['train_time_sec']}s  "
                  f"accuracy={r['accuracy']}")

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 58)
    print("PARTITION IMPACT ANALYSIS")
    print("=" * 58)
    print(f"{'Partitions':>12} {'Train rows':>11} {'Train(s)':>10} {'Accuracy':>10}")
    print("-" * 48)
    for r in results:
        if r["skipped"]:
            print(f"{r['n_partitions']:>12} {r['train_rows']:>11}   {'SKIPPED':>18}")
        else:
            print(
                f"{r['n_partitions']:>12} "
                f"{r['train_rows']:>11} "
                f"{r['train_time_sec']:>10.2f} "
                f"{r['accuracy']:>10.4f}"
            )

    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Partition Analysis] Results saved → {RESULTS}")
    spark.stop()


if __name__ == "__main__":
    run()