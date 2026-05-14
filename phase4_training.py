"""
Phase 4: Model Training – Spark MLlib
Trains Logistic Regression, Decision Tree, Random Forest, Naïve Bayes.
Compares accuracy / precision / recall / confusion matrix.
Saves each model to disk.

Memory notes (8 GB):
- Models trained sequentially (never two at once).
- Each model is unpersisted before the next begins.
- RandomForest uses conservative numTrees=20, maxDepth=5.
- All intermediate DFs read from Parquet (no recompute from images).
"""

import time
import json
import os
from config import create_spark
from pyspark.storagelevel import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier,
    NaiveBayes,
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StandardScaler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import functions as F

TRAIN_PARQUET = "data/train.parquet"
TEST_PARQUET  = "data/test.parquet"
MODEL_DIR     = "data/models"
RESULTS_FILE  = "data/model_results.json"


# ── helper ──────────────────────────────────────────────────────────────────
def evaluate(predictions, label_col="label_index", pred_col="prediction"):
    evaluator = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol=pred_col
    )
    accuracy  = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
    precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
    recall    = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
    f1        = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

    # Confusion matrix via RDD API
    rdd = predictions.select(pred_col, label_col) \
                     .rdd.map(lambda r: (float(r[0]), float(r[1])))
    metrics = MulticlassMetrics(rdd)
    cm = metrics.confusionMatrix().toArray().tolist()

    return {
        "accuracy" : round(accuracy,  4),
        "precision": round(precision, 4),
        "recall"   : round(recall,    4),
        "f1"       : round(f1,        4),
        "confusion_matrix": cm,
    }


def train_and_evaluate(spark, name, classifier, train_df, test_df):
    print(f"\n{'='*60}")
    print(f"[Phase 4] Training: {name}")
    print(f"{'='*60}")

    # StandardScaler in the pipeline keeps features well-conditioned
    scaler = StandardScaler(
        inputCol="features", outputCol="scaled_features",
        withMean=False, withStd=True   # withMean=False avoids dense conversion
    )
    classifier.setFeaturesCol("scaled_features")
    classifier.setLabelCol("label_index")

    pipeline = Pipeline(stages=[scaler, classifier])

    t0 = time.time()
    model = pipeline.fit(train_df)
    train_time = round(time.time() - t0, 2)
    print(f"  Training time: {train_time}s")

    predictions = model.transform(test_df)
    predictions.cache()

    metrics = evaluate(predictions)
    metrics["train_time_sec"] = train_time
    predictions.unpersist()

    # Save model
    model_path = os.path.join(MODEL_DIR, name.replace(" ", "_"))
    model.write().overwrite().save(model_path)
    print(f"  Saved → {model_path}")
    print(f"  Accuracy : {metrics['accuracy']}")
    print(f"  Precision: {metrics['precision']}")
    print(f"  Recall   : {metrics['recall']}")
    print(f"  F1       : {metrics['f1']}")

    return metrics


def run():
    spark = create_spark()
    spark.sparkContext.setLogLevel("WARN")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Load splits ──────────────────────────────────────────────────────────
    print("[Phase 4] Loading train / test splits …")
    train_df = spark.read.parquet(TRAIN_PARQUET).persist(StorageLevel.MEMORY_AND_DISK)
    test_df  = spark.read.parquet(TEST_PARQUET) .persist(StorageLevel.MEMORY_AND_DISK)

    n_classes = train_df.select("label_index").distinct().count()
    print(f"[Phase 4] Number of classes: {n_classes}")

    all_results = {}

    # ── Model 1: Logistic Regression ─────────────────────────────────────────
    lr = LogisticRegression(
        maxIter=50,
        regParam=0.1,
        elasticNetParam=0.0,
        family="multinomial",
    )
    all_results["Logistic Regression"] = train_and_evaluate(
        spark, "Logistic_Regression", lr, train_df, test_df
    )

    # ── Model 2: Decision Tree ────────────────────────────────────────────────
    dt = DecisionTreeClassifier(
        maxDepth=8,
        maxBins=32,
        impurity="gini",
    )
    all_results["Decision Tree"] = train_and_evaluate(
        spark, "Decision_Tree", dt, train_df, test_df
    )

    # ── Model 3: Random Forest ────────────────────────────────────────────────
    # Conservative settings to respect 8 GB RAM
    rf = RandomForestClassifier(
        numTrees=20,
        maxDepth=5,
        maxBins=32,
        seed=42,
    )
    all_results["Random Forest"] = train_and_evaluate(
        spark, "Random_Forest", rf, train_df, test_df
    )

    # ── Model 4: Naïve Bayes ─────────────────────────────────────────────────
    # NaiveBayes requires non-negative features.
    # StandardScaler with withMean=False keeps values ≥ 0 for our [0,1] input.
    nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
    all_results["Naive Bayes"] = train_and_evaluate(
        spark, "Naive_Bayes", nb, train_df, test_df
    )

    train_df.unpersist()
    test_df.unpersist()

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    header = f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Time(s)':>9}"
    print(header)
    print("-"*60)
    for name, m in all_results.items():
        print(
            f"{name:<22} {m['accuracy']:>9.4f} {m['precision']:>10.4f} "
            f"{m['recall']:>8.4f} {m['f1']:>8.4f} {m['train_time_sec']:>9.1f}"
        )

    # Best model
    best = max(all_results, key=lambda k: all_results[k]["accuracy"])
    print(f"\n★ Best model: {best}  (accuracy={all_results[best]['accuracy']})")


    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Phase 4] Results saved → {RESULTS_FILE}")
    print("[Phase 4] Complete ✓")
    spark.stop()


if __name__ == "__main__":
    run()