"""
app.py  –  Flask frontend for plant disease prediction

Runs entirely on the saved Spark ML model WITHOUT starting a full SparkSession
(which would eat RAM). Instead, we re-implement the same preprocessing in
plain Python/Pillow/NumPy, then load the best model via PySpark's local
model-load utility.

Usage:
    pip install flask pillow numpy
    python app.py

The page is served at http://127.0.0.1:5000
"""

import io
import json
import os
import time
import numpy as np

from flask import Flask, request, jsonify, render_template_string
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────
IMG_SIZE   = 64          # must match phase 2
MODEL_DIR  = "data/models"
LABEL_MAP  = "data/label_map.json"
RESULTS_F  = "data/model_results.json"

app = Flask(__name__)

# ── Load label map at startup ─────────────────────────────────────────────
label_map: dict = {}
if os.path.exists(LABEL_MAP):
    with open(LABEL_MAP) as f:
        label_map = json.load(f)

model_results: dict = {}
if os.path.exists(RESULTS_F):
    with open(RESULTS_F) as f:
        model_results = json.load(f)


# ── Lazy Spark session (only started when /predict is first hit) ──────────
_spark = None
_loaded_models = {}

def get_spark():
    global _spark
    if _spark is None:
        from config import create_spark
        _spark = create_spark("PlantDisease-Frontend")
        _spark.sparkContext.setLogLevel("ERROR")
    return _spark


def get_model(model_name: str):
    """Load a PipelineModel from disk, cache it in memory."""
    if model_name not in _loaded_models:
        from pyspark.ml import PipelineModel
        path = os.path.join(MODEL_DIR, model_name)
        _loaded_models[model_name] = PipelineModel.load(path)
    return _loaded_models[model_name]


def available_models():
    if not os.path.isdir(MODEL_DIR):
        return []
    return [d for d in os.listdir(MODEL_DIR)
            if os.path.isdir(os.path.join(MODEL_DIR, d))]


def preprocess_image(file_bytes: bytes):
    """Replicate Phase 2 preprocessing (Pillow only, no Spark needed)."""
    img = Image.open(io.BytesIO(file_bytes)).convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    pixels = np.array(img, dtype=np.float32).flatten() / 255.0
    return pixels.tolist()


# ── Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    models   = available_models()
    has_data = len(models) > 0
    best_model = ""
    if model_results:
        best_model = max(model_results, key=lambda k: model_results[k].get("accuracy", 0))
    return render_template_string(
        HTML_TEMPLATE,
        models=models,
        has_data=has_data,
        model_results=model_results,
        best_model=best_model,
        label_map=label_map,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    chosen_model = request.form.get("model", "")
    if not chosen_model:
        return jsonify({"error": "No model selected"}), 400

    model_path = os.path.join(MODEL_DIR, chosen_model)
    if not os.path.isdir(model_path):
        return jsonify({"error": f"Model '{chosen_model}' not found"}), 404

    file = request.files["image"]
    file_bytes = file.read()

    try:
        features = preprocess_image(file_bytes)
    except Exception as e:
        return jsonify({"error": f"Image processing failed: {e}"}), 400

    # Build a one-row Spark DataFrame and run through the pipeline
    spark = get_spark()
    from pyspark.ml.linalg import Vectors
    from pyspark.sql import Row

    row = Row(features=Vectors.dense(features), label="unknown",
              label_index=0.0, path="upload")
    one_row = spark.createDataFrame([row])

    try:
        model = get_model(chosen_model)
        t0 = time.time()
        pred_df = model.transform(one_row)
        elapsed = round((time.time() - t0) * 1000, 1)

        prediction_idx = str(int(pred_df.select("prediction").collect()[0][0]))
        label = label_map.get(prediction_idx, f"class_{prediction_idx}")

        # Probability vector (if model exposes it)
        try:
            prob_row = pred_df.select("probability").collect()[0][0]
            confidence = round(float(max(prob_row)) * 100, 1)
        except Exception:
            confidence = None

        return jsonify({
            "prediction"  : label,
            "class_index" : prediction_idx,
            "confidence"  : confidence,
            "model_used"  : chosen_model,
            "latency_ms"  : elapsed,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/results")
def results():
    return jsonify(model_results)


# ── HTML template (single-file, no separate static folder needed) ─────────

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Plant Disease Detector</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,700;1,9..144,300&display=swap');

  :root{
    --bg: #0e120e;
    --surface: #151a14;
    --border: #2a3328;
    --accent: #7bc96b;
    --accent2: #c8f0a0;
    --muted: #5a6e56;
    --text: #ddecd6;
    --danger: #e06c6c;
    --mono: 'DM Mono', monospace;
    --serif: 'Fraunces', Georgia, serif;
  }

  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:var(--mono);min-height:100vh;
       background-image:radial-gradient(ellipse 80% 60% at 50% -10%, #1e3a1a 0%, transparent 70%)}

  header{padding:2.5rem 2rem 1rem;border-bottom:1px solid var(--border);
         display:flex;align-items:baseline;gap:1rem}
  header h1{font-family:var(--serif);font-size:2rem;font-weight:300;color:var(--accent2);
             font-style:italic}
  header span{color:var(--muted);font-size:.75rem;letter-spacing:.12em;text-transform:uppercase}

  main{max-width:900px;margin:0 auto;padding:2rem 1.5rem;display:grid;
       grid-template-columns:1fr 1fr;gap:1.5rem}

  @media(max-width:680px){main{grid-template-columns:1fr}}

  .card{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:1.5rem}
  .card h2{font-family:var(--serif);font-size:1.1rem;font-weight:300;color:var(--accent2);
            margin-bottom:1.2rem;padding-bottom:.6rem;border-bottom:1px solid var(--border)}

  label{display:block;font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;
        color:var(--muted);margin-bottom:.4rem}

  select, input[type=file]{
    width:100%;background:var(--bg);border:1px solid var(--border);
    color:var(--text);font-family:var(--mono);font-size:.85rem;
    padding:.55rem .8rem;border-radius:4px;margin-bottom:1rem;
  }
  select:focus,input[type=file]:focus{outline:none;border-color:var(--accent)}

  button{
    width:100%;padding:.7rem;background:transparent;border:1px solid var(--accent);
    color:var(--accent);font-family:var(--mono);font-size:.85rem;letter-spacing:.06em;
    border-radius:4px;cursor:pointer;transition:background .2s,color .2s;
    text-transform:uppercase;
  }
  button:hover{background:var(--accent);color:#0e120e}
  button:disabled{opacity:.4;cursor:not-allowed}

  #drop-zone{
    border:1px dashed var(--border);border-radius:4px;padding:2rem 1rem;
    text-align:center;color:var(--muted);font-size:.8rem;margin-bottom:1rem;
    cursor:pointer;transition:border-color .2s;
  }
  #drop-zone.over{border-color:var(--accent);color:var(--accent)}
  #preview{max-width:100%;max-height:200px;margin:.8rem auto 0;display:block;
            border-radius:4px;border:1px solid var(--border)}

  .result-box{margin-top:1rem;padding:1rem;border-radius:4px;
               border:1px solid var(--border);background:var(--bg);min-height:80px;
               font-size:.85rem;line-height:1.8}
  .result-box .label-name{font-family:var(--serif);font-size:1.4rem;font-weight:700;
                           color:var(--accent);font-style:italic}
  .healthy .label-name{color:#7bc96b}
  .diseased .label-name{color:#e0b46c}
  .error-msg{color:var(--danger)}
  .dim{color:var(--muted);font-size:.75rem}

  /* ── metrics table ── */
  .metrics{grid-column:1/-1}
  table{width:100%;border-collapse:collapse;font-size:.78rem;margin-top:.5rem}
  th{color:var(--muted);text-transform:uppercase;letter-spacing:.09em;
     font-weight:400;padding:.4rem .6rem;border-bottom:1px solid var(--border);text-align:left}
  td{padding:.45rem .6rem;border-bottom:1px solid #1c221b}
  tr.best td{color:var(--accent2)}
  tr.best td:first-child::after{content:" ★";color:var(--accent)}
  .bar{height:6px;background:var(--border);border-radius:3px;margin-top:3px}
  .bar-fill{height:6px;background:var(--accent);border-radius:3px;transition:width .6s}

  /* spinner */
  .spin{display:inline-block;width:14px;height:14px;border:2px solid var(--border);
        border-top-color:var(--accent);border-radius:50%;animation:spin .6s linear infinite;
        vertical-align:middle;margin-right:.4rem}
  @keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>

<header>
  <h1>Plant Disease Detector</h1>
  <span>Spark MLlib · Pakistan Agriculture</span>
</header>

<main>

  <!-- ── Upload & Predict ──────────────────────────────────────────────── -->
  <div class="card">
    <h2>Predict Disease</h2>

    <label>Select Model</label>
    <select id="model-select">
      {% if not has_data %}
        <option value="">— run phases 2-4 first —</option>
      {% else %}
        {% for m in models %}
          <option value="{{ m }}" {% if m == best_model.replace(' ','_') %}selected{% endif %}>
            {{ m.replace('_',' ') }}
          </option>
        {% endfor %}
      {% endif %}
    </select>

    <label>Upload Image</label>
    <div id="drop-zone">Drop image here or click to browse</div>
    <input type="file" id="file-input" accept="image/*" style="display:none">
    <img id="preview" style="display:none" alt="preview">

    <button id="predict-btn" {% if not has_data %}disabled{% endif %}>
      Run Prediction
    </button>

    <div class="result-box" id="result-box">
      <span class="dim">Prediction will appear here …</span>
    </div>
  </div>

  <!-- ── Model Results ─────────────────────────────────────────────────── -->
  <div class="card">
    <h2>Model Performance</h2>
    {% if model_results %}
    <table>
      <thead>
        <tr>
          <th>Model</th><th>Accuracy</th><th>F1</th><th>Time (s)</th>
        </tr>
      </thead>
      <tbody>
        {% for name, m in model_results.items() %}
        <tr {% if name == best_model %}class="best"{% endif %}>
          <td>{{ name }}</td>
          <td>
            {{ "%.1f"|format(m.accuracy*100) }}%
            <div class="bar"><div class="bar-fill" style="width:{{ m.accuracy*100 }}%"></div></div>
          </td>
          <td>{{ "%.3f"|format(m.f1) }}</td>
          <td>{{ m.train_time_sec }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
    {% else %}
    <p class="dim">No results yet — run phase4_training.py first.</p>
    {% endif %}
  </div>

  <!-- ── Label map info ────────────────────────────────────────────────── -->
  {% if label_map %}
  <div class="card metrics">
    <h2>Classes ({{ label_map|length }})</h2>
    <div style="display:flex;flex-wrap:wrap;gap:.4rem;margin-top:.3rem">
      {% for idx, lbl in label_map.items() %}
      <span style="background:var(--bg);border:1px solid var(--border);
                   border-radius:3px;padding:.2rem .6rem;font-size:.72rem;color:var(--muted)">
        {{ lbl }}
      </span>
      {% endfor %}
    </div>
  </div>
  {% endif %}

</main>

<script>
const dropZone   = document.getElementById('drop-zone');
const fileInput  = document.getElementById('file-input');
const preview    = document.getElementById('preview');
const predictBtn = document.getElementById('predict-btn');
const resultBox  = document.getElementById('result-box');

// Drop zone
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('over');
  if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) loadFile(fileInput.files[0]); });

function loadFile(f) {
  const reader = new FileReader();
  reader.onload = e => {
    preview.src = e.target.result;
    preview.style.display = 'block';
    dropZone.textContent = f.name;
  };
  reader.readAsDataURL(f);
}

predictBtn.addEventListener('click', async () => {
  const file = fileInput.files[0];
  if (!file) { resultBox.innerHTML = '<span class="error-msg">Please select an image first.</span>'; return; }
  const model = document.getElementById('model-select').value;
  if (!model) { resultBox.innerHTML = '<span class="error-msg">Please select a model.</span>'; return; }

  predictBtn.disabled = true;
  resultBox.innerHTML = '<span class="spin"></span> Running inference …';

  const fd = new FormData();
  fd.append('image', file);
  fd.append('model', model);

  try {
    const res  = await fetch('/predict', { method:'POST', body: fd });
    const data = await res.json();
    if (data.error) {
      resultBox.innerHTML = `<span class="error-msg">Error: ${data.error}</span>`;
    } else {
      const isHealthy = data.prediction.toLowerCase().includes('healthy');
      const cls = isHealthy ? 'healthy' : 'diseased';
      resultBox.className = `result-box ${cls}`;
      resultBox.innerHTML = `
        <div class="label-name">${data.prediction}</div>
        ${data.confidence !== null ? `<div>Confidence: <b>${data.confidence}%</b></div>` : ''}
        <div class="dim">Model: ${data.model_used.replace(/_/g,' ')} · ${data.latency_ms} ms</div>
      `;
    }
  } catch(e) {
    resultBox.innerHTML = `<span class="error-msg">Request failed: ${e.message}</span>`;
  } finally {
    predictBtn.disabled = false;
  }
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("Starting Plant Disease Flask app on http://127.0.0.1:5000")
    app.run(debug=False, port=5000)