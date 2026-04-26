import csv
import os
import pickle
from datetime import datetime

from flask import Flask, jsonify, request

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "pipeline.pkl")
PRIMARY_LOG_PATH = os.path.join(BASE_DIR, "predictions_log.csv")
FALLBACK_LOG_PATH = os.path.join(BASE_DIR, "logs", "predictions_log.csv")
FRONTEND_PATH = os.path.join(BASE_DIR, "..", "frontend")

app = Flask(__name__, static_folder=FRONTEND_PATH, static_url_path="/")
SESSION_STATS = {
    "counts": {"bullying": 0, "offensive": 0, "neutral": 0},
    "total": 0,
}


def load_model():
    """Load the saved ML pipeline from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Trained model not found. Run backend/train_model.py first."
        )
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


model_pipeline = load_model()


def resolve_log_path() -> str:
    """Use the default log when writable, otherwise fall back to a runtime log."""
    for path in (PRIMARY_LOG_PATH, FALLBACK_LOG_PATH):
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "a", newline="", encoding="utf-8"):
                pass
            return path
        except PermissionError:
            continue

    raise PermissionError(
        f"Unable to write prediction logs to either '{PRIMARY_LOG_PATH}' or '{FALLBACK_LOG_PATH}'."
    )


ACTIVE_LOG_PATH = resolve_log_path()


def log_prediction(text: str, prediction: str, confidence: float):
    """Store prediction details in a CSV file."""
    is_new_file = not os.path.exists(ACTIVE_LOG_PATH) or os.path.getsize(ACTIVE_LOG_PATH) == 0
    with open(ACTIVE_LOG_PATH, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if is_new_file:
            writer.writerow(["timestamp", "text", "prediction", "confidence"])
        writer.writerow([datetime.utcnow().isoformat(), text, prediction, f"{confidence:.2f}"])


def record_session_prediction(prediction: str):
    """Update in-memory counts for the current server run only."""
    label = prediction.strip().lower()
    if label in SESSION_STATS["counts"]:
        SESSION_STATS["counts"][label] += 1
        SESSION_STATS["total"] += 1


@app.route("/", methods=["GET"])
def home():
    return app.send_static_file("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON."}), 400

    payload = request.get_json()
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    text = payload.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Please provide a non-empty 'text' field."}), 400

    try:
        prediction = model_pipeline.predict([text])[0]
        probabilities = model_pipeline.predict_proba([text])[0]
        confidence = float(max(probabilities)) * 100
        log_prediction(text, prediction, confidence)
        record_session_prediction(prediction)

        return jsonify(
            {
                "prediction": prediction,
                "confidence": round(confidence, 2),
            }
        )
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


@app.route("/stats", methods=["GET"])
def stats():
    return jsonify(SESSION_STATS)


if __name__ == "__main__":
    app.run(debug=True)
