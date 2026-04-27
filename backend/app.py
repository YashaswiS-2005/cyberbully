import csv
import os
import sys
import pickle
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request, send_file

from backend.social_media_collector import create_collector
from backend.batch_processing import BatchProcessor, AsyncDetectionQueue
from backend.train_model_large import LargeDatasetTrainer
from backend.feed_simulation import generate_live_feed

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "pipeline.pkl")
PRIMARY_LOG_PATH = os.path.join(BASE_DIR, "predictions_log.csv")
FALLBACK_LOG_PATH = os.path.join(BASE_DIR, "logs", "predictions_log.csv")
FRONTEND_PATH = os.path.join(BASE_DIR, "..", "frontend")
COLLECTED_DATA_PATH = os.path.join(BASE_DIR, "..", "dataset", "collected_data.csv")
RESULTS_PATH = os.path.join(BASE_DIR, "..", "dataset", "detection_results.csv")

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


# ============================================
# Social Media Collection Endpoints
# ============================================

@app.route("/collect", methods=["POST"])
def collect_social_media():
    """Collect posts/comments from social media platforms."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON."}), 400

    payload = request.get_json()
    platform = payload.get("platform", "twitter")
    query = payload.get("query", "")
    limit = payload.get("limit", 100)
    collect_comments = payload.get("collect_comments", False)
    post_ids = payload.get("post_ids", [])

    # Get API credentials from config or environment
    config = {
        "twitter_bearer_token": os.environ.get("TWITTER_BEARER_TOKEN"),
        "twitter_api_key": os.environ.get("TWITTER_API_KEY"),
        "twitter_api_secret": os.environ.get("TWITTER_API_SECRET"),
        "twitter_access_token": os.environ.get("TWITTER_ACCESS_TOKEN"),
        "twitter_access_secret": os.environ.get("TWITTER_ACCESS_SECRET"),
        "reddit_client_id": os.environ.get("REDDIT_CLIENT_ID"),
        "reddit_client_secret": os.environ.get("REDDIT_CLIENT_SECRET"),
        "reddit_username": os.environ.get("REDDIT_USERNAME"),
        "reddit_password": os.environ.get("REDDIT_PASSWORD"),
    }

    try:
        collector = create_collector(platform, config)
        
        if not collector.authenticate():
            return jsonify({"error": f"Authentication failed for {platform}"}), 401

        collected = []
        
        # Collect posts
        if query:
            posts = collector.fetch_posts(query, limit)
            collected.extend(posts)
        
        # Collect comments for specific posts
        if collect_comments and post_ids:
            for post_id in post_ids:
                comments = collector.fetch_comments(post_id, limit)
                collected.extend(comments)

        # Save to CSV
        if collected:
            collector.save_to_csv(COLLECTED_DATA_PATH, platform)

        return jsonify({
            "status": "success",
            "platform": platform,
            "collected_count": len(collected),
            "saved_to": COLLECTED_DATA_PATH
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/collect/batch", methods=["POST"])
def collect_batch():
    """Batch collect from multiple queries/platforms."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON."}), 400

    payload = request.get_json()
    queries = payload.get("queries", [])
    platform = payload.get("platform", "twitter")
    limit_per_query = payload.get("limit_per_query", 50)

    all_collected = []
    
    config = {
        "twitter_bearer_token": os.environ.get("TWITTER_BEARER_TOKEN"),
        "twitter_api_key": os.environ.get("TWITTER_API_KEY"),
        "twitter_api_secret": os.environ.get("TWITTER_API_SECRET"),
    }

    try:
        collector = create_collector(platform, config)
        
        if not collector.authenticate():
            return jsonify({"error": "Authentication failed"}), 401

        for query in queries:
            posts = collector.fetch_posts(query, limit_per_query)
            all_collected.extend(posts)

        if all_collected:
            collector.save_to_csv(COLLECTED_DATA_PATH, platform)

        return jsonify({
            "status": "success",
            "total_collected": len(all_collected),
            "queries_processed": len(queries)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================
# Large Dataset Processing Endpoints
# ============================================

@app.route("/process/batch", methods=["POST"])
def process_batch():
    """Process a batch of texts for detection."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON."}), 400

    payload = request.get_json()
    texts = payload.get("texts", [])

    if not texts:
        return jsonify({"error": "No texts provided"}), 400

    try:
        processor = BatchProcessor(model_pipeline, batch_size=1000)
        predictions = []
        confidences = []

        for i in range(0, len(texts), 1000):
            batch = texts[i:i+1000]
            preds = model_pipeline.predict(batch)
            probs = model_pipeline.predict_proba(batch)
            
            predictions.extend(preds)
            confidences.extend([float(max(p)) * 100 for p in probs])

        results = [
            {"text": t, "prediction": p, "confidence": round(c, 2)}
            for t, p, c in zip(texts, predictions, confidences)
        ]

        return jsonify({
            "status": "success",
            "processed": len(results),
            "results": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/process/csv", methods=["POST"])
def process_csv_file():
    """Process a CSV file and return detection results."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON."}), 400

    payload = request.get_json()
    input_path = payload.get("input_path", COLLECTED_DATA_PATH)
    text_column = payload.get("text_column", "text")
    output_path = payload.get("output_path", RESULTS_PATH)

    if not os.path.exists(input_path):
        return jsonify({"error": f"Input file not found: {input_path}"}), 404

    try:
        processor = BatchProcessor(model_pipeline, batch_size=1000)
        result_path = processor.process_csv(
            input_path,
            output_path,
            text_column=text_column
        )

        return jsonify({
            "status": "success",
            "input": input_path,
            "output": result_path
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/process/download", methods=["GET"])
def download_results():
    """Download detection results as CSV."""
    if not os.path.exists(RESULTS_PATH):
        return jsonify({"error": "No results available"}), 404

    return send_file(
        RESULTS_PATH,
        mimetype="text/csv",
        as_attachment=True,
        download_name="detection_results.csv"
    )


# ============================================
# Model Training Endpoints
# ============================================

@app.route("/train", methods=["POST"])
def train_model():
    """Train the model on a dataset."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON."}), 400

    payload = request.get_json()
    data_path = payload.get("data_path", COLLECTED_DATA_PATH)
    use_sgd = payload.get("use_sgd", True)
    balance = payload.get("balance", True)

    if not os.path.exists(data_path):
        return jsonify({"error": f"Data file not found: {data_path}"}), 404

    try:
        trainer = LargeDatasetTrainer()
        stats = trainer.train(data_path, use_sgd=use_sgd, balance=balance)
        trainer.save_model(MODEL_PATH)
        
        # Reload the model
        global model_pipeline
        model_pipeline = trainer.load_model()

        return jsonify({
            "status": "success",
            "training_stats": stats
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/train/incremental", methods=["POST"])
def train_incremental():
    """Incrementally train on new data."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON."}), 400

    payload = request.get_json()
    new_data_path = payload.get("new_data_path")
    labels_path = payload.get("labels_path")

    if not new_data_path or not labels_path:
        return jsonify({"error": "new_data_path and labels_path required"}), 400

    try:
        from backend.train_model_large import IncrementalTrainer
        trainer = IncrementalTrainer(MODEL_PATH)
        trainer.update_with_new_data(new_data_path, labels_path)

        return jsonify({
            "status": "success",
            "message": "Model updated incrementally"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================
# Analytics Endpoints
# ============================================

@app.route("/analytics/dataset", methods=["GET"])
def analyze_dataset():
    """Analyze dataset distribution."""
    data_path = request.args.get("data_path", COLLECTED_DATA_PATH)

    if not os.path.exists(data_path):
        return jsonify({"error": f"Data file not found: {data_path}"}), 404

    try:
        from backend.batch_processing import LargeDatasetHandler
        handler = LargeDatasetHandler(model_pipeline)
        stats = handler.analyze_distribution(data_path)

        return jsonify(stats)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analytics/sample", methods=["POST"])
def sample_dataset():
    """Sample a large dataset."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON."}), 400

    payload = request.get_json()
    data_path = payload.get("data_path", COLLECTED_DATA_PATH)
    sample_size = payload.get("sample_size", 1000)
    strategy = payload.get("strategy", "random")

    if not os.path.exists(data_path):
        return jsonify({"error": f"Data file not found: {data_path}"}), 404

    try:
        from backend.batch_processing import LargeDatasetHandler
        handler = LargeDatasetHandler(model_pipeline)
        df = handler.sample_dataset(data_path, sample_size, strategy)

        return jsonify({
            "status": "success",
            "sample_size": len(df),
            "label_distribution": df["label"].value_counts().to_dict() if "label" in df.columns else {}
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================
# Feed Simulation Endpoints
# ============================================

@app.route("/feed", methods=["GET"])
def get_feed():
    """
    Get simulated live feed from social media.
    Returns a list of comments with detection results.
    """
    try:
        # Get optional parameters
        count = request.args.get("count", 10, type=int)
        count = min(max(count, 1), 50)  # Limit between 1-50
        
        # Generate the feed
        result = generate_live_feed(count)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/feed/refresh", methods=["GET"])
def refresh_feed():
    """Refresh the feed with new random comments."""
    return get_feed()


# ============================================
# Dashboard Analytics Endpoint
# ============================================

@app.route("/dashboard", methods=["GET"])
def dashboard():
    """
    Get dashboard statistics including:
    - Total comments analyzed
    - Number of toxic comments
    - Percentage of toxicity
    """
    try:
        # Get feed for analysis
        result = generate_live_feed(20)
        stats = result["stats"]
        
        return jsonify({
            "total_comments": stats["total_comments"],
            "toxic_comments": stats["toxic_count"],
            "toxicity_percentage": stats["toxicity_rate"],
            "bullying_count": stats["bullying_count"],
            "offensive_count": stats["offensive_count"],
            "neutral_count": stats["neutral_count"],
            "alert": result["alert"],
            "alert_message": result["alert_message"]
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
