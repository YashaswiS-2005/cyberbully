import csv
import os
import sys
import pickle
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env when available, otherwise fall back to .env.example
project_root = Path(__file__).resolve().parents[1]
dotenv_file = project_root / ".env"
example_env_file = project_root / ".env.example"
if dotenv_file.exists():
    load_dotenv(dotenv_file)
elif example_env_file.exists():
    load_dotenv(example_env_file)

from flask import Flask, jsonify, request, send_file, redirect, url_for, render_template_string, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from flask_mail import Mail, Message

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

# Database setup
DEFAULT_DB_PATH = Path(BASE_DIR) / "app.db"
raw_database_url = os.environ.get("DATABASE_URL", "").strip()
if (
    not raw_database_url
    or raw_database_url.startswith("postgresql://user:password")
    or raw_database_url == "postgresql://user:password@localhost:5432/cyberbully"
):
    DATABASE_URL = f"sqlite:///{DEFAULT_DB_PATH.as_posix()}"
else:
    DATABASE_URL = raw_database_url

engine_kwargs = {}
if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}
engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# User model
class User(Base, UserMixin):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(150), unique=True, nullable=False)
    email = Column(String(150), unique=True, nullable=False)
    password_hash = Column(String(128), nullable=False)

Base.metadata.create_all(bind=engine)

app = Flask(__name__, static_folder=FRONTEND_PATH, static_url_path="/")
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')

# Flask-Mail configuration
app.config['MAIL_SERVER'] = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.environ.get('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.environ.get('MAIL_USE_TLS', 'True').lower() == 'true'
app.config['MAIL_USE_SSL'] = os.environ.get('MAIL_USE_SSL', 'False').lower() == 'true'
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_DEFAULT_SENDER', app.config['MAIL_USERNAME'])

mail = Mail(app)

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    db = SessionLocal()
    try:
        return db.query(User).filter(User.id == int(user_id)).first()
    finally:
        db.close()

def send_alert_email(user_email, text, prediction, confidence):
    """Send an email alert when bullying content is detected."""
    if not app.config['MAIL_USERNAME']:
        return  # Skip if email not configured

    try:
        msg = Message(
            subject="Cyberbullying Alert Detected",
            recipients=[user_email],
            body=f"""
Cyberbullying Detection Alert

A potentially harmful message has been detected:

Text: {text}
Prediction: {prediction}
Confidence: {confidence:.2f}%

Please review this content and take appropriate action.

This is an automated alert from your Cyberbullying Detection System.
            """
        )
        mail.send(msg)
        print(f"Alert email sent to {user_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")

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


print('APP DEBUG: About to load model pipeline', flush=True)
model_pipeline = load_model()
print('APP DEBUG: Loaded model pipeline', flush=True)


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
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    return app.send_static_file("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        email = request.form.get("email")  # Optional for login, but can update

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.username == username).first()
            if user and check_password_hash(user.password_hash, password):
                # Update email if provided and different
                if email and email != user.email:
                    user.email = email
                    db.commit()
                login_user(user)
                return redirect(url_for('home'))
            else:
                flash("Invalid username or password")
        finally:
            db.close()
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Login - Cyberbullying Detection</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                color: #e8e8e8;
                margin: 0;
                padding: 0;
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
            }
            .login-container {
                background: rgba(15, 23, 42, 0.9);
                padding: 2.5rem;
                border-radius: 16px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                width: 100%;
                max-width: 400px;
                text-align: center;
            }
            h2 {
                margin: 0 0 1.5rem 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                font-size: 2rem;
                font-weight: 600;
            }
            .subtitle {
                color: #a0a0a0;
                margin-bottom: 2rem;
                font-size: 0.95rem;
            }
            input {
                width: 100%;
                padding: 0.75rem;
                margin: 0.5rem 0;
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.05);
                color: #e8e8e8;
                font-size: 1rem;
                box-sizing: border-box;
            }
            input::placeholder {
                color: #a0a0a0;
            }
            input:focus {
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
            }
            button {
                width: 100%;
                padding: 0.75rem;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                margin-top: 1rem;
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            }
            .error {
                color: #ff6b6b;
                background: rgba(255, 107, 107, 0.1);
                padding: 0.5rem;
                border-radius: 4px;
                margin-bottom: 1rem;
                border: 1px solid rgba(255, 107, 107, 0.2);
            }
            .credentials {
                margin-top: 1.5rem;
                padding: 0.75rem;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 8px;
                font-size: 0.9rem;
                color: #a0a0a0;
            }
            .credentials strong {
                color: #667eea;
            }
        </style>
    </head>
    <body>
        <div class="login-container">
            <h2>Cyberbullying Detection</h2>
            <p class="subtitle">Please log in to access the system</p>
            {% with messages = get_flashed_messages() %}
                {% if messages %}
                    <div class="error">{{ messages[0] }}</div>
                {% endif %}
            {% endwith %}
            <form method="post">
                <input type="text" name="username" placeholder="Username" required>
                <input type="email" name="email" placeholder="Email (for alerts)" value="{{ current_user.email if current_user.is_authenticated else '' }}">
                <input type="password" name="password" placeholder="Password" required>
                <button type="submit">Login</button>
            </form>
            <div class="credentials">
                <strong>Default credentials:</strong><br>
                Username: admin<br>
                Password: admin
            </div>
        </div>
    </body>
    </html>
    """)


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    if request.method == "POST":
        email = request.form.get("email")
        if email:
            db = SessionLocal()
            try:
                current_user.email = email
                db.commit()
                flash("Email updated successfully!")
            finally:
                db.close()
        return redirect(url_for('settings'))
    
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Settings - Cyberbullying Detection</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                color: #e8e8e8;
                margin: 0;
                padding: 20px;
                min-height: 100vh;
            }
            .settings-container {
                max-width: 600px;
                margin: 0 auto;
                background: rgba(15, 23, 42, 0.9);
                padding: 2rem;
                border-radius: 16px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            h2 {
                margin: 0 0 1.5rem 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                text-align: center;
            }
            .form-group {
                margin-bottom: 1.5rem;
            }
            label {
                display: block;
                margin-bottom: 0.5rem;
                color: #a0a0a0;
                font-weight: 500;
            }
            input {
                width: 100%;
                padding: 0.75rem;
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.05);
                color: #e8e8e8;
                font-size: 1rem;
                box-sizing: border-box;
            }
            input:focus {
                outline: none;
                border-color: #667eea;
                box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
            }
            .btn {
                padding: 0.75rem 1.5rem;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                margin-right: 1rem;
            }
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            }
            .btn-secondary {
                background: rgba(255, 255, 255, 0.1);
                color: #a0a0a0;
            }
            .btn-secondary:hover {
                background: rgba(255, 255, 255, 0.2);
                color: #e8e8e8;
            }
            .success {
                color: #4caf50;
                background: rgba(76, 175, 80, 0.1);
                padding: 0.5rem;
                border-radius: 4px;
                margin-bottom: 1rem;
                border: 1px solid rgba(76, 175, 80, 0.2);
            }
            .info {
                background: rgba(255, 255, 255, 0.05);
                padding: 1rem;
                border-radius: 8px;
                margin-top: 2rem;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            .info h3 {
                margin: 0 0 0.5rem 0;
                color: #667eea;
            }
            .info p {
                margin: 0.5rem 0;
                color: #a0a0a0;
                line-height: 1.5;
            }
        </style>
    </head>
    <body>
        <div class="settings-container">
            <h2>Account Settings</h2>
            {% with messages = get_flashed_messages() %}
                {% if messages %}
                    <div class="success">{{ messages[0] }}</div>
                {% endif %}
            {% endwith %}
            <form method="post">
                <div class="form-group">
                    <label for="email">Email Address (for alerts)</label>
                    <input type="email" id="email" name="email" value="{{ current_user.email }}" placeholder="Enter your email for bullying alerts" required>
                </div>
                <button type="submit" class="btn">Update Email</button>
                <a href="/" class="btn btn-secondary">Back to Dashboard</a>
            </form>
            
            <div class="info">
                <h3>Email Alert Settings</h3>
                <p>When bullying content is detected, you'll receive email notifications at this address.</p>
                <p><strong>Note:</strong> Make sure to configure email settings in your environment variables for alerts to work.</p>
            </div>
        </div>
    </body>
    </html>
    """)


@app.route("/predict", methods=["POST"])
@login_required
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

        # Send email alert if bullying detected and user has email
        if prediction.lower() == "bullying" and current_user.is_authenticated and current_user.email:
            send_alert_email(current_user.email, text, prediction, confidence)

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


@app.route("/collect/reposts", methods=["POST"])
@login_required
def collect_reposts():
    """Collect reposts of specific posts."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON."}), 400

    payload = request.get_json()
    platform = payload.get("platform", "twitter")
    post_ids = payload.get("post_ids", [])
    limit_per_post = payload.get("limit_per_post", 50)

    if not post_ids:
        return jsonify({"error": "post_ids required"}), 400

    # Get API credentials
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

        all_reposts = []
        
        for post_id in post_ids:
            reposts = collector.fetch_reposts(post_id, limit_per_post)
            all_reposts.extend(reposts)

        # Save to CSV
        if all_reposts:
            collector.save_to_csv(COLLECTED_DATA_PATH, platform)

        return jsonify({
            "status": "success",
            "platform": platform,
            "collected_reposts": len(all_reposts),
            "saved_to": COLLECTED_DATA_PATH
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
    # Create default user if not exists
    db = SessionLocal()
    try:
        if not db.query(User).filter(User.username == "admin").first():
            admin_user = User(
                username="admin", 
                email="admin@example.com",  # Default email
                password_hash=generate_password_hash("admin")
            )
            db.add(admin_user)
            db.commit()
    finally:
        db.close()
    
    app.run(debug=True)
