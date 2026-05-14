# AI-Based Cyberbullying Detection System

This project detects whether a given text message is `bullying`, `offensive`, or `neutral` using a machine learning pipeline with NLP preprocessing.

## Project Structure

- `backend/` - Flask backend, model training, prediction API
- `frontend/` - HTML/CSS/JavaScript web UI
- `dataset/` - sample CSV dataset
- `models/` - trained pipeline saved as `pipeline.pkl`
- `requirements.txt` - Python dependencies

## Setup Instructions

1. Create and activate a Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Train the model and create `models/pipeline.pkl`:

```powershell
python -m backend.train_model
```

4. Run the Flask server:

```powershell
python -m backend.app
```

5. Open the frontend in your browser:

- Visit: `http://127.0.0.1:5000/`

## Email Alerts

The system can send email alerts when bullying content is detected:

1. **Configure Email Settings**: Copy `.env.example` to `.env` and fill in your email credentials
2. **Set Email Address**: Go to Settings page and enter your email address
3. **Receive Alerts**: When bullying content is detected with high confidence, you'll receive an email notification

### Email Configuration

For Gmail users:
- Enable 2-factor authentication
- Generate an App Password: https://support.google.com/accounts/answer/185833
- Use your Gmail address as `MAIL_USERNAME`
- Use the App Password as `MAIL_PASSWORD`

## API Endpoints

- `POST /predict`
  - Request JSON: `{ "text": "Your message here" }`
  - Response JSON: `{ "prediction": "bullying", "confidence": 85.12 }`
  - **New**: Sends email alert if bullying detected and user has email configured

- `GET /stats`
  - Response JSON: counts of logged predictions

- `GET /settings`
  - User settings page for email configuration

## Example Usage

### Predict using curl

```powershell
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"text":"You are stupid"}'
```

### Example API output

```json
{
  "prediction": "bullying",
  "confidence": 91.33
}
```

## Notes

- The dataset contains 50 sample rows with labels `bullying`, `offensive`, and `neutral`.
- Predictions are logged to `backend/predictions_log.csv`.
- The frontend displays the predicted label, confidence, and current counts.
