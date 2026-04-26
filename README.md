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

## API Endpoints

- `POST /predict`
  - Request JSON: `{ "text": "Your message here" }`
  - Response JSON: `{ "prediction": "bullying", "confidence": 85.12 }`

- `GET /stats`
  - Response JSON: counts of logged predictions

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
