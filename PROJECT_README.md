# Cloudburst Prediction using GAF and CNN

## Overview

This project is a cloudburst prediction platform focused on Uttarakhand, India. It leverages deep learning (CNN) and Gramian Angular Field (GAF) image encoding to forecast cloudburst events 6 and 12 hours in advance. The system provides a web dashboard, REST API, and automated data pipelines for real-time and batch predictions.

## Technologies Used

- **Python 3.11**
- **Flask** (web framework, REST API)
- **TensorFlow / Keras** (deep learning models)
- **NumPy, Pandas** (data processing)
- **PyTS** (GAF image generation)
- **Matplotlib** (visualizations)
- **SQLite** (default database, can be swapped for PostgreSQL)
- **Docker** (containerization)
- **HTML/CSS/JS** (dashboard UI)
- **Flask-Mail** (email notifications)
- **OpenWeatherMap & IMD APIs** (weather data sources)
- **dotenv** (environment variable management)

## Project Structure

- `app.py` - Main Flask app, API endpoints, dashboard, user management
- `API.py` - Core prediction logic, model loading, GAF generation
- `config.py` - Configuration and environment variables
- `data_updater.py` - Automated weather data collection
- `retrain_model.py` - Model retraining with new data/labels
- `interactive_predict.py`, `quick_prediction_test.py`, `minimal_test.py` - CLI prediction/testing scripts
- `Dockerfile`, `docker-compose.yml` - Containerization and deployment
- `templates/` - HTML templates for dashboard and web pages
- `instance/` - Database files
- `.npy`, `.csv`, `.h5` files - Data, labels, and trained models

## How It Works

1. **Data Collection**: Weather data is fetched from IMD (primary), OpenWeatherMap (secondary), or generated as realistic samples.
2. **Preprocessing**: Data is converted into GAF images, which encode time-series features as images.
3. **Prediction**: Pre-trained CNN models (`Model6.h5`, `Model12.h5`) predict cloudburst probability for 6h and 12h horizons.
4. **API & Dashboard**: Users can access predictions via REST API or a web dashboard. Premium users get real predictions; free users get demo data.
5. **Retraining**: Models can be retrained with new data and updated labels for improved accuracy.

## API Usage

**Endpoint:** `/api/predict`  
**Method:** `POST`  
**Headers:**  
- `X-API-Key: YOUR_API_KEY`  
- `Content-Type: application/json`  

**Request Example:**
```json
{
  "location": "Nainital, Uttarakhand"
}
```

**Response Example:**
```json
{
  "location": "Nainital, Uttarakhand",
  "predictions": {
    "6_hour": { "probability": 0.85, "percentage": "85.0%" },
    "12_hour": { "probability": 0.72, "percentage": "72.0%" }
  },
  "risk_level": "HIGH",
  "timestamp": "2024-07-04T14:30:00Z"
}
```

## Setup & Installation

1. **Clone the repository**
2. **Install dependencies**
   - `pip install -r requirements_saas.txt`
3. **Set up environment variables**
   - Create a `.env` file for API keys and secrets
4. **Prepare models and data**
   - Place `Model6.h5`, `Model12.h5`, and test data in the project directory
5. **Run the app**
   - `python app.py` or use Docker: `docker-compose up`
6. **Access dashboard**
   - Visit `http://localhost:5000`

## File Organization

- `data_updater.py` - Data collection
- `retrain_model.py` - Model retraining
- `uttarakhand_weather_*.csv` - Raw weather data
- `uttarakhand_gaf_data_*.npy` - GAF images
- `uttarakhand_labels_*.npy` - Event labels
- `Model6h_*.h5`, `Model12h_*.h5` - Trained models
- `training_history_*.png` - Training plots

## Troubleshooting

- **Model not loading**: Check model file paths and formats
- **API errors**: Verify API keys and data format
- **Data issues**: Validate input data, check for outliers
- **Performance**: Retrain with more data, update labels

## Next Steps

- Run `python data_updater.py` to collect new data
- Update labels with actual cloudburst events
- Retrain models with `python retrain_model.py`
- Deploy updated models for production
- Monitor and improve continuously

## License

MIT License

---

This README covers the architecture, tech stack, usage, and key files. You can use this to answer interview questions about how the system works, what technologies are used, and how to deploy or extend it.
