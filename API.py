import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
from sklearn.preprocessing import StandardScaler
import datetime
import requests
import warnings
from typing import Optional, Dict, List, Tuple
import tensorflow as tf
from tensorflow.keras.models import load_model

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class CloudburstPredictor:
    """Cloudburst prediction system using CNN and GAF"""
    
    def __init__(self):
        self.model_6h = None
        self.model_12h = None
        self.feature_names = [
            "RAIN FALL CUM. SINCE 0300 UTC (mm)",
            "TEMP. (°C)",
            "RH (%)",
            "WIND SPEED 10 m (Kt)",
            "SLP (hPa)",
            "MSLP (hPa / gpm)"
        ]
        
    def load_models(self, model_6h_path='Model6.h5', model_12h_path='Model12.h5'):
        """Load pre-trained models"""
        print("Loading pre-trained models...")
        
        # Try to load models (both original and fixed versions)
        model_files = {
            '6h': [model_6h_path, 'Model6_fixed.h5'],
            '12h': [model_12h_path, 'Model12_fixed.h5']
        }
        
        for model_type, paths in model_files.items():
            for path in paths:
                if os.path.exists(path):
                    try:
                        if model_type == '6h':
                            self.model_6h = load_model(path)
                            print(f"✓ 6-hour model loaded from {path}")
                        else:
                            self.model_12h = load_model(path)
                            print(f"✓ 12-hour model loaded from {path}")
                        break
                    except Exception as e:
                        print(f"Error loading {path}: {e}")
                        continue
            else:
                print(f"✗ No valid {model_type} model found")
    
    def generate_gaf_from_data(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Generate GAF images from meteorological data"""
        try:
            print("Generating GAF images from data...")
            
            # Check if data has required columns
            required_features = len(self.feature_names)
            if len(data.columns) < required_features:
                print(f"Warning: Data has only {len(data.columns)} columns, expected {required_features}")
                return None
            
            gaf_images = []
            
            for i, feature_name in enumerate(self.feature_names):
                if i < len(data.columns):
                    # Use the i-th column if exact feature name not found
                    if feature_name in data.columns:
                        data_feature = data[feature_name].copy()
                    else:
                        data_feature = data.iloc[:, i].copy()
                    
                    # Fill NaN values with the mean of the column
                    data_feature.fillna(data_feature.mean(), inplace=True)
                    
                    # Convert to numpy array
                    data_values = data_feature.values
                    
                    # Ensure we have enough data points
                    if len(data_values) < 10:
                        print(f"Warning: Only {len(data_values)} data points for feature {i}")
                        # Pad with mean if necessary
                        mean_val = np.mean(data_values)
                        data_values = np.pad(data_values, (0, max(0, 10 - len(data_values))), 
                                           constant_values=mean_val)
                    
                    # Create GAF
                    gaf = GramianAngularField(image_size=256, method='summation')
                    
                    try:
                        image = gaf.transform([data_values])
                        gaf_images.append(image[0])
                    except Exception as e:
                        print(f"Error generating GAF for feature {i}: {e}")
                        # Create dummy image if GAF fails
                        gaf_images.append(np.zeros((256, 256)))
                else:
                    # Create dummy image for missing features
                    print(f"Warning: Missing feature {i}, using dummy data")
                    gaf_images.append(np.zeros((256, 256)))
            
            return np.array([gaf_images])
            
        except Exception as e:
            print(f"Error generating GAF images: {e}")
            return None
    
    def generate_sample_data(self, num_points=100) -> pd.DataFrame:
        """Generate sample meteorological data for testing"""
        print("Generating sample meteorological data...")
        
        np.random.seed(42)  # For reproducible results
        
        # Generate realistic-looking weather data
        data = {
            "RAIN FALL CUM. SINCE 0300 UTC (mm)": np.random.exponential(2, num_points),
            "TEMP. (°C)": 20 + 10 * np.sin(np.linspace(0, 4*np.pi, num_points)) + np.random.normal(0, 2, num_points),
            "RH (%)": 50 + 30 * np.sin(np.linspace(0, 2*np.pi, num_points)) + np.random.normal(0, 5, num_points),
            "WIND SPEED 10 m (Kt)": 5 + np.random.exponential(3, num_points),
            "SLP (hPa)": 1013 + np.random.normal(0, 10, num_points),
            "MSLP (hPa / gpm)": 1013 + np.random.normal(0, 8, num_points)
        }
        
        # Ensure realistic ranges
        data["RH (%)"] = np.clip(data["RH (%)"], 0, 100)
        data["WIND SPEED 10 m (Kt)"] = np.clip(data["WIND SPEED 10 m (Kt)"], 0, 50)
        
        return pd.DataFrame(data)
    
    def fetch_imd_data(self, state="UTTARAKHAND", district="DEHRADUN", 
                      station="MUSSOORIE(UKG)_UKG", days_back=6) -> Optional[pd.DataFrame]:
        """Fetch data from IMD (India Meteorological Department)"""
        try:
            current_date = datetime.date.today()
            prev_date = current_date - datetime.timedelta(days=days_back)
            
            url = (
                f"http://aws.imd.gov.in:8091/AWS/dataview.php"
                f"?a=AWS&b={state}&c={district}&d={station}"
                f"&e={prev_date}&f={current_date}&g=ALL_HOUR&h=ALL_MINUTE"
            )
            
            print(f"Attempting to fetch data from IMD...")
            print(f"URL: {url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
            }
            
            response = requests.get(url, timeout=30, headers=headers)
            
            if response.status_code == 200:
                df_list = pd.read_html(response.content)
                if df_list and len(df_list) > 0:
                    print(f"✓ Successfully fetched data from IMD")
                    return df_list[0]
                else:
                    print("✗ No tables found in IMD response")
                    return None
            else:
                print(f"✗ Failed to fetch data from IMD (Status: {response.status_code})")
                return None
                
        except Exception as e:
            print(f"✗ Error fetching IMD data: {e}")
            return None
    
    def predict_cloudburst(self, data: pd.DataFrame, use_sample_data=False) -> Dict:
        """Predict cloudburst probability"""
        
        if data is None or len(data) == 0:
            if use_sample_data:
                print("Using sample data for prediction...")
                data = self.generate_sample_data()
            else:
                return {"error": "No data available for prediction"}
        
        gaf_data = self.generate_gaf_from_data(data)
        if gaf_data is None:
            return {"error": "Failed to generate GAF images"}
        
        results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "data_points": len(data),
            "predictions": {}
        }
        
        if self.model_6h is not None:
            try:
                prediction_6h = self.model_6h.predict(gaf_data)
                probability_6h = float(prediction_6h[0][0] * 100)
                results["predictions"]["6_hour"] = {
                    "probability": probability_6h,
                    "classification": "High Risk" if probability_6h > 70 else "Medium Risk" if probability_6h > 40 else "Low Risk",
                    "confidence": "High" if probability_6h > 80 or probability_6h < 20 else "Medium"
                }
                print(f"✓ 6-hour prediction: {probability_6h:.1f}%")
            except Exception as e:
                print(f"✗ Error in 6-hour prediction: {e}")
                results["predictions"]["6_hour"] = {"error": str(e)}
        
        if self.model_12h is not None:
            try:
                prediction_12h = self.model_12h.predict(gaf_data)
                probability_12h = float(prediction_12h[0][0] * 100)
                results["predictions"]["12_hour"] = {
                    "probability": probability_12h,
                    "classification": "High Risk" if probability_12h > 70 else "Medium Risk" if probability_12h > 40 else "Low Risk",
                    "confidence": "High" if probability_12h > 80 or probability_12h < 20 else "Medium"
                }
                print(f"✓ 12-hour prediction: {probability_12h:.1f}%")
            except Exception as e:
                print(f"✗ Error in 12-hour prediction: {e}")
                results["predictions"]["12_hour"] = {"error": str(e)}
        
        return results
    
    def visualize_gaf_images(self, gaf_data: np.ndarray):
        """Visualize the generated GAF images"""
        if gaf_data is None:
            return
            
        plt.figure(figsize=(15, 10))
        plt.suptitle('Generated GAF Images for Prediction', fontsize=16)
        
        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.imshow(gaf_data[0][i], cmap='viridis')
            plt.title(self.feature_names[i], fontsize=10)
            plt.colorbar()
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('current_gaf_images.png', dpi=150, bbox_inches='tight')
        plt.show()

def main():
    """Main API function"""
    print("Cloudburst Prediction API")
    print("=" * 50)
    
    predictor = CloudburstPredictor()
    predictor.load_models()
    
    if predictor.model_6h is None and predictor.model_12h is None:
        print("✗ No models loaded. Cannot make predictions.")
        return
    
    print("\n" + "="*30 + " DATA FETCHING " + "="*30)
    
    locations = [
        {"state": "UTTARAKHAND", "district": "DEHRADUN", "station": "MUSSOORIE(UKG)_UKG"},
        {"state": "UTTARAKHAND", "district": "NAINITAL", "station": "NAINITAL"},
    ]
    
    data = None
    for location in locations:
        print(f"\nTrying location: {location['district']}, {location['station']}")
        data = predictor.fetch_imd_data(**location)
        if data is not None:
            break
    
    print("\n" + "="*30 + " PREDICTIONS " + "="*30)
    
    if data is None:
        print("Unable to fetch real data. Using sample data for demonstration...")
        results = predictor.predict_cloudburst(None, use_sample_data=True)
        
        sample_data = predictor.generate_sample_data(50)
        gaf_data = predictor.generate_gaf_from_data(sample_data)
        if gaf_data is not None:
            predictor.visualize_gaf_images(gaf_data)
    else:
        print(f"Using real data with {len(data)} records")
        results = predictor.predict_cloudburst(data)
        
        gaf_data = predictor.generate_gaf_from_data(data)
        if gaf_data is not None:
            predictor.visualize_gaf_images(gaf_data)
    
    print("\n" + "="*30 + " RESULTS " + "="*30)
    print(f"Prediction Timestamp: {results.get('timestamp', 'N/A')}")
    print(f"Data Points Used: {results.get('data_points', 'N/A')}")
    
    for time_horizon, prediction in results.get('predictions', {}).items():
        print(f"\n{time_horizon.upper()} PREDICTION:")
        if 'error' in prediction:
            print(f"  ✗ Error: {prediction['error']}")
        else:
            print(f"  Probability: {prediction['probability']:.1f}%")
            print(f"  Classification: {prediction['classification']}")
            print(f"  Confidence: {prediction['confidence']}")
    
    import json
    with open('cloudburst_prediction_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to cloudburst_prediction_results.json")
    
    print("\n" + "="*50)
    print("API execution completed!")

if __name__ == "__main__":
    main()
