import requests
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import time
import os
from pyts.image import GramianAngularField
import warnings
warnings.filterwarnings('ignore')

class UttarakhandDataUpdater:
    def __init__(self):
        self.uttarakhand_stations = {
            # Major stations in Uttarakhand prone to cloudbursts
            'DEHRADUN': {
                'lat': 30.3165, 'lon': 78.0322,
                'imd_code': 'DEHRADUN',
                'aws_id': 'MUSSOORIE(UKG)_UKG'
            },
            'NAINITAL': {
                'lat': 29.3919, 'lon': 79.4542,
                'imd_code': 'NAINITAL',
                'aws_id': 'NAINITAL'
            },
            'HARIDWAR': {
                'lat': 29.9457, 'lon': 78.1642,
                'imd_code': 'HARIDWAR',
                'aws_id': 'HARIDWAR'
            },
            'RISHIKESH': {
                'lat': 30.0869, 'lon': 78.2676,
                'imd_code': 'RISHIKESH',
                'aws_id': 'RISHIKESH'
            },
            'TEHRI': {
                'lat': 30.3889, 'lon': 78.4808,
                'imd_code': 'TEHRI',
                'aws_id': 'TEHRI'
            },
            'CHAMOLI': {
                'lat': 30.4048, 'lon': 79.3311,
                'imd_code': 'CHAMOLI',
                'aws_id': 'CHAMOLI'
            },
            'PITHORAGARH': {
                'lat': 29.5827, 'lon': 80.2186,
                'imd_code': 'PITHORAGARH',
                'aws_id': 'PITHORAGARH'
            }
        }
        
        self.weather_features = [
            'rainfall',      # mm
            'temperature',   # °C
            'humidity',      # %
            'wind_speed',    # m/s
            'pressure',      # hPa
            'msl_pressure'   # hPa
        ]
        
    def fetch_imd_data(self, station, start_date, end_date):
        """Fetch data from IMD AWS"""
        station_info = self.uttarakhand_stations[station]
        
        # IMD AWS API endpoint
        url = f"http://aws.imd.gov.in:8091/AWS/dataview.php"
        params = {
            'a': 'AWS',
            'b': 'UTTARAKHAND',
            'c': station_info['imd_code'],
            'd': station_info['aws_id'],
            'e': start_date.strftime('%Y-%m-%d'),
            'f': end_date.strftime('%Y-%m-%d'),
            'g': 'ALL_HOUR',
            'h': 'ALL_MINUTE'
        }
        
        try:
            print(f"Fetching IMD data for {station}...")
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                # Parse IMD response (usually HTML table)
                data = self.parse_imd_response(response.text, station)
                print(f"✓ Fetched {len(data)} records from IMD for {station}")
                return data
            else:
                print(f"✗ IMD fetch failed for {station}: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"✗ Error fetching IMD data for {station}: {e}")
            return None
    
    def fetch_openweather_data(self, station, api_key, days=7):
        """Fetch historical data from OpenWeatherMap"""
        station_info = self.uttarakhand_stations[station]
        
        # Get historical weather data
        url = "http://api.openweathermap.org/data/2.5/onecall/timemachine"
        
        all_data = []
        end_date = datetime.now()
        
        for i in range(days):
            date = end_date - timedelta(days=i)
            timestamp = int(date.timestamp())
            
            params = {
                'lat': station_info['lat'],
                'lon': station_info['lon'],
                'dt': timestamp,
                'appid': api_key,
                'units': 'metric'
            }
            
            try:
                response = requests.get(url, params=params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    processed = self.process_openweather_data(data, station, date)
                    if processed:
                        all_data.extend(processed)
                    time.sleep(1)  # Rate limiting
                else:
                    print(f"OpenWeather API error for {station}: {response.status_code}")
                    
            except Exception as e:
                print(f"Error fetching OpenWeather data for {station}: {e}")
        
        print(f"✓ Fetched {len(all_data)} records from OpenWeatherMap for {station}")
        return all_data
    
    def process_openweather_data(self, data, station, date):
        """Process OpenWeatherMap API response"""
        if 'hourly' not in data:
            return None
            
        processed_data = []
        
        for hour_data in data['hourly']:
            record = {
                'station': station,
                'datetime': datetime.fromtimestamp(hour_data['dt']),
                'rainfall': hour_data.get('rain', {}).get('1h', 0),
                'temperature': hour_data['temp'],
                'humidity': hour_data['humidity'],
                'wind_speed': hour_data['wind_speed'],
                'pressure': hour_data['pressure'],
                'msl_pressure': hour_data['pressure'],  # Approximation
                'weather_main': hour_data['weather'][0]['main'],
                'weather_description': hour_data['weather'][0]['description']
            }
            processed_data.append(record)
            
        return processed_data
    
    def fetch_current_weather_multiple_sources(self, api_key=None):
        """Fetch current weather from multiple sources"""
        print("🌦️  Fetching Current Uttarakhand Weather Data")
        print("="*60)
        
        all_data = []
        
        for station in self.uttarakhand_stations.keys():
            print(f"\nProcessing {station}...")
            
            # Try IMD first
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            imd_data = self.fetch_imd_data(station, start_date, end_date)
            if imd_data:
                all_data.extend(imd_data)
            
            # Try OpenWeatherMap if API key provided
            if api_key:
                ow_data = self.fetch_openweather_data(station, api_key, days=7)
                if ow_data:
                    all_data.extend(ow_data)
            
            # Add delay between stations
            time.sleep(2)
        
        if all_data:
            df = pd.DataFrame(all_data)
            # Save raw data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"uttarakhand_weather_data_{timestamp}.csv"
            df.to_csv(filename, index=False)
            print(f"\n✓ Saved {len(df)} records to {filename}")
            return df
        else:
            print("\n⚠️  No data fetched. Generating sample data for demonstration...")
            return self.generate_sample_uttarakhand_data()
    
    def generate_sample_uttarakhand_data(self):
        """Generate realistic sample data based on Uttarakhand weather patterns"""
        print("Generating sample Uttarakhand weather data...")
        
        data = []
        end_date = datetime.now()
        
        for station in self.uttarakhand_stations.keys():
            for i in range(168):  # 7 days * 24 hours
                dt = end_date - timedelta(hours=i)
                
                # Monsoon season characteristics (June-September)
                is_monsoon = dt.month in [6, 7, 8, 9]
                
                # Base values for Uttarakhand
                if station in ['DEHRADUN', 'HARIDWAR']:  # Lower altitude
                    temp_base = 28 if is_monsoon else 22
                    rainfall_prob = 0.4 if is_monsoon else 0.1
                else:  # Higher altitude
                    temp_base = 22 if is_monsoon else 15
                    rainfall_prob = 0.6 if is_monsoon else 0.15
                
                # Add realistic variations
                temp_variation = np.random.normal(0, 5)
                
                record = {
                    'station': station,
                    'datetime': dt,
                    'rainfall': max(0, np.random.exponential(2) if np.random.random() < rainfall_prob else 0),
                    'temperature': temp_base + temp_variation,
                    'humidity': np.clip(np.random.normal(75 if is_monsoon else 60, 15), 30, 100),
                    'wind_speed': np.clip(np.random.normal(8 if is_monsoon else 5, 3), 0, 25),
                    'pressure': np.random.normal(1013.25, 5),
                    'msl_pressure': np.random.normal(1013.25, 5),
                    'weather_main': 'Rain' if np.random.random() < rainfall_prob else 'Clear'
                }
                data.append(record)
        
        df = pd.DataFrame(data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sample_uttarakhand_data_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"✓ Generated {len(df)} sample records and saved to {filename}")
        return df
    
    def prepare_training_data(self, weather_df, cloudburst_labels=None):
        """Prepare weather data for model training"""
        print("\n🔄 Preparing training data...")
        
        # Group by station and create sequences
        training_sequences = []
        labels = []
        
        for station in weather_df['station'].unique():
            station_data = weather_df[weather_df['station'] == station].sort_values('datetime')
            
            # Create 6-hour sequences
            for i in range(len(station_data) - 6):
                sequence = station_data.iloc[i:i+6]
                
                # Extract features
                features = []
                for feature in self.weather_features:
                    if feature in sequence.columns:
                        features.append(sequence[feature].values)
                    else:
                        features.append(np.zeros(6))  # Fill missing features
                
                features_array = np.array(features).T  # Shape: (6, 6)
                training_sequences.append(features_array)
                
                # Generate label (you'll need to replace this with actual cloudburst data)
                if cloudburst_labels is not None:
                    # Use provided labels
                    label = cloudburst_labels[len(labels)]
                else:
                    # Heuristic based on weather conditions (for demonstration)
                    rainfall = sequence['rainfall'].sum()
                    max_temp = sequence['temperature'].max()
                    avg_humidity = sequence['humidity'].mean()
                    
                    # Simple heuristic for cloudburst conditions
                    cloudburst_score = 0
                    if rainfall > 20:  # Heavy rainfall
                        cloudburst_score += 0.4
                    if max_temp > 30:  # High temperature
                        cloudburst_score += 0.2
                    if avg_humidity > 80:  # High humidity
                        cloudburst_score += 0.3
                    
                    label = 1 if cloudburst_score > 0.6 else 0
                
                labels.append(label)
        
        # Convert to numpy arrays
        X = np.array(training_sequences)
        y = np.array(labels)
        
        print(f"✓ Prepared {len(X)} training sequences")
        print(f"  - Positive samples (cloudbursts): {np.sum(y)}")
        print(f"  - Negative samples: {len(y) - np.sum(y)}")
        
        return X, y
    
    def generate_gaf_images(self, sequences):
        """Convert weather sequences to GAF images"""
        print("\n🖼️  Generating GAF images...")
        
        gaf = GramianAngularField(method='summation', image_size=min(len(sequences[0]), 256))
        gaf_images = []
        
        for i, sequence in enumerate(sequences):
            if i % 100 == 0:
                print(f"Processing sequence {i+1}/{len(sequences)}")
            
            # Generate GAF for each feature
            sequence_gafs = []
            for feature_idx in range(sequence.shape[1]):
                try:
                    feature_data = sequence[:, feature_idx].reshape(1, -1)
                    gaf_img = gaf.fit_transform(feature_data)[0]
                    
                    # Resize to 256x256 if needed
                    if gaf_img.shape[0] != 256:
                        from scipy.ndimage import zoom
                        zoom_factor = 256 / gaf_img.shape[0]
                        gaf_img = zoom(gaf_img, zoom_factor)
                    
                    sequence_gafs.append(gaf_img)
                except Exception as e:
                    print(f"Warning: GAF generation failed for sequence {i}, feature {feature_idx}: {e}")
                    sequence_gafs.append(np.zeros((256, 256)))
            
            gaf_images.append(np.array(sequence_gafs))
        
        final_images = np.array(gaf_images)
        print(f"✓ Generated GAF images with shape: {final_images.shape}")
        
        return final_images

def main():
    """Main function to update Uttarakhand weather database"""
    print("🌩️ UTTARAKHAND CLOUDBURST DATA UPDATER")
    print("="*60)
    
    updater = UttarakhandDataUpdater()
    
    # Option to use OpenWeatherMap API
    print("\nDo you have an OpenWeatherMap API key? (Get free key from openweathermap.org)")
    use_api = input("Enter API key (or press Enter to skip): ").strip()
    api_key = use_api if use_api else None
    
    # Fetch current weather data
    weather_df = updater.fetch_current_weather_multiple_sources(api_key)
    
    if weather_df is not None and not weather_df.empty:
        # Prepare training data
        X_sequences, y_labels = updater.prepare_training_data(weather_df)
        
        # Generate GAF images
        X_gaf = updater.generate_gaf_images(X_sequences)
        
        # Save processed data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save GAF images
        np.save(f'uttarakhand_gaf_data_{timestamp}.npy', X_gaf)
        np.save(f'uttarakhand_labels_{timestamp}.npy', y_labels)
        
        print(f"\n✅ DATA UPDATE COMPLETE!")
        print(f"📁 Files created:")
        print(f"   - uttarakhand_gaf_data_{timestamp}.npy ({X_gaf.shape})")
        print(f"   - uttarakhand_labels_{timestamp}.npy ({y_labels.shape})")
        print(f"   - Weather CSV file")
        
        print(f"\n🔄 Next steps:")
        print(f"   1. Review the generated data")
        print(f"   2. Add real cloudburst occurrence labels")
        print(f"   3. Run model retraining with: python retrain_model.py")
        
        return X_gaf, y_labels
    else:
        print("❌ Failed to fetch/generate weather data")
        return None, None

if __name__ == "__main__":
    main()
