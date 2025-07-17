import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
from datetime import datetime

class CloudburstPredictor:
    def __init__(self):
        self.model6 = None
        self.model12 = None
        self.load_models()
    
    def load_models(self):
        """Load the pre-trained models"""
        try:
            print("Loading models...")
            self.model6 = load_model('Model6.h5')
            self.model12 = load_model('Model12.h5')
            print("✓ Models loaded successfully!")
        except Exception as e:
            print(f"✗ Error loading models: {e}")
            return False
        return True
    
    def predict_from_test_data(self, sample_index=None):
        """Make predictions using test data"""
        try:
            # Load test data
            test6 = np.load('TEST6.npy')
            test12 = np.load('TEST12.npy')
            
            if sample_index is None:
                # Use all test samples
                samples = min(len(test6), len(test12))
                sample_indices = range(samples)
            else:
                sample_indices = [sample_index]
                
            results = []
            
            for idx in sample_indices:
                if idx >= len(test6) or idx >= len(test12):
                    print(f"Sample {idx} is out of range")
                    continue
                    
                # Get samples
                sample6 = test6[idx:idx+1]
                sample12 = test12[idx:idx+1]
                
                # Make predictions
                pred6 = self.model6.predict(sample6, verbose=0)[0][0]
                pred12 = self.model12.predict(sample12, verbose=0)[0][0]
                
                # Create result
                result = {
                    'sample_index': idx,
                    'timestamp': datetime.now().isoformat(),
                    'predictions': {
                        '6_hour': {
                            'probability': float(pred6),
                            'percentage': f"{pred6*100:.1f}%",
                            'classification': self.classify_risk(pred6)
                        },
                        '12_hour': {
                            'probability': float(pred12),
                            'percentage': f"{pred12*100:.1f}%",
                            'classification': self.classify_risk(pred12)
                        }
                    }
                }
                
                results.append(result)
                
                # Print result
                print(f"\n{'='*50}")
                print(f"SAMPLE {idx + 1} PREDICTION RESULTS")
                print(f"{'='*50}")
                print(f"6-hour prediction:  {pred6*100:.1f}% → {self.classify_risk(pred6)}")
                print(f"12-hour prediction: {pred12*100:.1f}% → {self.classify_risk(pred12)}")
                print(f"Overall Risk: {self.get_overall_risk(pred6, pred12)}")
            
            return results
            
        except Exception as e:
            print(f"Error making predictions: {e}")
            return None
    
    def classify_risk(self, probability):
        """Classify risk level based on probability"""
        if probability >= 0.8:
            return "🔴 HIGH RISK"
        elif probability >= 0.5:
            return "🟡 MODERATE RISK"
        else:
            return "🟢 LOW RISK"
    
    def get_overall_risk(self, pred6, pred12):
        """Get overall risk assessment"""
        max_risk = max(pred6, pred12)
        if max_risk >= 0.8:
            return "🔴 HIGH RISK - Immediate attention required"
        elif max_risk >= 0.5:
            return "🟡 MODERATE RISK - Monitor conditions closely"
        else:
            return "🟢 LOW RISK - Normal conditions"
    
    def batch_predict(self, num_samples=5):
        """Predict multiple samples"""
        print(f"\n{'='*60}")
        print(f"BATCH PREDICTION - {num_samples} SAMPLES")
        print(f"{'='*60}")
        
        results = self.predict_from_test_data()
        if results:
            # Show summary
            high_risk = sum(1 for r in results[:num_samples] 
                          if max(r['predictions']['6_hour']['probability'],
                                 r['predictions']['12_hour']['probability']) >= 0.8)
            moderate_risk = sum(1 for r in results[:num_samples] 
                              if 0.5 <= max(r['predictions']['6_hour']['probability'],
                                           r['predictions']['12_hour']['probability']) < 0.8)
            low_risk = num_samples - high_risk - moderate_risk
            
            print(f"\n{'='*30}")
            print("BATCH SUMMARY")
            print(f"{'='*30}")
            print(f"🔴 High Risk Samples: {high_risk}")
            print(f"🟡 Moderate Risk Samples: {moderate_risk}")
            print(f"🟢 Low Risk Samples: {low_risk}")
            
            return results[:num_samples]
        return None

def main():
    print("🌩️  CLOUDBURST PREDICTION SYSTEM")
    print("="*50)
    
    predictor = CloudburstPredictor()
    
    if not predictor.model6 or not predictor.model12:
        print("Failed to load models. Please check your model files.")
        return
    
    while True:
        print("\n" + "="*50)
        print("PREDICTION OPTIONS:")
        print("="*50)
        print("1. Predict single sample")
        print("2. Predict batch of samples")
        print("3. Predict specific sample by index")
        print("4. Quick demo (first 3 samples)")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            import random
            try:
                test6 = np.load('TEST6.npy')
                sample_idx = random.randint(0, len(test6)-1)
                predictor.predict_from_test_data(sample_idx)
            except Exception as e:
                print(f"Error: {e}")
                
        elif choice == '2':
            try:
                num_samples = int(input("Enter number of samples (1-8): "))
                if 1 <= num_samples <= 8:
                    predictor.batch_predict(num_samples)
                else:
                    print("Please enter a number between 1 and 8")
            except ValueError:
                print("Please enter a valid number")
                
        elif choice == '3':
            try:
                sample_idx = int(input("Enter sample index (0-7): "))
                if 0 <= sample_idx <= 7:
                    predictor.predict_from_test_data(sample_idx)
                else:
                    print("Please enter an index between 0 and 7")
            except ValueError:
                print("Please enter a valid number")
                
        elif choice == '4':
            print("\nRunning quick demo...")
            predictor.batch_predict(3)
            
        elif choice == '5':
            print("Goodbye! 👋")
            break
            
        else:
            print("Invalid choice. Please select 1-5.")

if __name__ == "__main__":
    main()
