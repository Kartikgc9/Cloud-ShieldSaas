import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def test_prediction():
    print("🌩️ Minimal Cloudburst Prediction Test")
    print("="*50)
    
    try:
        # Load test data
        print("Loading test data...")
        test6 = np.load('TEST6.npy')
        test12 = np.load('TEST12.npy')
        print(f"✓ Test data loaded: {test6.shape}, {test12.shape}")
        
        # Load models
        print("Loading models...")
        model6 = load_model('Model6.h5')
        model12 = load_model('Model12.h5')
        print("✓ Models loaded successfully!")
        
        # Make predictions on first sample
        print("\nMaking predictions...")
        sample6 = test6[0:1]  # First sample
        sample12 = test12[0:1]
        
        pred6 = model6.predict(sample6, verbose=0)[0][0]
        pred12 = model12.predict(sample12, verbose=0)[0][0]
        
        print(f"\n🎯 PREDICTION RESULTS:")
        print(f"6-hour prediction:  {pred6*100:.1f}%")
        print(f"12-hour prediction: {pred12*100:.1f}%")
        
        # Risk assessment
        risk6 = "HIGH RISK" if pred6 > 0.8 else "MODERATE RISK" if pred6 > 0.5 else "LOW RISK"
        risk12 = "HIGH RISK" if pred12 > 0.8 else "MODERATE RISK" if pred12 > 0.5 else "LOW RISK"
        
        print(f"6-hour risk:  🔴 {risk6}")
        print(f"12-hour risk: 🔴 {risk12}")
        
        print(f"\n✅ SUCCESS! Your cloudburst prediction system is working!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have the required files:")
        print("- Model6.h5")
        print("- Model12.h5") 
        print("- TEST6.npy")
        print("- TEST12.npy")

if __name__ == "__main__":
    test_prediction()
