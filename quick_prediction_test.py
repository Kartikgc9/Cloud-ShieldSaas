#!/usr/bin/env python3
"""
Quick Cloudburst Prediction Test
This script demonstrates how to make predictions using the trained models.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def load_test_data():
    """Load test data for prediction"""
    print("Loading test data...")
    test6 = np.load('TEST6.npy')
    test12 = np.load('TEST12.npy')
    print(f"6-hour test data shape: {test6.shape}")
    print(f"12-hour test data shape: {test12.shape}")
    return test6, test12

def make_predictions():
    """Make predictions using both models"""
    print("\n" + "="*50)
    print("CLOUDBURST PREDICTION TEST")
    print("="*50)
    
    # Load test data
    test6, test12 = load_test_data()
    
    try:
        # Load models
        print("\nLoading models...")
        model6 = load_model('Model6.h5')
        model12 = load_model('Model12.h5')
        print("✓ Models loaded successfully!")
        
        # Make predictions on first few samples
        print("\n" + "-"*30)
        print("6-HOUR PREDICTIONS")
        print("-"*30)
        
        predictions6 = model6.predict(test6[:5])  # Predict first 5 samples
        for i, pred in enumerate(predictions6):
            confidence = pred[0] * 100
            result = "CLOUDBURST" if pred[0] > 0.5 else "NO CLOUDBURST"
            print(f"Sample {i+1}: {confidence:.1f}% confidence → {result}")
        
        print("\n" + "-"*30)
        print("12-HOUR PREDICTIONS")
        print("-"*30)
        
        predictions12 = model12.predict(test12[:5])  # Predict first 5 samples
        for i, pred in enumerate(predictions12):
            confidence = pred[0] * 100
            result = "CLOUDBURST" if pred[0] > 0.5 else "NO CLOUDBURST"
            print(f"Sample {i+1}: {confidence:.1f}% confidence → {result}")
            
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Make sure Model6.h5 and Model12.h5 are in the current directory")

def predict_custom_sample(sample_index=0):
    """Predict a specific sample and show details"""
    print(f"\n" + "="*50)
    print(f"DETAILED PREDICTION FOR SAMPLE {sample_index + 1}")
    print("="*50)
    
    test6, test12 = load_test_data()
    
    try:
        model6 = load_model('Model6.h5')
        model12 = load_model('Model12.h5')
        
        # Get single sample
        sample6 = test6[sample_index:sample_index+1]
        sample12 = test12[sample_index:sample_index+1]
        
        # Make predictions
        pred6 = model6.predict(sample6)[0][0]
        pred12 = model12.predict(sample12)[0][0]
        
        print(f"Sample shape: {sample6.shape}")
        print(f"6-hour prediction: {pred6:.4f} ({pred6*100:.1f}%)")
        print(f"12-hour prediction: {pred12:.4f} ({pred12*100:.1f}%)")
        
        # Risk assessment
        print("\nRisk Assessment:")
        if pred6 > 0.8 or pred12 > 0.8:
            print("🔴 HIGH RISK - Cloudburst very likely")
        elif pred6 > 0.5 or pred12 > 0.5:
            print("🟡 MODERATE RISK - Cloudburst possible")
        else:
            print("🟢 LOW RISK - Cloudburst unlikely")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Test basic predictions
    make_predictions()
    
    # Test detailed prediction for first sample
    predict_custom_sample(0)
    
    print("\n" + "="*50)
    print("TEST COMPLETE!")
    print("="*50)
    print("\nTo make predictions with your own data:")
    print("1. Use API_Fixed.py for real-time weather data")
    print("2. Or modify this script to load your custom GAF images")
