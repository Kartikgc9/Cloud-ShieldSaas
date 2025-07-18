import os
import sys
# Ensure compatible numpy version
try:
    import numpy as np
    if int(np.__version__.split('.')[0]) >= 2:
        raise ImportError(f"Incompatible numpy version: {np.__version__}. Please install numpy<2.0.0 for compatibility with matplotlib and scientific libraries.")
except ImportError as e:
    print(f"\n[ERROR] {e}\nRun: pip install 'numpy<2.0.0'\n")
    sys.exit(1)
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report
import tensorflow as tf
from tensorflow.keras.models import load_model
 
def load_test_data(data_type='6'):
    """Load test data for evaluation"""
    print(f"Loading test data for {data_type}-hour prediction...")
    
    # Use correct file names (CB6.npy instead of CB18.npy)
    cloudburst_file = f'CB{data_type}.npy'
    non_cloudburst_file = 'NB.npy'
    test_file = f'TEST{data_type}.npy'
    
    if not all(os.path.exists(f) for f in [cloudburst_file, non_cloudburst_file]):
        print(f"Error: Required data files not found!")
        return None, None, None
    
    # Load training data for comparison
    cloudburst_data = np.load(cloudburst_file)
    non_cloudburst_data = np.load(non_cloudburst_file)
    
    print(f"Cloudburst data shape: {cloudburst_data.shape}")
    print(f"Non-cloudburst data shape: {non_cloudburst_data.shape}")
    
    # Combine the data
    X = np.concatenate((cloudburst_data, non_cloudburst_data))
    y = np.concatenate((np.ones(cloudburst_data.shape[0]), np.zeros(non_cloudburst_data.shape[0])))
    
    # Shuffle the data (same as training)
    np.random.seed(42)  # For reproducible results
    shuffle_index = np.random.permutation(X.shape[0])
    X = X[shuffle_index]
    y = y[shuffle_index]
    
    print(f"Combined test data shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Load separate test data if available
    test_data = None
    if os.path.exists(test_file):
        test_data = np.load(test_file)
        print(f"Additional test data shape: {test_data.shape}")
    
    return X, y, test_data

def visualize_gaf_images(X, title_prefix="GAF Images"):
    """Visualize GAF images for understanding the data"""
    
    # Define the labels for each feature
    labels = [
        "RAIN FALL CUM. SINCE 0300 UTC (mm)", 
        "TEMP. (°C)", 
        "RH (%)", 
        "WIND SPEED 10 m (Kt)", 
        "SLP (hPa)", 
        "MSLP (hPa / gpm)"
    ]
    
    # Create visualizations for first few samples
    for sample_idx in range(min(2, X.shape[0])):
        plt.figure(figsize=(15, 10))
        plt.suptitle(f'{title_prefix} - Sample {sample_idx + 1}', fontsize=16)
        
        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.imshow(X[sample_idx][i], cmap='viridis')
            plt.title(labels[i], fontsize=10)
            plt.colorbar()
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'gaf_visualization_sample_{sample_idx + 1}.png', dpi=150, bbox_inches='tight')
        plt.show()

def evaluate_model(model_path, X, y, test_data=None, data_type='6'):
    """Evaluate a trained model"""
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        return None
    
    print(f"\nEvaluating {data_type}-hour model: {model_path}")
    
    try:
        # Load the model
        model = load_model(model_path)
        print("Model loaded successfully!")
        print("\nModel Summary:")
        model.summary()
        
        # Make predictions on training/validation data
        print("\nEvaluating on main dataset...")
        y_pred_prob = model.predict(X)
        y_pred = np.round(y_pred_prob).flatten()
        y_true = y.flatten()
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Calculate metrics
        accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        
        print(f"\nModel Performance Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1_score:.4f}")
        
        # Detailed classification report
        print("\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred, target_names=['Non-Cloudburst', 'Cloudburst']))
        
        # Test on additional test data if available
        if test_data is not None:
            print(f"\nTesting on additional test dataset...")
            test_predictions = model.predict(test_data)
            test_pred_percentages = (test_predictions * 100).astype(int)
            
            print("Test Predictions (confidence %):")
            for i, pred in enumerate(test_pred_percentages):
                confidence = pred[0]
                prediction = "Cloudburst" if confidence > 50 else "Non-Cloudburst"
                print(f"Sample {i+1}: {confidence}% ({prediction})")
        
        # Visualize results
        plot_evaluation_results(cm, y_pred_prob, y_true, data_type)
        
        return model
        
    except Exception as e:
        print(f"Error loading or evaluating model: {e}")
        return None

def plot_evaluation_results(cm, y_pred_prob, y_true, data_type):
    """Plot evaluation results"""
    
    plt.figure(figsize=(15, 5))
    
    # Plot confusion matrix
    plt.subplot(1, 3, 1)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {data_type}H Model')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Non-Cloudburst', 'Cloudburst'])
    plt.yticks(tick_marks, ['Non-Cloudburst', 'Cloudburst'])
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Plot prediction distribution
    plt.subplot(1, 3, 2)
    plt.hist(y_pred_prob[y_true == 0], bins=30, alpha=0.7, label='Non-Cloudburst')
    plt.hist(y_pred_prob[y_true == 1], bins=30, alpha=0.7, label='Cloudburst')
    plt.xlabel('Prediction Probability')
    plt.ylabel('Frequency')
    plt.title(f'Prediction Distribution - {data_type}H Model')
    plt.legend()
    plt.axvline(x=0.5, linestyle='--', label='Decision Threshold')
    
    # Plot accuracy vs threshold
    plt.subplot(1, 3, 3)
    thresholds = np.linspace(0, 1, 100)
    accuracies = []
    for thresh in thresholds:
        y_pred_thresh = (y_pred_prob > thresh).astype(int).flatten()
        accuracies.append(np.mean(y_pred_thresh == y_true))
    plt.plot(thresholds, accuracies)
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs Threshold - {data_type}H Model')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'evaluation_results_{data_type}h.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main testing function"""
    print("Cloudburst Prediction Model Testing")
    print("=" * 40)
    
    # Test both models
    for data_type in ['6', '12']:
        print(f"\n{'='*20} {data_type}-Hour Model Testing {'='*20}")
        
        # Load test data
        X, y, test_data = load_test_data(data_type)
        if X is None:
            print(f"Failed to load test data for {data_type}-hour model")
            continue
        
        # Visualize some GAF images
        if data_type == '6':
            print("Visualizing GAF images...")
            visualize_gaf_images(X, f"GAF Images ({data_type}H)")
        
        # Try to load and evaluate both original and fixed models
        model_files = [f'Model{data_type}.h5', f'Model{data_type}_fixed.h5']
        for model_file in model_files:
            if os.path.exists(model_file):
                print(f"\nTesting model: {model_file}")
                evaluate_model(model_file, X, y, test_data, data_type)
            else:
                print(f"Model file {model_file} not found, skipping...")
        
        print("\n" + "="*60)
    
    print("Testing complete!")

if __name__ == "__main__":
    main()
