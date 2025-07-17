import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.regularizers import l1_l2
import h5py

def train_model(data_type='6'):
    """Train model for either 6-hour or 12-hour prediction"""
    print(f"Training {data_type}-hour prediction model...")
    
    # Load the data (fixed paths)
    cloudburst_file = f'CB{data_type}.npy'
    non_cloudburst_file = 'NB.npy'
    test_file = f'TEST{data_type}.npy'
    
    if not os.path.exists(cloudburst_file):
        print(f"Error: {cloudburst_file} not found!")
        return None, None
    if not os.path.exists(non_cloudburst_file):
        print(f"Error: {non_cloudburst_file} not found!")
        return None, None
        
    cloudburst_data = np.load(cloudburst_file)
    non_cloudburst_data = np.load(non_cloudburst_file)
    
    print(f"Cloudburst data shape: {cloudburst_data.shape}")
    print(f"Non-cloudburst data shape: {non_cloudburst_data.shape}")
    
    # Combine the cloudburst and non-cloudburst data into one array
    X = np.concatenate((cloudburst_data, non_cloudburst_data))
    # Create a target vector (1 for cloudburst, 0 for non-cloudburst)
    y = np.concatenate((np.ones(cloudburst_data.shape[0]), np.zeros(non_cloudburst_data.shape[0])))
    
    # Shuffle the data
    shuffle_index = np.random.permutation(X.shape[0])
    X = X[shuffle_index]
    y = y[shuffle_index]
    
    print(f"Combined data shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Model hyperparameters based on README
    if data_type == '6':
        optimizer_name = 'adam'
        kernel_size = (2, 2)
        pooling_size = (3, 3)
        learning_rate = 0.0005
    else:  # 12-hour
        optimizer_name = 'rmsprop'
        kernel_size = (1, 1)
        pooling_size = (3, 3)
        learning_rate = 0.0005
    
    # Clear any existing session
    tf.keras.backend.clear_session()
    
    try:
        # Create the model
        model = Sequential(name=f'cloudburst_model_{data_type}h')
        model.add(InputLayer(input_shape=(6, 256, 256)))
        model.add(Conv2D(filters=128, kernel_size=kernel_size, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=pooling_size, padding='same'))
        model.add(Conv2D(filters=256, kernel_size=kernel_size, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(1, 1), padding='same'))
        model.add(Conv2D(filters=512, kernel_size=kernel_size, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=pooling_size, padding='same'))
        model.add(Flatten())
        model.add(Dense(units=256, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
        model.add(Dropout(rate=0.5))
        model.add(Dense(units=1, activation='sigmoid'))
        
        # Set up optimizer
        if optimizer_name == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = SGD(learning_rate=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = RMSprop(learning_rate=learning_rate)
        
        # Compile the model
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
        print("Model architecture:")
        model.summary()
        
        # Train the model
        print("Starting training...")
        history = model.fit(X, y, epochs=50, batch_size=1, validation_split=0.3, verbose=1)
        
        # Print training results
        print("\nTraining Results:")
        print(f"Optimizer: {optimizer_name}, Kernel Size: {kernel_size}, Pooling Size: {pooling_size}, Learning Rate: {learning_rate}")
        print(f"Final Training accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"Final Validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        
        # Calculate and print the confusion matrix
        y_pred = np.round(model.predict(X)).flatten()
        y_true = y.flatten()
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
        print(f"Overall Accuracy: {accuracy:.4f}")
        
        # Calculate precision, recall, F1-score
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1_score:.4f}")
        
        # Test on validation set if available
        if os.path.exists(test_file):
            print(f"\nTesting on {test_file}...")
            test_data = np.load(test_file)
            print(f"Test data shape: {test_data.shape}")
            predictions = model.predict(test_data)
            prediction_percentages = (predictions * 100).astype(int)
            print("Test Predictions (confidence %):")
            for i, pred in enumerate(prediction_percentages):
                print(f"Sample {i+1}: {pred[0]}%")
        
        # Save the model
        model_filename = f'Model{data_type}_fixed.h5'
        model.save(model_filename)
        print(f"\nModel saved as {model_filename}")
        
        return model, history
        
    except Exception as e:
        print(f"Error during training: {e}")
        return None, None

def plot_training_history(history, data_type):
    """Plot training history"""
    if history is None:
        return
    
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{data_type}-Hour Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{data_type}-Hour Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'training_history_{data_type}h.png')
    plt.show()

def main():
    """Main training function"""
    print("Cloudburst Prediction Model Training")
    print("=" * 40)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Train both models
    for data_type in ['6', '12']:
        print(f"\n{'='*20} {data_type}-Hour Model {'='*20}")
        model, history = train_model(data_type)
        
        if model is not None:
            plot_training_history(history, data_type)
            print(f"{data_type}-hour model training completed successfully!")
        else:
            print(f"{data_type}-hour model training failed!")
        
        print("\n" + "="*60)
    
    print("Training complete!")

if __name__ == "__main__":
    main()
