import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import glob
import os

class ModelRetrainer:
    def __init__(self):
        self.model_configs = {
            '6h': {
                'filters': [128, 256, 512],
                'kernel_sizes': [(4, 4), (4, 4), (4, 4)],
                'optimizer': Adam(learning_rate=0.001),
                'dropout_rate': 0.5
            },
            '12h': {
                'filters': [128, 256, 512],
                'kernel_sizes': [(2, 2), (2, 2), (2, 2)],
                'optimizer': SGD(learning_rate=0.01, momentum=0.9),
                'dropout_rate': 0.3
            }
        }
    
    def load_existing_data(self):
        """Load the original training data"""
        print("Loading existing training data...")
        
        try:
            # Load original data
            cb6 = np.load('CB6.npy')
            cb12 = np.load('CB12.npy')
            nb = np.load('NB.npy')
            
            print(f"✓ Loaded original data:")
            print(f"  - CB6: {cb6.shape}")
            print(f"  - CB12: {cb12.shape}")
            print(f"  - NB: {nb.shape}")
            
            return cb6, cb12, nb
            
        except FileNotFoundError as e:
            print(f"✗ Error loading original data: {e}")
            print("Make sure CB6.npy, CB12.npy, and NB.npy are in the current directory")
            return None, None, None
    
    def load_new_uttarakhand_data(self):
        """Load newly collected Uttarakhand data"""
        print("\nLooking for new Uttarakhand data...")
        
        # Find the latest GAF data files
        gaf_files = glob.glob('uttarakhand_gaf_data_*.npy')
        label_files = glob.glob('uttarakhand_labels_*.npy')
        
        if not gaf_files or not label_files:
            print("⚠️  No new Uttarakhand data found.")
            print("   Run 'python data_updater.py' first to collect current data.")
            return None, None
        
        # Get the latest files
        latest_gaf = max(gaf_files, key=os.path.getctime)
        latest_labels = max(label_files, key=os.path.getctime)
        
        try:
            new_gaf = np.load(latest_gaf)
            new_labels = np.load(latest_labels)
            
            print(f"✓ Loaded new Uttarakhand data:")
            print(f"  - GAF data: {new_gaf.shape} from {latest_gaf}")
            print(f"  - Labels: {new_labels.shape} from {latest_labels}")
            print(f"  - Positive samples: {np.sum(new_labels)}")
            print(f"  - Negative samples: {len(new_labels) - np.sum(new_labels)}")
            
            return new_gaf, new_labels
            
        except Exception as e:
            print(f"✗ Error loading new data: {e}")
            return None, None
    
    def combine_datasets(self, cb_old, nb_old, new_gaf, new_labels):
        """Combine old and new datasets"""
        print("\n🔄 Combining datasets...")
        
        # Separate new data by labels
        cloudburst_mask = new_labels == 1
        new_cb = new_gaf[cloudburst_mask]
        new_nb = new_gaf[~cloudburst_mask]
        
        print(f"New data breakdown:")
        print(f"  - New cloudburst samples: {len(new_cb)}")
        print(f"  - New non-cloudburst samples: {len(new_nb)}")
        
        # Combine datasets
        if len(new_cb) > 0:
            combined_cb = np.concatenate([cb_old, new_cb], axis=0)
        else:
            combined_cb = cb_old
            
        if len(new_nb) > 0:
            combined_nb = np.concatenate([nb_old, new_nb], axis=0)
        else:
            combined_nb = nb_old
        
        print(f"\nCombined dataset sizes:")
        print(f"  - Total cloudburst samples: {len(combined_cb)}")
        print(f"  - Total non-cloudburst samples: {len(combined_nb)}")
        
        return combined_cb, combined_nb
    
    def prepare_training_data(self, cb_data, nb_data):
        """Prepare training and validation sets"""
        print("\n📊 Preparing training data...")
        
        # Create labels
        cb_labels = np.ones(len(cb_data))
        nb_labels = np.zeros(len(nb_data))
        
        # Combine data and labels
        X = np.concatenate([cb_data, nb_data], axis=0)
        y = np.concatenate([cb_labels, nb_labels], axis=0)
        
        # Shuffle data
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # Split into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✓ Training data prepared:")
        print(f"  - Training samples: {len(X_train)} (CB: {np.sum(y_train)}, NB: {len(y_train) - np.sum(y_train)})")
        print(f"  - Validation samples: {len(X_val)} (CB: {np.sum(y_val)}, NB: {len(y_val) - np.sum(y_val)})")
        print(f"  - Data shape: {X_train.shape}")
        
        return X_train, X_val, y_train, y_val
    
    def create_model(self, input_shape, config_key):
        """Create CNN model with specified configuration"""
        config = self.model_configs[config_key]
        
        model = Sequential([
            Conv2D(config['filters'][0], config['kernel_sizes'][0],
                   activation='relu', input_shape=input_shape),
            MaxPooling2D((3, 3)),
            
            Conv2D(config['filters'][1], config['kernel_sizes'][1],
                   activation='relu'),
            MaxPooling2D((1, 1)),
            
            Conv2D(config['filters'][2], config['kernel_sizes'][2],
                   activation='relu'),
            MaxPooling2D((3, 3)),
            
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(config['dropout_rate']),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=config['optimizer'],
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, X_train, X_val, y_train, y_val, model_type='6h'):
        """Train the model with current data"""
        print(f"\n🚀 Training {model_type} model...")
        
        # Create model
        input_shape = X_train.shape[1:]  # (6, 256, 256)
        model = self.create_model(input_shape, model_type)
        
        print(f"Model architecture for {model_type}:")
        model.summary()
        
        # Callbacks
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f'Model{model_type}_{timestamp}.h5'
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint(model_filename, save_best_only=True, monitor='val_accuracy'),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=100,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Load best model
        best_model = load_model(model_filename)
        
        # Evaluate
        train_loss, train_acc = best_model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc = best_model.evaluate(X_val, y_val, verbose=0)
        
        print(f"\n📈 {model_type} Model Results:")
        print(f"  - Training Accuracy: {train_acc:.4f}")
        print(f"  - Validation Accuracy: {val_acc:.4f}")
        print(f"  - Model saved as: {model_filename}")
        
        # Detailed evaluation
        y_pred = (best_model.predict(X_val) > 0.5).astype(int).flatten()
        
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_val, y_pred,
                                     target_names=['Non-Cloudburst', 'Cloudburst']))
        
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_val, y_pred))
        
        # Plot training history
        self.plot_training_history(history, model_type, timestamp)
        
        return best_model, model_filename, history
    
    def plot_training_history(self, history, model_type, timestamp):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'{model_type} Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{model_type} Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'training_history_{model_type}_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def update_models(self):
        """Main function to update both models"""
        print("🔄 UPDATING CLOUDBURST PREDICTION MODELS")
        print("="*60)
        
        # Load existing data
        cb6, cb12, nb = self.load_existing_data()
        if cb6 is None:
            return
        
        # Load new Uttarakhand data
        new_gaf, new_labels = self.load_new_uttarakhand_data()
        if new_gaf is None:
            print("\n⚠️  Proceeding with existing data only...")
            cb6_combined, nb_combined = cb6, nb
            cb12_combined = cb12
        else:
            # Combine datasets
            cb6_combined, nb_combined = self.combine_datasets(cb6, nb, new_gaf, new_labels)
            cb12_combined, _ = self.combine_datasets(cb12, nb, new_gaf, new_labels)
        
        # Train 6-hour model
        print(f"\n{'='*60}")
        print("TRAINING 6-HOUR MODEL")
        print(f"{'='*60}")
        
        X_train_6h, X_val_6h, y_train_6h, y_val_6h = self.prepare_training_data(cb6_combined, nb_combined)
        model_6h, filename_6h, history_6h = self.train_model(X_train_6h, X_val_6h, y_train_6h, y_val_6h, '6h')
        
        # Train 12-hour model
        print(f"\n{'='*60}")
        print("TRAINING 12-HOUR MODEL")
        print(f"{'='*60}")
        
        X_train_12h, X_val_12h, y_train_12h, y_val_12h = self.prepare_training_data(cb12_combined, nb_combined)
        model_12h, filename_12h, history_12h = self.train_model(X_train_12h, X_val_12h, y_train_12h, y_val_12h, '12h')
        
        print(f"\n✅ MODEL RETRAINING COMPLETE!")
        print(f"📁 New model files:")
        print(f"   - 6-hour model: {filename_6h}")
        print(f"   - 12-hour model: {filename_12h}")
        
        print(f"\n🔄 Next steps:")
        print(f"   1. Test new models: python test_updated_models.py")
        print(f"   2. Replace old models if performance is better")
        print(f"   3. Deploy updated models for predictions")
        
        return {
            '6h': {'model': model_6h, 'filename': filename_6h, 'history': history_6h},
            '12h': {'model': model_12h, 'filename': filename_12h, 'history': history_12h}
        }

def main():
    """Main execution function"""
    retrainer = ModelRetrainer()
    retrainer.update_models()

if __name__ == "__main__":
    main()
