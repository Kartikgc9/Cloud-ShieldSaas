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
import gc

class ModelRetrainer:
    def __init__(self):
        # Configure TensorFlow for memory efficiency
        self.configure_tensorflow()
        
        # Only 12h model configuration with memory optimizations
        self.model_config = {
            'filters': [32, 64, 128],  # Reduced for memory efficiency
            'kernel_sizes': [(2, 2), (2, 2), (2, 2)],
            'optimizer': Adam(learning_rate=0.0005),  # Changed from SGD to Adam
            'dropout_rate': 0.3,
            'dense_units': 64  # Reduced from 256
        }

    def configure_tensorflow(self):
        """Configure TensorFlow for memory efficiency"""
        try:
            # Set memory growth for GPU if available
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set CPU memory limit
            tf.config.threading.set_inter_op_parallelism_threads(2)
            tf.config.threading.set_intra_op_parallelism_threads(2)
            
        except RuntimeError as e:
            print(f"TensorFlow configuration warning: {e}")

    def load_existing_data(self):
        """Load the original training data with memory optimization"""
        print("Loading existing training data...")
        try:
            # Load only 12h and NB data (skip CB6 since we're not training 6h model)
            cb12 = np.load('CB12.npy')
            nb = np.load('NB.npy')

            # Transpose to channels_last format and convert to float32
            cb12 = np.transpose(cb12, (0, 2, 3, 1)).astype(np.float32)
            nb = np.transpose(nb, (0, 2, 3, 1)).astype(np.float32)

            # Normalize data to [0, 1] range
            cb12 = cb12 / 255.0 if cb12.max() > 1.0 else cb12
            nb = nb / 255.0 if nb.max() > 1.0 else nb

            print(f"✓ Loaded and preprocessed original data:")
            print(f"  - CB12: {cb12.shape}")
            print(f"  - NB: {nb.shape}")

            return cb12, nb

        except FileNotFoundError as e:
            print(f"✗ Error loading original data: {e}")
            print("Make sure CB12.npy and NB.npy are in the current directory")
            return None, None

    def load_new_uttarakhand_data(self):
        """Load newly collected Uttarakhand data"""
        print("\nLooking for new Uttarakhand data...")
        
        gaf_files = glob.glob('uttarakhand_gaf_data_*.npy')
        label_files = glob.glob('uttarakhand_labels_*.npy')

        if not gaf_files or not label_files:
            print("⚠️ No new Uttarakhand data found.")
            print("  Run 'python data_updater.py' first to collect current data.")
            return None, None

        latest_gaf = max(gaf_files, key=os.path.getctime)
        latest_labels = max(label_files, key=os.path.getctime)

        try:
            new_gaf = np.load(latest_gaf)
            new_labels = np.load(latest_labels)

            # Transpose and normalize
            new_gaf = np.transpose(new_gaf, (0, 2, 3, 1)).astype(np.float32)
            new_gaf = new_gaf / 255.0 if new_gaf.max() > 1.0 else new_gaf

            print(f"✓ Loaded and preprocessed new Uttarakhand data:")
            print(f"  - GAF data: {new_gaf.shape} from {latest_gaf}")
            print(f"  - Labels: {new_labels.shape} from {latest_labels}")
            print(f"  - Positive samples: {np.sum(new_labels)}")
            print(f"  - Negative samples: {len(new_labels) - np.sum(new_labels)}")

            return new_gaf, new_labels

        except Exception as e:
            print(f"✗ Error loading new data: {e}")
            return None, None

    def combine_datasets(self, cb_old, nb_old, new_gaf, new_labels):
        """Combine old and new datasets with memory management"""
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

        # Clear intermediate variables
        del new_cb, new_nb
        gc.collect()

        print(f"\nCombined dataset sizes:")
        print(f"  - Total cloudburst samples: {len(combined_cb)}")
        print(f"  - Total non-cloudburst samples: {len(combined_nb)}")

        return combined_cb, combined_nb

    def prepare_training_data(self, cb_data, nb_data):
        """Prepare training and validation sets with memory optimization"""
        print("\n📊 Preparing training data...")

        # Create labels
        cb_labels = np.ones(len(cb_data), dtype=np.float32)
        nb_labels = np.zeros(len(nb_data), dtype=np.float32)

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

    def create_model(self, input_shape):
        """Create CNN model with memory-efficient configuration"""
        config = self.model_config
        
        model = Sequential([
            Conv2D(config['filters'][0], config['kernel_sizes'][0],
                   activation='relu', input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(config['filters'][1], config['kernel_sizes'][1],
                   activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(config['filters'][2], config['kernel_sizes'][2],
                   activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(config['dense_units'], activation='relu'),
            Dropout(config['dropout_rate']),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=config['optimizer'],
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train_12h_model(self, X_train, X_val, y_train, y_val):
        """Train only the 12-hour model with memory optimization"""
        print(f"\n🚀 Training 12-hour model...")

        # Clear any previous models
        tf.keras.backend.clear_session()
        gc.collect()

        # Create model
        input_shape = X_train.shape[1:]
        model = self.create_model(input_shape)

        print(f"Model architecture for 12h:")
        model.summary()

        # Callbacks
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f'Model12h_{timestamp}.h5'
        
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
            ModelCheckpoint(model_filename, save_best_only=True, monitor='val_accuracy', verbose=1),
            ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-7, verbose=1)
        ]

        # Use very small batch size for memory efficiency
        batch_size = 2

        try:
            # Train model with memory-efficient settings
            history = model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=50,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1,
                shuffle=True
            )

            # Load best model
            best_model = load_model(model_filename)

            # Evaluate
            train_loss, train_acc = best_model.evaluate(X_train, y_train, verbose=0, batch_size=batch_size)
            val_loss, val_acc = best_model.evaluate(X_val, y_val, verbose=0, batch_size=batch_size)

            print(f"\n📈 12-Hour Model Results:")
            print(f"  - Training Accuracy: {train_acc:.4f}")
            print(f"  - Validation Accuracy: {val_acc:.4f}")
            print(f"  - Model saved as: {model_filename}")

            # Detailed evaluation
            y_pred = (best_model.predict(X_val, batch_size=batch_size) > 0.5).astype(int).flatten()
            print(f"\nDetailed Classification Report:")
            print(classification_report(y_val, y_pred, target_names=['Non-Cloudburst', 'Cloudburst']))

            print(f"\nConfusion Matrix:")
            print(confusion_matrix(y_val, y_pred))

            # Plot training history
            self.plot_training_history(history, timestamp)

            return best_model, model_filename, history

        except Exception as e:
            print(f"❌ Training failed: {e}")
            return None, None, None

    def plot_training_history(self, history, timestamp):
        """Plot training history"""
        plt.figure(figsize=(12, 4))

        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('12-Hour Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('12-Hour Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'training_history_12h_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def update_12h_model_only(self):
        """Main function to update only the 12-hour model"""
        print("🔄 UPDATING 12-HOUR CLOUDBURST PREDICTION MODEL")
        print("="*60)

        # Load existing data (only CB12 and NB)
        cb12, nb = self.load_existing_data()
        if cb12 is None:
            return

        # Load new Uttarakhand data
        new_gaf, new_labels = self.load_new_uttarakhand_data()

        if new_gaf is None:
            print("\n⚠️ Proceeding with existing data only...")
            cb12_combined, nb_combined = cb12, nb
        else:
            # Combine datasets
            cb12_combined, nb_combined = self.combine_datasets(cb12, nb, new_gaf, new_labels)

        # Train 12-hour model only
        print(f"\n{'='*60}")
        print("TRAINING 12-HOUR MODEL")
        print(f"{'='*60}")
        
        X_train_12h, X_val_12h, y_train_12h, y_val_12h = self.prepare_training_data(cb12_combined, nb_combined)
        model_12h, filename_12h, history_12h = self.train_12h_model(X_train_12h, X_val_12h, y_train_12h, y_val_12h)
        
        if model_12h is not None:
            print(f"\n✅ 12-HOUR MODEL RETRAINING COMPLETE!")
            print(f"📁 New model file: {filename_12h}")
            
            print(f"\n🔄 Next steps:")
            print(f"  1. Test new model: python test_12h_model.py")
            print(f"  2. Replace old Model12.h5 if performance is better")
            print(f"  3. Deploy updated model for predictions")
            
            return {'model': model_12h, 'filename': filename_12h, 'history': history_12h}
        else:
            print(f"\n❌ 12-HOUR MODEL TRAINING FAILED!")
            return None

def main():
    """Main execution function"""
    retrainer = ModelRetrainer()
    retrainer.update_12h_model_only()

if __name__ == "__main__":
    main()
