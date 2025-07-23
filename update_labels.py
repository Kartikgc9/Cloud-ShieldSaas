import numpy as np
import pandas as pd
import glob
import os

def update_cloudburst_labels():
    """
    A utility to manually update cloudburst labels for the most recent dataset.
    """
    print("☁️  Cloudburst Label Updater")
    print("=============================")

    # --- 1. Find the latest data files ---
    try:
        # Find the latest weather data CSV to get timestamps
        csv_files = glob.glob('uttarakhand_weather_data_*.csv')
        if not csv_files:
            print("❌ Error: No weather data CSV files found. Run data_updater.py first.")
            return
        latest_csv = max(csv_files, key=os.path.getctime)
        
        # Find the corresponding labels file
        timestamp = os.path.basename(latest_csv).split('_')[3]
        labels_file_pattern = f"uttarakhand_labels_{timestamp[:9]}*.npy" # Match date and hour
        
        possible_label_files = glob.glob(labels_file_pattern)
        if not possible_label_files:
             print(f"❌ Error: Could not find a matching labels file for CSV: {latest_csv}")
             return
        latest_labels_file = max(possible_label_files, key=os.path.getctime)

        print(f"Found latest weather data: {os.path.basename(latest_csv)}")
        print(f"Found corresponding labels file: {os.path.basename(latest_labels_file)}\n")

    except Exception as e:
        print(f"❌ Error finding latest data files: {e}")
        return

    # --- 2. Load the data ---
    try:
        weather_df = pd.read_csv(latest_csv, parse_dates=['datetime'])
        labels = np.load(latest_labels_file)
        print(f"Loaded {len(labels)} labels. Currently, {np.sum(labels)} are marked as cloudbursts.")
    except Exception as e:
        print(f"❌ Error loading data files: {e}")
        return

    # --- 3. User input loop ---
    while True:
        print("\nEnter the date and time of a known cloudburst event.")
        print("Format: YYYY-MM-DD HH:MM (e.g., 2024-07-15 14:30)")
        print("Type 'save' to finish and save your changes, or 'exit' to cancel.")
        
        user_input = input("> ").strip()

        if user_input.lower() == 'save':
            break
        if user_input.lower() == 'exit':
            print("Changes discarded.")
            return

        try:
            # Parse the user's date and find the closest timestamp in the data
            event_time = pd.to_datetime(user_input)
            
            # Find the index of the closest time in the DataFrame
            time_diff = (weather_df['datetime'] - event_time).abs()
            closest_index = time_diff.idxmin()
            
            actual_time = weather_df.loc[closest_index, 'datetime']
            
            print(f"Found closest timestamp in data: {actual_time}")
            
            # --- 4. Update the label ---
            confirm = input(f"Mark this timestamp as a cloudburst event? (y/n): ").strip().lower()
            if confirm == 'y':
                labels[closest_index] = 1
                print(f"✅ Label at index {closest_index} updated to 1.")
            else:
                print("Skipped.")

        except ValueError:
            print("❌ Invalid date format. Please use 'YYYY-MM-DD HH:MM'.")
        except Exception as e:
            print(f"❌ An error occurred: {e}")

    # --- 5. Save the updated labels ---
    try:
        np.save(latest_labels_file, labels)
        print("\n💾 Labels successfully updated and saved!")
        print(f"Total cloudbursts in file is now: {np.sum(labels)}.")
    except Exception as e:
        print(f"❌ Error saving the updated labels file: {e}")

if __name__ == "__main__":
    update_cloudburst_labels() 