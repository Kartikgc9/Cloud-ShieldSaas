import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


def generate_test_uttarakhand_data():
    """Generate test data that mimics current Uttarakhand weather patterns"""
    print("🌦️  Generating Test Uttarakhand Weather Data")
    print("=" * 50)

    stations = {
        'DEHRADUN': {'altitude': 'low', 'region': 'valley'},
        'NAINITAL': {'altitude': 'high', 'region': 'hills'},
        'HARIDWAR': {'altitude': 'low', 'region': 'plains'},
        'RISHIKESH': {'altitude': 'medium', 'region': 'foothills'},
        'TEHRI': {'altitude': 'medium', 'region': 'dam'},
        'CHAMOLI': {'altitude': 'high', 'region': 'mountains'},
        'PITHORAGARH': {'altitude': 'high', 'region': 'border'}
    }

    # Current date and time
    end_time = datetime.now()

    all_data = []

    for station, info in stations.items():
        print(f"Generating data for {station} ({info['region']}, {info['altitude']} altitude)...")

        # Generate 7 days of hourly data
        for hour in range(168):  # 7 days * 24 hours
            dt = end_time - timedelta(hours=hour)

            # Determine seasonal characteristics
            is_monsoon = dt.month in [6, 7, 8, 9]
            is_winter = dt.month in [12, 1, 2]

            # Altitude-based temperature adjustments
            if info['altitude'] == 'high':
                temp_base = 15 if is_monsoon else (5 if is_winter else 20)
                rainfall_multiplier = 1.5  # Higher rainfall in hills
            elif info['altitude'] == 'medium':
                temp_base = 22 if is_monsoon else (12 if is_winter else 25)
                rainfall_multiplier = 1.2
            else:  # low altitude
                temp_base = 28 if is_monsoon else (18 if is_winter else 30)
                rainfall_multiplier = 1.0

            # Time of day effects
            hour_of_day = dt.hour
            if 6 <= hour_of_day <= 18:
                temp_adjustment = 5
            else:
                temp_adjustment = -3

            # Generate weather parameters
            temperature = temp_base + temp_adjustment + np.random.normal(0, 3)

            # Rainfall probability based on season and region
            if is_monsoon:
                rainfall_prob = 0.6 if info['altitude'] == 'high' else 0.4
            else:
                rainfall_prob = 0.15 if info['altitude'] == 'high' else 0.1

            rainfall = 0
            if np.random.random() < rainfall_prob:
                if is_monsoon:
                    rainfall = np.random.exponential(5) * rainfall_multiplier
                else:
                    rainfall = np.random.exponential(1) * rainfall_multiplier

            # Humidity based on rainfall and season
            base_humidity = 85 if is_monsoon else 60
            if rainfall > 0:
                humidity = min(95, base_humidity + np.random.normal(10, 5))
            else:
                humidity = max(30, base_humidity + np.random.normal(0, 10))

            # Wind speed (higher during storms)
            if rainfall > 10:
                wind_speed = np.random.normal(12, 4)
            elif is_monsoon:
                wind_speed = np.random.normal(8, 3)
            else:
                wind_speed = np.random.normal(5, 2)
            wind_speed = max(0, wind_speed)

            # Pressure (lower during storms)
            base_pressure = 1013.25 - (info['altitude'] == 'high') * 50
            if rainfall > 5:
                pressure = base_pressure + np.random.normal(-5, 3)
            else:
                pressure = base_pressure + np.random.normal(0, 2)

            # Create record
            record = {
                'station': station,
                'datetime': dt,
                'rainfall': max(0, rainfall),
                'temperature': temperature,
                'humidity': np.clip(humidity, 20, 100),
                'wind_speed': np.clip(wind_speed, 0, 30),
                'pressure': pressure,
                'msl_pressure': pressure + np.random.normal(0, 1),
                'altitude': info['altitude'],
                'region': info['region'],
                'is_monsoon': is_monsoon,
                'hour_of_day': hour_of_day
            }

            all_data.append(record)

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    # Save data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_uttarakhand_data_{timestamp}.csv"
    df.to_csv(filename, index=False)

    print(f"\n✅ Generated {len(df)} weather records")
    print(f"📁 Saved to: {filename}")

    return df, filename


def analyze_generated_data(df):
    """Analyze the generated data"""
    print(f"\n📊 DATA ANALYSIS")
    print("=" * 50)

    print(f"Total records: {len(df)}")
    print(f"Stations: {', '.join(df['station'].unique())}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    print(f"\n🌧️  Rainfall Statistics:")
    rain_df = df[df['rainfall'] > 0]
    print(f"Records with rainfall: {len(rain_df)}")
    if not rain_df.empty:
        print(f"Average rainfall (when raining): {rain_df['rainfall'].mean():.1f} mm")
    print(f"Maximum rainfall: {df['rainfall'].max():.1f} mm")

    print(f"\n🌡️  Temperature Statistics:")
    print(f"Average temperature: {df['temperature'].mean():.1f}°C")
    print(f"Temperature range: {df['temperature'].min():.1f}°C to {df['temperature'].max():.1f}°C")

    print(f"\n💨 Wind Speed Statistics:")
    print(f"Average wind speed: {df['wind_speed'].mean():.1f} m/s")
    print(f"Maximum wind speed: {df['wind_speed'].max():.1f} m/s")

    print(f"\n🏔️  Station-wise Summary:")
    for station in df['station'].unique():
        station_data = df[df['station'] == station]
        rainy_days = len(station_data[station_data['rainfall'] > 0])
        avg_temp = station_data['temperature'].mean()
        max_rain = station_data['rainfall'].max()
        print(f"{station:12} | Rainy records: {rainy_days:3} | Avg temp: {avg_temp:5.1f}°C | Max rain: {max_rain:5.1f}mm")


def plot_weather_patterns(df, filename_base):
    """Create visualizations of weather patterns"""
    print(f"\n📈 Creating Weather Visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Uttarakhand Weather Patterns - Test Data', fontsize=16)

    # Plot 1: Rainfall by station
    station_rainfall = df.groupby('station')['rainfall'].sum().sort_values(ascending=False)
    axes[0, 0].bar(station_rainfall.index, station_rainfall.values)
    axes[0, 0].set_title('Total Rainfall by Station (7 days)')
    axes[0, 0].set_ylabel('Rainfall (mm)')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Plot 2: Temperature variation (24h sample for first 3)
    for station in df['station'].unique()[:3]:
        station_data = df[df['station'] == station].head(24)
        axes[0, 1].plot(
            station_data['hour_of_day'],
            station_data['temperature'],
            label=station,
            marker='o',
            markersize=3
        )
    axes[0, 1].set_title('Temperature Variation (24h sample)')
    axes[0, 1].set_xlabel('Hour of Day')
    axes[0, 1].set_ylabel('Temperature (°C)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Rainfall vs Humidity
    axes[1, 0].scatter(df['rainfall'], df['humidity'], alpha=0.6, s=20)
    axes[1, 0].set_title('Rainfall vs Humidity')
    axes[1, 0].set_xlabel('Rainfall (mm)')
    axes[1, 0].set_ylabel('Humidity (%)')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Wind speed distribution
    axes[1, 1].hist(df['wind_speed'], bins=20, alpha=0.7)
    axes[1, 1].set_title('Wind Speed Distribution')
    axes[1, 1].set_xlabel('Wind Speed (m/s)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_filename = f"{filename_base}_analysis.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Analysis plot saved: {plot_filename}")

    try:
        plt.show()
    except:
        pass


def identify_potential_cloudbursts(df):
    """Identify weather patterns that could indicate cloudburst conditions"""
    print(f"\n⚠️  POTENTIAL CLOUDBURST CONDITIONS")
    print("=" * 50)

    conditions = (
        (df['rainfall'] > 15) &
        (df['humidity'] > 80) &
        (df['wind_speed'] > 10) &
        (df['temperature'] > 25)
    )
    events = df[conditions]

    if not events.empty:
        print(f"🚨 Found {len(events)} potential cloudburst conditions:")
        for idx, event in events.head(10).iterrows():
            print(
                f"   {event['station']:12} | "
                f"{event['datetime'].strftime('%Y-%m-%d %H:%M')} | "
                f"Rain: {event['rainfall']:5.1f}mm | "
                f"Temp: {event['temperature']:4.1f}°C | "
                f"Wind: {event['wind_speed']:4.1f}m/s | "
                f"Humidity: {event['humidity']:4.1f}%"
            )
    else:
        print("✅ No extreme weather conditions detected in test data")

    return events


def main():
    """Main test function"""
    print("🧪 UTTARAKHAND WEATHER DATA COLLECTION TEST")
    print("=" * 60)

    df, filename = generate_test_uttarakhand_data()
    analyze_generated_data(df)
    filename_base = filename.replace('.csv', '')
    plot_weather_patterns(df, filename_base)
    identify_potential_cloudbursts(df)

    print(f"\n✅ TEST COMPLETE!")
    print(f"📁 Files created:")
    print(f"   - {filename} (weather data)")
    print(f"   - {filename_base}_analysis.png (analysis plot)")
    print("\n🔄 Next Steps:")
    print("   1. Review the generated data in", filename)
    print("   2. Run actual data collection: python data_updater.py")
    print("   3. Add real cloudburst labels to improve accuracy")
    print("   4. Retrain models: python retrain_model.py")


if __name__ == "__main__":
    main()
