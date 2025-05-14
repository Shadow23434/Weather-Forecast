"""
Example script demonstrating how to use the historical weather data tool directly.

Run this script from the root directory of the project:
python script_example.py
"""

import os
import sys
import django
import pandas as pd

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'weatherForecast.settings')
django.setup()

# Import after Django setup
from forecast.services.historical_weather import (
    fetch_maximum_historical_data,
    fetch_all_capitals_historical_data,
    fetch_capital_historical_data,
    load_historical_data
)
from forecast.services.capitals import get_all_countries_and_capitals

def main():
    print("Welcome to the Historical Weather Data Tool")
    print("===========================================")
    
    # Display available options
    print("\nAvailable options:")
    print("1. Fetch maximum historical data for all capital cities (5 days)")
    print("2. Fetch custom period historical data for all capital cities")
    print("3. Fetch data for a specific capital city")
    print("4. Load and display most recent historical data")
    print("0. Exit")
    
    choice = input("\nEnter your choice (0-4): ")
    
    if choice == '1':
        print("\nFetching maximum historical data (5 days) for all capital cities...")
        file_path = fetch_maximum_historical_data()
        print(f"Data saved to {file_path}")
    
    elif choice == '2':
        try:
            days = int(input("Enter number of days back to fetch (max 5): "))
            days = min(days, 5)  # Limit to 5 days
            print(f"\nFetching {days} days of historical data for all capital cities...")
            file_path = fetch_all_capitals_historical_data(days_back=days)
            print(f"Data saved to {file_path}")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    elif choice == '3':
        # Display available countries and their codes
        print("\nAvailable countries and their capital cities:")
        countries_and_capitals = get_all_countries_and_capitals()
        for i, (code, capital) in enumerate(countries_and_capitals):
            print(f"{code}: {capital}".ljust(20), end='\t')
            if (i + 1) % 3 == 0:
                print()  # Newline every 3 countries
        print()
        
        country_code = input("\nEnter country code (e.g., US, GB, FR): ").upper()
        days = min(int(input("Enter number of days back to fetch (max 5): ") or "5"), 5)
        
        print(f"\nFetching {days} days of historical data for the capital of {country_code}...")
        data = fetch_capital_historical_data(country_code, days_back=days)
        
        if not data:
            print(f"No data found for country code: {country_code}")
        else:
            # Convert to DataFrame and display summary
            df = pd.DataFrame(data)
            print(f"\nFetched {len(df)} records.")
            print("\nData summary:")
            print(df[['city', 'country', 'date', 'time', 'temp']].head())
            
            # Save to CSV
            timestamp = df['timestamp'].iloc[0] if not df.empty else "unknown"
            file_path = f"data/{country_code}_historical_{timestamp}.csv"
            df.to_csv(file_path, index=False)
            print(f"Data saved to {file_path}")
    
    elif choice == '4':
        print("\nLoading most recent historical data...")
        df = load_historical_data()
        
        if df.empty:
            print("No historical data files found.")
        else:
            # Display summary statistics
            print(f"Loaded {len(df)} records.")
            print("\nCountries and cities in dataset:")
            city_country = df[['city', 'country']].drop_duplicates()
            for _, row in city_country.iterrows():
                print(f"- {row['city']}, {row['country']}")
            
            print("\nDate range:")
            min_date, max_date = df['date'].min(), df['date'].max()
            print(f"From {min_date} to {max_date}")
            
            print("\nTemperature statistics:")
            stats = df.groupby('city')['temp'].agg(['mean', 'min', 'max'])
            print(stats.head())
    
    elif choice == '0':
        print("\nExiting...")
        sys.exit(0)
    
    else:
        print("\nInvalid choice. Exiting...")

if __name__ == "__main__":
    main() 