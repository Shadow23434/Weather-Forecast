"""
Django management command to fetch historical weather data for capital cities.
"""

from django.core.management.base import BaseCommand
from forecast.services.historical_weather import fetch_all_capitals_historical_data, fetch_maximum_historical_data
import time

class Command(BaseCommand):
    help = 'Fetch historical weather data for capital cities of all supported countries'

    def add_arguments(self, parser):
        parser.add_argument(
            '--days', 
            type=int, 
            default=5,
            help='Number of days back to fetch data for (max 5)'
        )
        parser.add_argument(
            '--max', 
            action='store_true',
            help='Fetch maximum possible historical data (5 days)'
        )

    def handle(self, *args, **options):
        start_time = time.time()
        
        if options['max']:
            self.stdout.write(self.style.WARNING('Fetching maximum historical data (5 days) for all capital cities...'))
            file_path = fetch_maximum_historical_data()
        else:
            days = min(options['days'], 5)  # Limit to 5 days as per API limitation
            self.stdout.write(self.style.WARNING(f'Fetching {days} days of historical data for all capital cities...'))
            file_path = fetch_all_capitals_historical_data(days_back=days)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        self.stdout.write(self.style.SUCCESS(f'Successfully fetched historical data in {execution_time:.2f} seconds'))
        self.stdout.write(self.style.SUCCESS(f'Data saved to {file_path}')) 