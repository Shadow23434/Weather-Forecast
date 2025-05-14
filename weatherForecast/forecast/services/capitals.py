"""
This module contains information about country capitals and utilities to work with them.
"""

# Dictionary mapping country codes to capital cities
CAPITAL_CITIES = {
    "US": "Washington",
    "GB": "London",
    "FR": "Paris",
    "DE": "Berlin",
    "IT": "Rome",
    "ES": "Madrid",
    "JP": "Tokyo",
    "CN": "Beijing",
    "RU": "Moscow",
    "IN": "New Delhi",
    "BR": "Brasilia",
    "CA": "Ottawa",
    "AU": "Canberra",
    "MX": "Mexico City",
    "KR": "Seoul",
    "ID": "Jakarta",
    "TR": "Ankara",
    "SA": "Riyadh",
    "ZA": "Pretoria",
    "AR": "Buenos Aires",
    "TH": "Bangkok",
    "EG": "Cairo",
    "VN": "Hanoi",
    "PH": "Manila",
    "MY": "Kuala Lumpur",
    "PK": "Islamabad",
    "NG": "Abuja",
    "NO": "Oslo",
    "NZ": "Wellington",
    "SE": "Stockholm",
    "FI": "Helsinki",
    "DK": "Copenhagen",
    "AT": "Vienna",
    "BE": "Brussels",
    "CH": "Bern",
    "NL": "Amsterdam",
    "PT": "Lisbon",
    "GR": "Athens",
    "IE": "Dublin",
    "SG": "Singapore",
    "IL": "Jerusalem",
    "HK": "Hong Kong",
    "AE": "Abu Dhabi",
    "QA": "Doha",
    "KW": "Kuwait City",
    "OM": "Muscat",
    "BH": "Manama",
}

# Dictionary mapping capital cities to their coordinates (latitude, longitude)
CAPITAL_COORDINATES = {
    "Washington": (38.8951, -77.0364),  # US
    "London": (51.5074, -0.1278),       # GB
    "Paris": (48.8566, 2.3522),         # FR
    "Berlin": (52.5200, 13.4050),       # DE
    "Rome": (41.9028, 12.4964),         # IT
    "Madrid": (40.4168, -3.7038),       # ES
    "Tokyo": (35.6762, 139.6503),       # JP
    "Beijing": (39.9042, 116.4074),     # CN
    "Moscow": (55.7558, 37.6173),       # RU
    "New Delhi": (28.6139, 77.2090),    # IN
    "Brasilia": (-15.7942, -47.8822),   # BR
    "Ottawa": (45.4215, -75.6972),      # CA
    "Canberra": (-35.2809, 149.1300),   # AU
    "Mexico City": (19.4326, -99.1332), # MX
    "Seoul": (37.5665, 126.9780),       # KR
    "Jakarta": (-6.2088, 106.8456),     # ID
    "Ankara": (39.9334, 32.8597),       # TR
    "Riyadh": (24.7136, 46.6753),       # SA
    "Pretoria": (-25.7479, 28.2293),    # ZA
    "Buenos Aires": (-34.6037, -58.3816), # AR
    "Bangkok": (13.7563, 100.5018),     # TH
    "Cairo": (30.0444, 31.2357),        # EG
    "Hanoi": (21.0278, 105.8342),       # VN
    "Manila": (14.5995, 120.9842),      # PH
    "Kuala Lumpur": (3.1390, 101.6869), # MY
    "Islamabad": (33.6844, 73.0479),    # PK
    "Abuja": (9.0765, 7.3986),          # NG
    "Oslo": (59.9139, 10.7522),         # NO
    "Wellington": (-41.2865, 174.7762), # NZ
    "Stockholm": (59.3293, 18.0686),    # SE
    "Helsinki": (60.1699, 24.9384),     # FI
    "Copenhagen": (55.6761, 12.5683),   # DK
    "Vienna": (48.2082, 16.3738),       # AT
    "Brussels": (50.8503, 4.3517),      # BE
    "Bern": (46.9480, 7.4474),          # CH
    "Amsterdam": (52.3676, 4.9041),     # NL
    "Lisbon": (38.7223, -9.1393),       # PT
    "Athens": (37.9838, 23.7275),       # GR
    "Dublin": (53.3498, -6.2603),       # IE
    "Singapore": (1.3521, 103.8198),    # SG
    "Jerusalem": (31.7683, 35.2137),    # IL
    "Hong Kong": (22.3193, 114.1694),   # HK
    "Abu Dhabi": (24.4539, 54.3773),    # AE
    "Doha": (25.2854, 51.5310),         # QA
    "Kuwait City": (29.3759, 47.9774),  # KW
    "Muscat": (23.5880, 58.3829),       # OM
    "Manama": (26.2285, 50.5860),       # BH
}

def get_capital_city(country_code):
    """
    Get the capital city for a given country code.
    
    Args:
        country_code (str): ISO 3166-1 alpha-2 country code
        
    Returns:
        str: The capital city of the country or None if country code is not found
    """
    return CAPITAL_CITIES.get(country_code)

def get_capital_coordinates(city, country_code=None):
    """
    Get coordinates (latitude, longitude) for a capital city.
    
    Args:
        city (str): Capital city name
        country_code (str, optional): Country code to validate city is a capital
        
    Returns:
        tuple: (latitude, longitude) coordinates or (None, None) if not found
    """
    if country_code:
        # Validate that the city is actually the capital of the given country
        capital = get_capital_city(country_code)
        if capital != city:
            return None, None
    
    return CAPITAL_COORDINATES.get(city, (None, None))

def get_all_capital_cities():
    """
    Returns a list of all capital cities in the database.
    
    Returns:
        list: List of capital city names
    """
    return list(CAPITAL_CITIES.values())

def get_all_countries_and_capitals():
    """
    Returns a list of tuples containing (country_code, capital_city).
    
    Returns:
        list: List of tuples with country codes and their capital cities
    """
    return [(code, city) for code, city in CAPITAL_CITIES.items()]

def get_all_capitals_with_coordinates():
    """
    Returns a list of tuples containing (country_code, capital_city, latitude, longitude).
    
    Returns:
        list: List of tuples with country codes, capital cities, and their coordinates
    """
    result = []
    for country_code, capital in CAPITAL_CITIES.items():
        lat, lon = get_capital_coordinates(capital)
        if lat is not None and lon is not None:
            result.append((country_code, capital, lat, lon))
    return result 