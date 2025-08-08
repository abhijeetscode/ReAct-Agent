from typing import Dict, Any


def get_weather(city: str, country_code: str = "US") -> Dict[str, Any]:
    """
    Get current weather information for a specific city.
    
    Args:
        city (str): Name of the city
        country_code (str): ISO country code (default: "US")
    
    Returns:
        dict: Weather information including temperature, description, humidity, etc.
    """
    # Mock implementation - replace with real API call
    mock_weather_data = {
        "city": city,
        "country": country_code,
        "temperature": 22,
        "temperature_unit": "celsius",
        "description": "partly cloudy",
        "humidity": 65,
        "wind_speed": 12,
        "wind_unit": "km/h"
    }
    return mock_weather_data


def calculate_compound_interest(principal: float, rate: float, time: int, compound_frequency: int = 1) -> Dict[str, Any]:
    """
    Calculate compound interest and return detailed breakdown.
    
    Args:
        principal (float): Initial amount invested
        rate (float): Annual interest rate (as percentage, e.g., 5 for 5%)
        time (int): Time period in years
        compound_frequency (int): How many times interest is compounded per year (default: 1)
    
    Returns:
        dict: Calculation results including final amount, interest earned, and breakdown
    """
    rate_decimal = rate / 100
    final_amount = principal * ((1 + rate_decimal / compound_frequency) ** (compound_frequency * time))
    interest_earned = final_amount - principal
    
    return {
        "principal": principal,
        "rate_percent": rate,
        "time_years": time,
        "compound_frequency": compound_frequency,
        "final_amount": round(final_amount, 2),
        "interest_earned": round(interest_earned, 2),
        "total_return_percent": round((interest_earned / principal) * 100, 2)
    }