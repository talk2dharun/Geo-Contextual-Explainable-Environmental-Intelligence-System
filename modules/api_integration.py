"""
GEEIS - API Integration Module
Handles fetching data from OpenWeather, NASA, and NewsAPI.
"""

import os
import requests
import json
from datetime import datetime


# API Keys
# Read API keys from environment variables to avoid committing secrets to git.
# Expected env var names:
# - OPENWEATHER_API_KEY
# - NASA_API_KEY
# - NEWS_API_KEY
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
NASA_API_KEY = os.getenv("NASA_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")


def fetch_weather_data(city_name):
    """
    Fetch current weather data for a city using OpenWeather API.
    Returns dict with temperature, humidity, pressure, wind_speed, 
    description, rainfall, etc.
    """
    try:
        if not OPENWEATHER_API_KEY:
            return {
                'status': 'error',
                'message': 'Missing OPENWEATHER_API_KEY (set it as an environment variable).',
                'city': city_name
            }
        url = "http://api.openweathermap.org/data/2.5/weather"
        params = {
            'q': city_name,
            'appid': OPENWEATHER_API_KEY,
            'units': 'metric'
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        weather_info = {
            'city': data.get('name', city_name),
            'country': data.get('sys', {}).get('country', 'N/A'),
            'temperature': data.get('main', {}).get('temp', 0),
            'feels_like': data.get('main', {}).get('feels_like', 0),
            'humidity': data.get('main', {}).get('humidity', 0),
            'pressure': data.get('main', {}).get('pressure', 0),
            'wind_speed': data.get('wind', {}).get('speed', 0),
            'description': data.get('weather', [{}])[0].get('description', 'N/A'),
            'icon': data.get('weather', [{}])[0].get('icon', ''),
            'visibility': data.get('visibility', 0),
            'clouds': data.get('clouds', {}).get('all', 0),
            'rainfall': data.get('rain', {}).get('1h', 0),
            'coordinates': {
                'lat': data.get('coord', {}).get('lat', 0),
                'lon': data.get('coord', {}).get('lon', 0)
            },
            'timestamp': datetime.fromtimestamp(
                data.get('dt', 0)
            ).strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'success'
        }

        return weather_info

    except requests.exceptions.RequestException as e:
        return {
            'status': 'error',
            'message': f"Weather API Error: {str(e)}",
            'city': city_name
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': f"Unexpected error: {str(e)}",
            'city': city_name
        }


def fetch_nasa_environmental_data(lat, lon):
    """
    Fetch environmental/satellite data from NASA API.
    Uses NASA EONET (Earth Observatory Natural Event Tracker) 
    and EPIC (Earth Polychromatic Imaging Camera) APIs.
    Returns environmental context information.
    """
    environmental_data = {
        'status': 'success',
        'natural_events': [],
        'earth_imagery': None,
        'atmospheric_data': {}
    }

    # Fetch natural events (EONET)
    try:
        eonet_url = "https://eonet.gsfc.nasa.gov/api/v3/events"
        params = {
            'status': 'open',
            'limit': 10
        }
        response = requests.get(eonet_url, params=params, timeout=15)
        response.raise_for_status()
        events_data = response.json()

        events = events_data.get('events', [])
        for event in events[:5]:
            event_info = {
                'title': event.get('title', 'N/A'),
                'category': event.get('categories', [{}])[0].get('title', 'N/A'),
                'date': event.get('geometry', [{}])[0].get('date', 'N/A') if event.get('geometry') else 'N/A',
                'source': event.get('sources', [{}])[0].get('url', 'N/A') if event.get('sources') else 'N/A'
            }
            environmental_data['natural_events'].append(event_info)

    except Exception as e:
        environmental_data['natural_events_error'] = str(e)

    # Fetch Earth imagery (NASA Planetary API)
    try:
        imagery_url = "https://api.nasa.gov/planetary/earth/assets"
        if not NASA_API_KEY:
            environmental_data['earth_imagery'] = None
            environmental_data['earth_imagery_error'] = 'Missing NASA_API_KEY (set it as an environment variable).'
        else:
            params = {
                'lon': lon,
                'lat': lat,
                'dim': 0.1,
                'api_key': NASA_API_KEY
            }
            response = requests.get(imagery_url, params=params, timeout=15)
            if response.status_code == 200:
                imagery_data = response.json()
                environmental_data['earth_imagery'] = {
                    'date': imagery_data.get('date', 'N/A'),
                    'url': imagery_data.get('url', ''),
                    'resource_type': imagery_data.get('resource', {}).get('type', 'N/A')
                }
    except Exception as e:
        environmental_data['earth_imagery_error'] = str(e)

    # Fetch APOD (Astronomy Picture of the Day) - as a proxy for NASA data quality
    try:
        apod_url = "https://api.nasa.gov/planetary/apod"
        if not NASA_API_KEY:
            environmental_data['atmospheric_data'] = {
                'nasa_status': 'Limited',
                'error': 'Missing NASA_API_KEY (set it as an environment variable).'
            }
        else:
            params = {'api_key': NASA_API_KEY}
            response = requests.get(apod_url, params=params, timeout=10)
            if response.status_code == 200:
                apod_data = response.json()
                environmental_data['atmospheric_data'] = {
                    'nasa_status': 'API Connected',
                    'latest_observation': apod_data.get('date', 'N/A'),
                    'satellite_coverage': 'Active',
                    'data_quality': 'High'
                }
    except Exception as e:
        environmental_data['atmospheric_data'] = {
            'nasa_status': 'Limited',
            'error': str(e)
        }

    return environmental_data


def fetch_news_data(keywords=None):
    """
    Fetch water pollution related news using NewsAPI.
    Returns list of article summaries.
    """
    if keywords is None:
        keywords = ["water pollution", "river contamination", "drinking water quality"]

    if not NEWS_API_KEY:
        return [{
            'title': 'Error fetching news',
            'description': 'Missing NEWS_API_KEY (set it as an environment variable).',
            'source': 'Error',
            'url': '',
            'published_at': '',
            'content': '',
            'keyword': kw
        } for kw in keywords]

    all_articles = []

    for keyword in keywords:
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': keyword,
                'apiKey': NEWS_API_KEY,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 5
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            articles = data.get('articles', [])
            for article in articles:
                article_info = {
                    'title': article.get('title', 'N/A'),
                    'description': article.get('description', 'N/A'),
                    'source': article.get('source', {}).get('name', 'N/A'),
                    'url': article.get('url', ''),
                    'published_at': article.get('publishedAt', 'N/A'),
                    'content': article.get('content', ''),
                    'keyword': keyword
                }
                all_articles.append(article_info)

        except requests.exceptions.RequestException as e:
            all_articles.append({
                'title': f'Error fetching news for: {keyword}',
                'description': str(e),
                'source': 'Error',
                'url': '',
                'published_at': '',
                'content': '',
                'keyword': keyword
            })
        except Exception as e:
            all_articles.append({
                'title': f'Error: {keyword}',
                'description': str(e),
                'source': 'Error',
                'url': '',
                'published_at': '',
                'content': '',
                'keyword': keyword
            })

    return all_articles
