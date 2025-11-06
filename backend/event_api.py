# backend/event_api.py
import os
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("EVENTBRITE_TOKEN")  # add to .env

BASE = "https://www.eventbriteapi.com/v3/"

DEFAULT_HEADERS = {
    "Authorization": f"Bearer {TOKEN}"
}

def search_events(keyword="run marathon triathlon cycling", location=None, radius_km=50, page_size=50, page=1):
    # location can be {"lat":.., "lon":..} or city string
    params = {
        "q": keyword,
        "page": page,
        "expand": "venue",
        "sort_by": "date",
    }
    if isinstance(location, dict) and 'lat' in location and 'lon' in location:
        params['location.latitude'] = str(location['lat'])
        params['location.longitude'] = str(location['lon'])
        params['location.within'] = f"{radius_km}km"
    elif isinstance(location, str):
        params['location.address'] = location

    url = BASE + "events/search/"
    r = requests.get(url, headers=DEFAULT_HEADERS, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    return data

def normalize_eventbrite(item):
    # Simplify output into a minimal schema
    ev = {}
    ev['id'] = item.get('id')
    ev['name'] = item.get('name',{}).get('text')
    ev['description'] = item.get('description',{}).get('text')
    ev['start'] = item.get('start',{}).get('local')
    ev['end'] = item.get('end',{}).get('local')
    ev['url'] = item.get('url')
    ev['category'] = item.get('category_id')
    # venue
    venue = item.get('venue')
    if venue:
        address = venue.get('address',{})
        ev['venue_name'] = venue.get('name')
        ev['city'] = address.get('city')
        ev['lat'] = address.get('latitude')
        ev['lon'] = address.get('longitude')
    return ev

def fetch_and_normalize(location=None, radius_km=50):
    out = []
    page = 1
    while True:
        data = search_events(location=location, radius_km=radius_km, page=page)
        events = data.get('events', [])
        for e in events:
            out.append(normalize_eventbrite(e))
        pagination = data.get('pagination', {})
        if not pagination.get('has_more_items'):
            break
        page += 1
    return out
