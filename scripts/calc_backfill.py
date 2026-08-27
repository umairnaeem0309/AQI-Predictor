import time, requests, os, sys
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv(".env")
ow_key = os.environ.get("OPENWEATHER_API_KEY")
now = int(time.time())
cities = {"karachi": (24.8607, 67.0011), "lahore": (31.5204, 74.3587), "islamabad": (33.6844, 73.0479)}
url = "https://api.openweathermap.org/data/2.5/air_pollution/history"
print("HISTORICAL BACKFILL POTENTIAL (OpenWeather Air Pollution)")
print("=" * 60)
for days in [21, 30, 90, 365]:
    start = now - (days * 24 * 3600)
    total = 0
    for city, (lat, lon) in cities.items():
        params = {"lat": lat, "lon": lon, "start": start, "end": now, "appid": ow_key}
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            count = len(data.get("list", []))
            total += count
    print(f"{days:>4} days: {total:>5} total obs ({total//3} per city)")
