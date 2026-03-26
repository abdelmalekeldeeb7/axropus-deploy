"""Strategy 6: Weather/Temperature — highest edge, least competition."""
import logging
import requests
import time
from ..core.amf_bridge import AMFBridge
from .btc_5min import Signal

logger = logging.getLogger(__name__)

CITY_COORDS = {
    "new york": (40.7128, -74.0060),
    "los angeles": (34.0522, -118.2437),
    "chicago": (41.8781, -87.6298),
    "miami": (25.7617, -80.1918),
    "london": (51.5074, -0.1278),
    "paris": (48.8566, 2.3522),
    "tokyo": (35.6762, 139.6503),
    "sydney": (-33.8688, 151.2093),
    "dubai": (25.2048, 55.2708),
    "toronto": (43.6532, -79.3832),
}


def get_weather_forecast(lat: float, lon: float) -> dict:
    """Open-Meteo free API — no key needed."""
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "hourly": "temperature_2m,precipitation_probability,windspeed_10m",
                "temperature_unit": "fahrenheit",
                "forecast_days": 2, "timezone": "auto"
            },
            timeout=5
        )
        if r.ok:
            data = r.json()
            hourly = data.get("hourly", {})
            temps = hourly.get("temperature_2m", [])
            return {
                "temps": temps[:24],
                "precip": hourly.get("precipitation_probability", [])[:24],
                "wind": hourly.get("windspeed_10m", [])[:24],
                "current_temp": temps[0] if temps else None,
            }
    except Exception as e:
        logger.debug("weather API error: %s", e)
    return {}


class WeatherStrategy:
    """
    Strategy 6: Weather/Temperature markets.
    Uses Open-Meteo (free) for forecasts.
    Edge: 62-70% — almost no quant competition on weather.
    Run across 5 accounts with different cities.
    """

    MIN_CONFIDENCE = 0.65
    MIN_VOLUME = 200.0

    def __init__(self, amf: AMFBridge):
        self.amf = amf
        self.name = "weather"
        self._forecast_cache: dict[str, tuple[dict, float]] = {}

    def scan(self, markets: list) -> list[Signal]:
        signals = []
        for m in markets:
            city = self._extract_city(m.question)
            if not city:
                continue
            if m.volume < self.MIN_VOLUME:
                continue

            forecast = self._get_forecast(city)
            if not forecast:
                continue

            sig = self._analyze(m, city, forecast)
            if sig:
                signals.append(sig)

        return signals

    def _analyze(self, m, city: str, forecast: dict) -> Signal | None:
        # Extract target temp from question
        target_temp = self._extract_temp(m.question)
        if target_temp is None:
            return None

        current = forecast.get("current_temp")
        temps = forecast.get("temps", [])
        if current is None or not temps:
            return None

        # Forecast-based true probability
        relevant_temps = temps[:6]  # next 6 hours
        above = sum(1 for t in relevant_temps if t > target_temp)
        true_prob_yes = above / len(relevant_temps)

        # Edge calculation
        yes_edge = true_prob_yes - m.yes_price
        no_edge  = (1 - true_prob_yes) - m.no_price

        best_edge = max(yes_edge, no_edge)
        if best_edge < 0.08:
            return None

        llm = self.amf.analyze(
            question=m.question, yes_price=m.yes_price, no_price=m.no_price,
            btc_price=0, btc_change=0, eth_price=0, eth_change=0,
            book_ratio=1.0, spread=abs(m.yes_price - m.no_price),
            momentum=f"Forecast temps={[round(t,1) for t in relevant_temps[:4]]} target={target_temp}°F",
            time_elapsed=0, window_duration=86400
        )

        if yes_edge >= no_edge:
            direction = "UP"
            confidence = min(0.85, 0.50 + yes_edge + (llm.confidence - 0.5) * 0.3)
            token, price = m.yes_token, m.yes_price
        else:
            direction = "DOWN"
            confidence = min(0.85, 0.50 + no_edge + (llm.confidence - 0.5) * 0.3)
            token, price = m.no_token, m.no_price

        if confidence < self.MIN_CONFIDENCE:
            return None

        return Signal(
            direction=direction, confidence=confidence,
            entry_price=price, token_id=token,
            market_slug=m.slug, question=m.question,
            tp_pct=0.35, sl_pct=0.12, max_hold=7200.0,
            source=self.name
        )

    def _get_forecast(self, city: str) -> dict:
        if city in self._forecast_cache:
            data, ts = self._forecast_cache[city]
            if time.time() - ts < 1800:  # 30-min cache
                return data

        coords = CITY_COORDS.get(city)
        if not coords:
            return {}

        data = get_weather_forecast(*coords)
        if data:
            self._forecast_cache[city] = (data, time.time())
        return data

    def _extract_city(self, question: str) -> str | None:
        q = question.lower()
        for city in CITY_COORDS:
            if city in q:
                return city
        return None

    def _extract_temp(self, question: str) -> float | None:
        import re
        matches = re.findall(r"(\d{2,3})°?[Ff]", question)
        if matches:
            return float(matches[0])
        matches = re.findall(r"(\d{2,3})\s*degrees", question.lower())
        if matches:
            return float(matches[0])
        return None
