"""Cliente HTTP y llamadas a la API de TMDb."""
from __future__ import annotations

import json
import os
import time
from typing import Any
import requests

TIMEOUT_SECONDS = 10
BASE_URL = os.getenv("BASE_URL")


def get_api_key() -> str:
    """Obtiene la API key desde las variables de entorno."""
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        raise ValueError(
            "No se encontró TMDB_API_KEY en las variables de entorno. "
            "Definila en el archivo .env de tu proyecto."
        )
    return api_key


def fetch_popular_movie_ids(page: int, min_votes: int = 100) -> list[int]:
    """Obtiene los IDs de las películas de una página del catálogo."""
    api_key = get_api_key()
    url = f"{BASE_URL}/discover/movie"
    params = {
        "api_key": api_key,
        "page": page,
        "vote_count.gte": min_votes,
        "sort_by": "vote_count.desc",
        "include_adult": "false",
        "include_video": "false",
    }
    
    response = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    data = response.json()
    
    results = data.get("results") or []
    return [int(m["id"]) for m in results if m.get("id") is not None]


def fetch_movie_payload(movie_id: int) -> dict[str, Any]:
    """Obtiene el payload completo de una película con sus créditos."""
    api_key = get_api_key()
    url = f"{BASE_URL}/movie/{movie_id}"
    params = {
        "api_key": api_key,
        "append_to_response": "credits",
    }
    
    response = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    time.sleep(0.05)  # Respeto básico de rate limit
    return response.json()