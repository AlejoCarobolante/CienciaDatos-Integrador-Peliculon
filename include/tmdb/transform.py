"""Del JSON crudo de TMDb al esquema canónico de la cátedra."""
from __future__ import annotations

import datetime
from typing import Any

from tmdb import schema


def parse_date(date_str: str | None) -> tuple[int | None, int | None, int | None]:
    """'2010-07-16' -> (2010, 7, 4) (año, mes, día de la semana)."""
    if not date_str:
        return None, None, None
    try:
        dt = datetime.date.fromisoformat(date_str)
        return dt.year, dt.month, dt.weekday()
    except (ValueError, TypeError):
        return None, None, None


def primary_genre(genres: list[dict] | None) -> str:
    """Toma el primer género de la lista (género dominante)."""
    if genres and isinstance(genres, list) and len(genres) > 0:
        return str(genres[0].get("name", "Unknown"))
    return "Unknown"


def primary_company(companies: list[dict] | None) -> str:
    """Toma la primera productora del listado."""
    if companies and isinstance(companies, list) and len(companies) > 0:
        return str(companies[0].get("name", "Unknown"))
    return "Unknown"


def director_popularity(crew: list[dict] | None) -> float:
    """Busca el director en el equipo técnico y extrae su popularidad."""
    if not crew or not isinstance(crew, list):
        return 0.0
    for member in crew:
        if member.get("job") == "Director":
            try:
                return float(member.get("popularity", 0.0))
            except (ValueError, TypeError):
                return 0.0
    return 0.0


def lead_actor_popularity(cast: list[dict] | None) -> float:
    """Extrae la popularidad del actor principal (order == 0)."""
    if not cast or not isinstance(cast, list) or len(cast) == 0:
        return 0.0
    try:
        return float(cast[0].get("popularity", 0.0))
    except (ValueError, TypeError):
        return 0.0


def to_row(raw: dict[str, Any]) -> dict[str, Any]:
    """Transforma el JSON de una película a una fila del esquema canónico."""
    r = {c: None for c in schema.COLUMNS}

    r["movie_id"] = raw.get("id")
    r["title"] = raw.get("title")

    year, month, dow = parse_date(raw.get("release_date"))
    r["release_year"] = year
    r["release_month"] = month
    r["release_day_of_week"] = dow

    try:
        r["budget"] = float(raw.get("budget", 0.0))
    except (ValueError, TypeError):
        r["budget"] = 0.0

    try:
        r["runtime"] = int(raw.get("runtime") or 0)
    except (ValueError, TypeError):
        r["runtime"] = 0

    r["original_language"] = raw.get("original_language")
    r["primary_genre"] = primary_genre(raw.get("genres"))
    r["primary_production_company"] = primary_company(raw.get("production_companies"))

    credits = raw.get("credits") or {}
    r["director_popularity"] = director_popularity(credits.get("crew"))
    r["lead_actor_popularity"] = lead_actor_popularity(credits.get("cast"))

    try:
        r["vote_average"] = float(raw.get("vote_average", 0.0))
    except (ValueError, TypeError):
        r["vote_average"] = 0.0

    try:
        r["vote_count"] = int(raw.get("vote_count", 0))
    except (ValueError, TypeError):
        r["vote_count"] = 0

    return r