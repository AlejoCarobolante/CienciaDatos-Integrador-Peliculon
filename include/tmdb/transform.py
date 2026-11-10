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


def secondary_genre(genres: list[dict] | None) -> str:
    """Toma el segundo género de la lista si existe."""
    if genres and isinstance(genres, list) and len(genres) > 1:
        return str(genres[1].get("name", "None"))
    return "None"


def primary_company(companies: list[dict] | None) -> str:
    """Toma la primera productora del listado."""
    if companies and isinstance(companies, list) and len(companies) > 0:
        return str(companies[0].get("name", "Unknown"))
    return "Unknown"


def co_production_company(companies: list[dict] | None) -> str:
    """Toma la segunda productora si existe."""
    if companies and isinstance(companies, list) and len(companies) > 1:
        return str(companies[1].get("name", "None"))
    return "None"


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


def co_star_popularity(cast: list[dict] | None) -> float:
    """Extrae la popularidad del segundo actor coprotagónico (order == 1)."""
    if not cast or not isinstance(cast, list) or len(cast) < 2:
        return 0.0
    try:
        return float(cast[1].get("popularity", 0.0))
    except (ValueError, TypeError):
        return 0.0


def director_id(crew: list[dict] | None) -> int | None:
    """ID de TMDb del director (sirve para calcular cómo le fue en películas anteriores)."""
    for member in crew or []:
        if member.get("job") == "Director" and member.get("id") is not None:
            return int(member["id"])
    return None


def lead_actor_id(cast: list[dict] | None) -> int | None:
    """ID de TMDb del actor principal (order == 0)."""
    if cast and isinstance(cast, list) and cast[0].get("id") is not None:
        return int(cast[0]["id"])
    return None


def director_name(crew: list[dict] | None) -> str:
    for member in crew or []:
        if member.get("job") == "Director":
            return str(member.get("name") or "")
    return ""


def lead_actor_name(cast: list[dict] | None) -> str:
    if cast and isinstance(cast, list):
        return str(cast[0].get("name") or "")
    return ""


def collection_id(collection: dict | None) -> int | None:
    """ID de la saga a la que pertenece la película, si pertenece a una."""
    if collection and collection.get("id") is not None:
        return int(collection["id"])
    return None


def production_countries(countries: list[dict] | None) -> str:
    """Códigos ISO de los países de producción, separados por '|'."""
    return "|".join(c["iso_3166_1"] for c in countries or [] if c.get("iso_3166_1"))


def keywords(payload: dict | None) -> str:
    """Palabras clave de TMDb en minúsculas, separadas por '|'."""
    lista = (payload or {}).get("keywords") or []
    return "|".join(str(k.get("name", "")).strip().lower() for k in lista if k.get("name"))


def saga_positions(parts: list[dict] | None) -> dict[int, int]:
    """Posición de cada película dentro de su saga según la fecha de estreno (0 = la primera)."""
    con_fecha = [p for p in parts or [] if p.get("release_date") and p.get("id") is not None]
    con_fecha.sort(key=lambda p: p["release_date"])
    return {int(p["id"]): i for i, p in enumerate(con_fecha)}


def contexto(raw: dict[str, Any]) -> dict[str, Any]:
    """Columnas del bloque CONTEXTO (saga_position se completa después, con la saga entera)."""
    credits = raw.get("credits") or {}
    return {
        "director_id": director_id(credits.get("crew")),
        "director_name": director_name(credits.get("crew")),
        "lead_actor_id": lead_actor_id(credits.get("cast")),
        "lead_actor_name": lead_actor_name(credits.get("cast")),
        "collection_id": collection_id(raw.get("belongs_to_collection")),
        "collection_name": str((raw.get("belongs_to_collection") or {}).get("name") or ""),
        "saga_position": None,
        "production_countries": production_countries(raw.get("production_countries")),
        "keywords": keywords(raw.get("keywords")),
    }


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
        r["revenue"] = float(raw.get("revenue", 0.0))
    except (ValueError, TypeError):
        r["revenue"] = 0.0

    try:
        r["runtime"] = int(raw.get("runtime") or 0)
    except (ValueError, TypeError):
        r["runtime"] = 0

    r["original_language"] = raw.get("original_language")
    r["primary_genre"] = primary_genre(raw.get("genres"))
    r["secondary_genre"] = secondary_genre(raw.get("genres"))
    r["primary_production_company"] = primary_company(raw.get("production_companies"))
    r["co_production_company"] = co_production_company(raw.get("production_companies"))

    credits = raw.get("credits") or {}
    r["director_popularity"] = director_popularity(credits.get("crew"))
    r["lead_actor_popularity"] = lead_actor_popularity(credits.get("cast"))
    r["co_star_popularity"] = co_star_popularity(credits.get("cast"))

    try:
        r["vote_average"] = float(raw.get("vote_average", 0.0))
    except (ValueError, TypeError):
        r["vote_average"] = 0.0

    try:
        r["vote_count"] = int(raw.get("vote_count", 0))
    except (ValueError, TypeError):
        r["vote_count"] = 0

    r.update(contexto(raw))
    return r
