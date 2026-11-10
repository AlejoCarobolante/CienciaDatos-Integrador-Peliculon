"""Agrega las columnas de CONTEXTO a un CSV ya generado por el DAG, sin cambiar las demás.

Sirve para sumar los datos nuevos a la corrida del 24/10 sin volver a muestrear películas
(una corrida nueva del DAG traería otra muestra, porque la popularidad de TMDb cambia todos los días).
Usa el mismo cliente y las mismas transformaciones que el DAG.

    cd include
    TMDB_API_KEY=... BASE_URL=https://api.themoviedb.org/3 python -m tmdb.enriquecer frozen/ultimo_ok.csv
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from tmdb import schema
from tmdb.client import fetch_collection, fetch_movie_payload
from tmdb.transform import contexto, saga_positions


def _contexto_de(movie_id: int) -> dict:
    for _ in range(4):
        try:
            return {"movie_id": movie_id, **contexto(fetch_movie_payload(movie_id))}
        except Exception:
            continue
    return {"movie_id": movie_id}


def enriquecer(ruta: str, hilos: int = 8) -> pd.DataFrame:
    df = pd.read_csv(ruta, low_memory=False)
    base = [c for c in df.columns if c not in schema.CONTEXTO]
    with ThreadPoolExecutor(hilos) as ex:
        ctx = pd.DataFrame(list(ex.map(_contexto_de, df["movie_id"].astype(int))))

    posiciones: dict[int, int] = {}
    with ThreadPoolExecutor(hilos) as ex:
        for partes in ex.map(lambda cid: fetch_collection(int(cid)).get("parts"),
                             ctx["collection_id"].dropna().unique()):
            posiciones.update(saga_positions(partes))
    ctx["saga_position"] = ctx["movie_id"].map(posiciones)

    df = df[base].merge(ctx, on="movie_id", how="left")
    for c in schema.ENTEROS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in schema.TEXTO_LIBRE:
        df[c] = df[c].fillna("").astype(str)
    return df[schema.COLUMNS]


if __name__ == "__main__":
    ruta = sys.argv[1]
    df = enriquecer(ruta)
    df.to_csv(ruta, index=False)
    print(f"{len(df)} películas | con director: {df['director_id'].notna().sum()} | "
          f"en una saga: {df['collection_id'].notna().sum()} | secuelas: {(df['saga_position'] > 0).sum()}")
