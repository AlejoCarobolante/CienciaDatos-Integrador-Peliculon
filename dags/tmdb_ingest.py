"""
### Pipeline TMDb — Ciencia de Datos, UTN FRM 2026

Construye el dataset de películas para predecir éxito de crítica
a partir de la API oficial de The Movie Database (TMDb).

Capas del modelo medallón:
  * Bronce (land_bronze): Payloads JSON tal como los devolvió TMDb, comprimidos.
  * Plata (refine_silver + consolidate / load_frozen): Filas tipadas (14 columnas), deduplicadas y validadas.
"""
from __future__ import annotations

import gzip
import json
import logging
import shutil
from pathlib import Path

import pendulum
from airflow.sdk import Param, PokeReturnValue, dag, task
from airflow.utils.trigger_rule import TriggerRule

from tmdb import schema
from tmdb.client import fetch_movie_payload, fetch_popular_movie_ids
from tmdb.transform import to_row

log = logging.getLogger(__name__)

INCLUDE_DIR = Path("/usr/local/airflow/include")
FROZEN_DIR = INCLUDE_DIR / "frozen"
OUTPUT_DIR = INCLUDE_DIR / "output"
BRONZE_DIR = OUTPUT_DIR / "bronze"
SILVER_DIR = OUTPUT_DIR / "silver"
PARTIAL_DIR = SILVER_DIR / "_parciales"


def bronze_path(movie_id: int) -> Path:
    """Ruta del archivo crudo particionado por ID."""
    return BRONZE_DIR / f"movie_{movie_id:08d}.json.gz"


def bronze_write(destino: Path, data: dict) -> None:
    """Escribe el JSON crudo comprimido en gzip."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(destino, "wt", encoding="utf-8") as f:
        json.dump(data, f)


def bronze_read(ruta: Path) -> dict:
    """Lee el JSON crudo comprimido desde disco."""
    with gzip.open(ruta, "rt", encoding="utf-8") as f:
        return json.load(f)


@dag(
    dag_id="tmdb_ingest",
    schedule="@daily",
    start_date=pendulum.datetime(2026, 8, 1, tz="America/Argentina/Buenos_Aires"),
    catchup=False,
    max_active_tasks=8,
    tags=["ciencia-de-datos", "proyecto-integrador", "tmdb"],
    doc_md=__doc__,
    params={
        "mode": Param(
            "subset",
            enum=["subset", "full"],
            title="Modo de corrida",
            description="subset: 5 páginas (~100 películas). full: 175 páginas (~3.500 películas).",
        ),
        "force": Param(
            False,
            type="boolean",
            title="Forzar la corrida",
            description="Vuelve a consultar la API aunque el archivo ya exista en bronce.",
        ),
    },
)
def tmdb_ingest():

    @task.branch
    def check_source_and_branch() -> str:
        """Verifica disponibilidad de la API y bifurca a la rama viva o frozen."""
        try:
            ids = fetch_popular_movie_ids(page=1, min_votes=100)
            if ids and len(ids) > 0:
                log.info("API TMDb respondiendo correctamente (%s IDs). Rama viva.", len(ids))
                return "discover_batches"
        except Exception as e:
            log.warning("Fallo al consultar la API de TMDb: %s", e)

        log.warning("Degradando a rama de respaldo (load_frozen).")
        return "load_frozen"

    @task
    def load_frozen() -> str:
        """Rama de respaldo: Carga snapshot existente ante falla de red."""
        candidatos = [
            FROZEN_DIR / "ultimo_ok.csv",
            FROZEN_DIR / "tmdb_snapshot.csv",
        ]
        origen = None
        for c in candidatos:
            if c.exists():
                origen = c
                break

        if not origen:
            raise FileNotFoundError(
                "No se encontró ningún archivo de respaldo en include/frozen/ "
                "(se buscó 'ultimo_ok.csv' y 'tmdb_snapshot.csv')."
            )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        destino = OUTPUT_DIR / "_consolidado.csv"
        shutil.copyfile(origen, destino)
        log.warning("Usando dataset de respaldo desde -> %s", origen)
        return str(destino)

    @task
    def discover_batches(**context) -> list[int]:
        """Define la lista de páginas a procesar según el modo."""
        params = context["params"]
        pages_count = 5 if params["mode"] == "subset" else 175
        pages = list(range(1, pages_count + 1))
        log.info("Modo %s: generando %s tareas de lote.", params["mode"], len(pages))
        return pages

    @task(retries=2, retry_delay=pendulum.duration(seconds=10))
    def land_bronze(page: int, **context) -> dict:
        """Capa Bronce: Persiste cada payload JSON crudo comprimido."""
        params = context["params"]
        force = params["force"]
        movie_ids = fetch_popular_movie_ids(page=page, min_votes=100)
        archivos = []

        for m_id in movie_ids:
            destino = bronze_path(m_id)
            if destino.exists() and not force:
                archivos.append(str(destino))
                continue

            try:
                payload = fetch_movie_payload(m_id)
                bronze_write(destino, payload)
                archivos.append(str(destino))
            except Exception as e:
                log.error("Error al obtener película ID %s: %s", m_id, e)

        log.info("Página %s: %s películas procesadas en bronce.", page, len(archivos))
        return {"page": page, "files": archivos}

    @task
    def refine_silver(lote: dict) -> str:
        """Capa Plata: Convierte de Bronce a formato tabular parcial."""
        import pandas as pd

        filas = []
        for ruta_str in lote["files"]:
            payload = bronze_read(Path(ruta_str))
            fila = to_row(payload)
            if fila.get("movie_id") is not None:
                filas.append(fila)

        PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
        destino = PARTIAL_DIR / f"lote_pagina_{lote['page']:04d}.csv"
        pd.DataFrame(filas, columns=schema.COLUMNS).to_csv(destino, index=False)
        log.info("Parcial generado -> %s (%s registros)", destino.name, len(filas))
        return str(destino)

    @task
    def consolidate(rutas: list[str]) -> str:
        """Consolida parciales, deduplica y fuerza tipos de datos."""
        import pandas as pd

        partes = [pd.read_csv(r, low_memory=False) for r in rutas if r]
        partes = [p for p in partes if len(p)]
        if not partes:
            raise ValueError("Ningún lote devolvió filas.")

        df = pd.concat(partes, ignore_index=True)[schema.COLUMNS]
        antes = len(df)
        df = df.drop_duplicates(subset=["movie_id"]).reset_index(drop=True)
        log.info("%s filas consolidadas, %s tras deduplicar.", antes, len(df))

        for c in schema.ENTEROS:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        for c in schema.FLOTANTES:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
        for c in schema.TEXTO:
            df[c] = df[c].fillna("Unknown").astype(str)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        destino = OUTPUT_DIR / "_consolidado.csv"
        df.to_csv(destino, index=False)
        return str(destino)

    @task(trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
    def validate(**context) -> str:
        """Chequeos de integridad y reglas Tidy sobre _consolidado.csv."""
        import pandas as pd

        # Ambas ramas dejan el archivo en _consolidado.csv
        ruta_archivo = OUTPUT_DIR / "_consolidado.csv"
        if not ruta_archivo.exists():
            raise FileNotFoundError(f"No existe el archivo a validar en {ruta_archivo}")

        df = pd.read_csv(ruta_archivo, low_memory=False)
        problemas = []
        params = context["params"]
        modo = params.get("mode", "subset")

        min_filas_esperadas = 3000 if modo == "full" else 80
        if len(df) < min_filas_esperadas:
            problemas.append(
                f"Volumen de datos insuficiente para modo '{modo}': "
                f"obtenidas {len(df)} filas, mínimo requerido {min_filas_esperadas}."
            )

        if list(df.columns) != schema.COLUMNS:
            faltan = set(schema.COLUMNS) - set(df.columns)
            sobran = set(df.columns) - set(schema.COLUMNS)
            problemas.append(f"Columnas no coinciden. Faltan: {faltan}, Sobran: {sobran}")

        if df["movie_id"].duplicated().any():
            problemas.append(f"{df['movie_id'].duplicated().sum()} movie_id duplicados.")

        for c in schema.OBLIGATORIAS:
            if c in df.columns and df[c].isna().any():
                problemas.append(f"Columna obligatoria '{c}' contiene {df[c].isna().sum()} nulos.")

        if "vote_average" in df.columns and not df["vote_average"].between(0.0, 10.0).all():
            problemas.append("vote_average contiene valores fuera del rango 0.0 - 10.0.")

        if problemas:
            raise ValueError("Fallo en validación de esquema Tidy:\n - " + "\n - ".join(problemas))

        log.info("Validación exitosa: %s películas conformes con el esquema.", len(df))
        return str(ruta_archivo)

    @task(trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
    def save(ruta: str, **context) -> str:
        """Publica el entregable final con fecha y refresca ultimo_ok.csv."""
        dag_run = context["dag_run"]
        momento = dag_run.logical_date or dag_run.run_after
        ds = momento.date().isoformat()

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        FROZEN_DIR.mkdir(parents=True, exist_ok=True)

        destino_entregable = OUTPUT_DIR / f"movies_{ds}.csv"
        destino_cache = FROZEN_DIR / "ultimo_ok.csv"

        # copyfile copia el contenido sin tocar permisos/metadatos del OS
        shutil.copyfile(ruta, destino_entregable)
        shutil.copyfile(ruta, destino_cache)
        
        log.info("Entregable generado en -> %s (Caché local actualizada)", destino_entregable)
        return str(destino_entregable)

    # --- Grafo ---
    branch = check_source_and_branch()

    # Rama viva
    batches = discover_batches()
    bronces = land_bronze.expand(page=batches)
    parciales = refine_silver.expand(lote=bronces)
    consolidado = consolidate(parciales)

    # Rama respaldo
    respaldo = load_frozen()

    # Validación y Guardado
    validado = validate()
    guardado = save(validado)

    # Conexiones
    branch >> [batches, respaldo]
    [consolidado, respaldo] >> validado >> guardado


tmdb_ingest()