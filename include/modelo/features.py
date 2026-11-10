"""Carga, alcance, objetivo y features derivadas.

Es la misma lógica que la sección 1 de presentacion3.ipynb (antes estaba escrita en la celda).
La usan el notebook, build_app.py y la app («Probá el modelo»), así que hay una sola versión.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
UMBRAL_MONTO = 10_000  # USD: montos menores son errores de carga de TMDb
ANIO_CORTE = 2026      # las de 2026 siguen en cartelera: revenue incompleto
COLUMNAS_FUGA = ["revenue", "vote_average", "vote_count"]

RAIZ = Path(__file__).resolve().parents[2]


def ruta_csv(raiz: Path | str = RAIZ) -> Path:
    """CSV más reciente de include/output (movies_<fecha>.csv) o, si no hay, el congelado.

    El nombre lleva la fecha ISO, así que el mayor alfabético es el último.
    """
    raiz = Path(raiz)
    corridas = sorted((raiz / "include" / "output").glob("movies_*.csv"))
    if corridas:
        return corridas[-1]
    return raiz / "include" / "frozen" / "ultimo_ok.csv"


def cargar_dataset(path: Path | str, minimo_filas: int = 7_000) -> pd.DataFrame:
    df = pd.read_csv(path)
    if len(df) < minimo_filas:
        raise ValueError(f"{path} tiene {len(df):,} filas: parece una corrida en modo subset. "
                         "Correr el DAG con mode=full antes de seguir.")
    return df


# Cada paso: (nombre, condición, por qué). El orden importa: es el embudo que se muestra en la app.
PASOS_ALCANCE = [
    ("Con budget y revenue > 0",
     lambda d: (d["budget"] > 0) & (d["revenue"] > 0),
     "Sin los dos montos no hay ROI. No se imputa: inventar un presupuesto es inventar el objetivo."),
    ("Con budget >= 10.000 USD",
     lambda d: d["budget"] >= UMBRAL_MONTO,
     "Presupuestos de 1, 2 o 7 dólares son errores de carga de TMDb."),
    ("Con revenue >= 10.000 USD",
     lambda d: d["revenue"] >= UMBRAL_MONTO,
     "Recaudaciones imposibles (estrenos de streaming sin taquilla o errores de carga)."),
    ("Estrenadas antes de 2026",
     lambda d: d["release_year"] < ANIO_CORTE,
     "Las de 2026 siguen en cartelera: su revenue está incompleto."),
]


def filtrar_alcance(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Aplica los cuatro filtros en orden. Devuelve los datos y el embudo (filas en cada paso)."""
    pasos = [{"paso": "Dataset completo", "filas": len(df),
              "motivo": "Todas las películas que bajó el DAG de TMDb."}]
    d = df
    for nombre, condicion, motivo in PASOS_ALCANCE:
        d = d[condicion(d)]
        pasos.append({"paso": nombre, "filas": len(d), "motivo": motivo})
    return d.copy(), pasos


def agregar_objetivo(data: pd.DataFrame) -> pd.DataFrame:
    """ROI y objetivo y = log10(revenue / budget) = log10(1 + ROI). Ordena cronológicamente."""
    data = data.copy()
    data["roi"] = (data["revenue"] - data["budget"]) / data["budget"]
    data["y"] = np.log10(data["revenue"] / data["budget"])
    # Orden cronológico estricto: lo usan la partición temporal y el encoding histórico
    return data.sort_values(["release_year", "release_month", "movie_id"]).reset_index(drop=True)


# Palabras clave de TMDb que se conocen antes del estreno (de qué trata y de dónde sale la historia).
# No se usan palabras que dependan del resultado ni la cantidad de palabras (crece con la popularidad).
GRUPOS_KEYWORDS = {
    "kw_secuela": ["sequel"],
    "kw_remake": ["remake"],
    "kw_libro": ["based on novel or book", "based on young adult novel"],
    "kw_comic": ["based on comic", "based on manga", "superhero"],
    "kw_historia_real": ["based on true story", "biography"],
    "kw_videojuego": ["based on video game"],
    "kw_navidad": ["christmas"],
    "kw_musical": ["musical"],
    "kw_found_footage": ["found footage"],
}

# Historias: cómo les fue antes al director, al actor principal y a la saga (mediana de y en años anteriores)
CLAVES_HISTORIA = {"director": "director_id", "actor": "lead_actor_id", "saga": "collection_id"}


@dataclass
class Referencias:
    """Lo que las features derivadas necesitan saber del resto de las películas.

    - mediana_budget_genero: mediana de budget por primary_genre sobre todo el alcance
      (no usa el objetivo; es la misma cuenta que hacía el notebook con groupby().transform()).
    - historia: (release_year, primary_production_company, y) de todo el alcance, para el
      target encoding histórico de la productora.
    """
    mediana_budget_genero: dict
    mediana_budget_global: float
    historia: pd.DataFrame
    mediana_y_global: float
    _cache: dict = field(default_factory=dict, repr=False)
    _cache_ctx: dict = field(default_factory=dict, repr=False)

    def previas_por(self, anio: int, clave: str) -> tuple[dict, dict]:
        """Mediana de y y cantidad de películas por director, actor o saga en años estrictamente < anio."""
        anio = int(anio)
        if (anio, clave) not in self._cache_ctx:
            past = self.historia[(self.historia["release_year"] < anio) & self.historia[clave].notna()]
            g = past.groupby(clave)["y"]
            self._cache_ctx[(anio, clave)] = (g.median().to_dict(), g.size().to_dict())
        return self._cache_ctx[(anio, clave)]

    def medianas_previas(self, anio: int) -> tuple[dict, float]:
        """Mediana de y por productora y global con las películas de años estrictamente < anio.

        El y de las películas del mismo año no entra: una película nunca ve su propio resultado
        ni el de sus contemporáneas.
        """
        anio = int(anio)
        if anio not in self._cache:
            past = self.historia[self.historia["release_year"] < anio]
            if len(past) > 0:
                self._cache[anio] = (past.groupby("primary_production_company")["y"].median().to_dict(),
                                     float(past["y"].median()))
            else:
                self._cache[anio] = ({}, self.mediana_y_global)
        return self._cache[anio]


def construir_referencias(data: pd.DataFrame) -> Referencias:
    return Referencias(
        mediana_budget_genero=data.groupby("primary_genre")["budget"].median().to_dict(),
        mediana_budget_global=float(data["budget"].median()),
        historia=data[["release_year", "primary_production_company", "y"]
                      + list(CLAVES_HISTORIA.values())].reset_index(drop=True),
        mediana_y_global=float(data["y"].median()),
    )


def agregar_features(data: pd.DataFrame, ref: Referencias) -> pd.DataFrame:
    """Features derivadas. Sirve igual para el alcance entero y para un caso nuevo de la app."""
    data = data.copy()
    log_b = np.log10(data["budget"])
    star_sum = data["lead_actor_popularity"] + data["co_star_popularity"]

    # Ratios de eficiencia de capital y sinergias
    data["star_power_ratio"] = star_sum / log_b
    data["director_leverage"] = data["director_popularity"] / log_b
    data["director_vs_cast"] = (data["director_popularity"] + 1.0) / (data["lead_actor_popularity"] + 1.0)
    data["power_pack"] = data["director_popularity"] * data["lead_actor_popularity"]

    # Interacción género x escala de presupuesto
    mediana_genero = data["primary_genre"].map(ref.mediana_budget_genero).fillna(ref.mediana_budget_global)
    data["budget_vs_genre_median"] = data["budget"] / mediana_genero
    data["is_horror_low_budget"] = ((data["primary_genre"] == "Horror") & (data["budget"] < 5_000_000)).astype(int)

    # Target encoding histórico de la productora: mediana de y en años < año de estreno.
    # Si la productora no tiene historia, la mediana global de esos años.
    def encoding(fila):
        por_prod, global_ = ref.medianas_previas(fila["release_year"])
        return por_prod.get(fila["primary_production_company"], global_)

    data["prod_hist_roi"] = data[["release_year", "primary_production_company"]].apply(encoding, axis=1)

    # Contexto (Entrega 4): historia del director, del actor principal y de la saga.
    # Misma regla que la productora: solo películas de años anteriores; sin historia queda vacío.
    for nombre, clave in CLAVES_HISTORIA.items():
        hist = pd.Series(np.nan, index=data.index)
        cant = pd.Series(0.0, index=data.index)
        for anio, idx in data.groupby("release_year").groups.items():
            medianas, conteos = ref.previas_por(anio, clave)
            k = data.loc[idx, clave]
            hist.loc[idx] = k.map(medianas).astype(float).values
            cant.loc[idx] = k.map(conteos).fillna(0).astype(float).values
        data[f"{nombre}_hist"] = hist
        data[f"{nombre}_n_prev"] = cant

    # Secuela: no es la primera película de su saga. («Pertenece a una saga» no se usa: TMDb crea la saga
    # cuando hay secuela, y eso pasa cuando a la primera le fue bien. Se conoce después del estreno.)
    data["es_secuela"] = (pd.to_numeric(data["saga_position"], errors="coerce").fillna(0) > 0).astype(int)

    palabras = data["keywords"].fillna("").astype(str).str.split("|").apply(lambda l: {k.strip() for k in l})
    for col, lista in GRUPOS_KEYWORDS.items():
        data[col] = palabras.apply(lambda st, lista=lista: int(any(k in st for k in lista)))

    paises = data["production_countries"].fillna("").astype(str).str.split("|").apply(lambda l: [p for p in l if p])
    data["paises_n"] = paises.apply(len)
    data["es_eeuu"] = paises.apply(lambda l: int("US" in l))
    return data


def preparar_datos(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], Referencias]:
    """Alcance + objetivo + features, en el orden del notebook."""
    data, pasos = filtrar_alcance(df)
    data = agregar_objetivo(data)
    ref = construir_referencias(data)
    data = agregar_features(data, ref)
    return data, pasos, ref
