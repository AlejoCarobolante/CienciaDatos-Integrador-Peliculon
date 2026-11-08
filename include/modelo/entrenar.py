"""Columnas, preprocesador, partición temporal, candidatos y comparaciones.

Es la misma lógica de las secciones 4 a 6 de presentacion3.ipynb. El notebook y build_app.py
la importan de acá, así que el modelo que se guarda para la app es el mismo que se mide en el notebook.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from include.modelo.features import SEED

COLS_LOG = ["budget", "director_popularity", "lead_actor_popularity", "co_star_popularity", "power_pack"]
COLS_NUM = ["runtime", "release_year", "star_power_ratio", "director_leverage", "director_vs_cast",
            "budget_vs_genre_median", "prod_hist_roi"]
COLS_BOOL = ["is_horror_low_budget"]
COLS_CAT = ["original_language", "primary_genre", "secondary_genre", "release_month", "release_day_of_week"]
COLS_PROD = ["primary_production_company", "co_production_company"]
FEATURES = COLS_LOG + COLS_NUM + COLS_BOOL + COLS_CAT + COLS_PROD

# Columnas que carga el usuario (el resto se deriva con features.agregar_features)
COLS_BASE = ["budget", "runtime", "release_year", "release_month", "release_day_of_week",
             "original_language", "primary_genre", "secondary_genre",
             "primary_production_company", "co_production_company",
             "director_popularity", "lead_actor_popularity", "co_star_popularity"]

FRAC_TEST = 0.20
N_FOLDS = 5
NOMBRE_FINAL = "Random Forest ajustado"
GRILLA_RF = {"modelo__min_samples_leaf": [1, 5, 10, 20, 40],
             "modelo__max_features": [0.3, 0.6, 1.0]}


def log10_1p(X):
    return np.log10(1 + np.maximum(np.asarray(X, dtype=float), 0))


def cero_a_nan(X):
    X = np.asarray(X, dtype=float).copy()
    X[X <= 0] = np.nan
    return X


def armar_preprocesador(excluir=()):
    quedan = lambda cols: [c for c in cols if c not in excluir]
    bloques = [
        ("log", make_pipeline(FunctionTransformer(log10_1p, feature_names_out="one-to-one"),
                              StandardScaler()), quedan(COLS_LOG)),
        ("num", make_pipeline(FunctionTransformer(cero_a_nan, feature_names_out="one-to-one"),
                              SimpleImputer(strategy="median"), StandardScaler()), quedan(COLS_NUM)),
        ("bool", "passthrough", quedan(COLS_BOOL)),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                              sparse_output=False), quedan(COLS_CAT)),
        ("prod", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30,
                               sparse_output=False), quedan(COLS_PROD)),
    ]
    return ColumnTransformer([b for b in bloques if b[2]])


def modelo(estimador, excluir=()):
    return Pipeline([("pre", armar_preprocesador(excluir)), ("modelo", estimador)])


def particion_temporal(data: pd.DataFrame, frac_test: float = FRAC_TEST):
    """El 20% más reciente es el test (se mira una vez); el 80% anterior, entrenamiento."""
    datos_ord = data.sort_values(["release_year", "release_month", "movie_id"]).reset_index(drop=True)
    n_test = int(len(datos_ord) * frac_test)
    return datos_ord.iloc[:-n_test], datos_ord.iloc[-n_test:]


def cv_temporal():
    return TimeSeriesSplit(n_splits=N_FOLDS)


def candidatos():
    """Los cuatro niveles de la sección 5, con sus valores por defecto razonables."""
    return {
        "Piso (mediana)": DummyRegressor(strategy="median"),
        "Ridge": modelo(Ridge(alpha=10.0)),
        "Random Forest": modelo(RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                                      max_features=0.3, n_jobs=-1, random_state=SEED)),
        # loss="huber" no existe en HistGradientBoostingRegressor (y quantile solo aplica con
        # loss="quantile"): se usa absolute_error, robusta a la cola y alineada con el MAE.
        "HistGB (absolute_error)": modelo(HistGradientBoostingRegressor(loss="absolute_error",
                                                                       learning_rate=0.04, max_iter=300,
                                                                       min_samples_leaf=15,
                                                                       l2_regularization=2.0,
                                                                       random_state=SEED)),
    }


def grilla_rf(cv):
    return GridSearchCV(modelo(RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED)),
                        param_grid=GRILLA_RF, cv=cv, scoring="neg_mean_absolute_error")


def params_rf(best_params: dict) -> dict:
    """Hiperparámetros del Random Forest ajustado a partir de grilla.best_params_."""
    return dict(n_estimators=300, n_jobs=-1, random_state=SEED,
                min_samples_leaf=best_params["modelo__min_samples_leaf"],
                max_features=best_params["modelo__max_features"])


def rf_ajustado(best_params: dict, excluir=()):
    return modelo(RandomForestRegressor(**params_rf(best_params)), excluir=excluir)


def evaluar_mae_folds(modelos: dict, X, y, cv) -> pd.DataFrame:
    """MAE de cada modelo en cada fold (mismos folds para todos: permite comparar de a pares)."""
    por_fold = {}
    for nombre, m in modelos.items():
        r = cross_validate(m, X, y, cv=cv, scoring="neg_mean_absolute_error")
        por_fold[nombre] = -r["test_score"]
    return pd.DataFrame(por_fold, index=[f"fold {i + 1}" for i in range(cv.get_n_splits())])


METRICAS = {"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error", "R2": "r2"}


def evaluar_multimetrica(modelos: dict, X, y, cv) -> pd.DataFrame:
    resultados = []
    for nombre, m in modelos.items():
        r = cross_validate(m, X, y, cv=cv, scoring=METRICAS)
        mae_m = -r["test_MAE"].mean()
        resultados.append({
            "Modelo": nombre,
            "MAE medio": mae_m,
            "factor de error": 10 ** mae_m,
            "RMSE medio": -r["test_RMSE"].mean(),
            "R² medio": r["test_R2"].mean(),
            "R² desvío": r["test_R2"].std(ddof=0),
        })
    return pd.DataFrame(resultados).set_index("Modelo")


def comparar(folds: pd.DataFrame, a: str, b: str) -> dict:
    """Diferencia de MAE fold por fold. Positivo = b se equivoca menos que a."""
    dif = folds[a] - folds[b]
    return {"comparación": f"{b} vs {a}", "mejora media": dif.mean(), "desvío de la mejora": dif.std(ddof=0),
            "folds que gana": f"{(dif > 0).sum()} de {len(dif)}", "supera el ruido": dif.mean() > dif.std(ddof=0)}


TRAMOS_ROI = ([-1.01, 0, 1, 3, 10, np.inf],
              ["Pérdida (ROI<0)", "Recupera hasta 2x", "2x a 4x", "4x a 11x", "Más de 11x"])


def tramo_roi(roi: pd.Series) -> pd.Series:
    return pd.cut(roi, TRAMOS_ROI[0], labels=TRAMOS_ROI[1])
