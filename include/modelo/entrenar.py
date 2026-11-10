"""Columnas, preprocesador, partición temporal, candidatos y comparaciones.

Es la misma lógica de las secciones 4 a 6 de presentacion3.ipynb. El notebook y build_app.py
la importan de acá, así que el modelo que se guarda para la app es el mismo que se mide en el notebook.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, VotingRegressor
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
# Contexto agregado en la Entrega 4 (director, actor, saga, palabras clave y países)
COLS_CTX = (["director_hist", "director_n_prev", "actor_hist", "actor_n_prev", "saga_hist", "es_secuela"]
            + ["kw_secuela", "kw_remake", "kw_libro", "kw_comic", "kw_historia_real", "kw_videojuego",
               "kw_navidad", "kw_musical", "kw_found_footage"]
            + ["paises_n", "es_eeuu"])
FEATURES = COLS_LOG + COLS_NUM + COLS_BOOL + COLS_CAT + COLS_PROD + COLS_CTX

# Columnas que carga el usuario (el resto se deriva con features.agregar_features)
COLS_BASE = ["budget", "runtime", "release_year", "release_month", "release_day_of_week",
             "original_language", "primary_genre", "secondary_genre",
             "primary_production_company", "co_production_company",
             "director_popularity", "lead_actor_popularity", "co_star_popularity",
             "director_id", "lead_actor_id", "collection_id", "saga_position", "production_countries", "keywords"]

FRAC_TEST = 0.20
N_FOLDS = 5
NOMBRE_RF = "Random Forest ajustado"
NOMBRE_FINAL = "Promedio de 3 modelos"
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
        # Historias vacías (sin películas anteriores) -> mediana + una columna que marca que faltaba
        ("ctx", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()),
         quedan(COLS_CTX)),
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


HISTGB = dict(loss="absolute_error", learning_rate=0.04, max_iter=300, min_samples_leaf=15,
              l2_regularization=2.0, random_state=SEED)


def modelo_final(best_params: dict, excluir=()):
    """Modelo final: promedio de las predicciones del Random Forest ajustado, HistGB y Ridge."""
    return VotingRegressor([
        ("rf", rf_ajustado(best_params, excluir=excluir)),
        ("histgb", modelo(HistGradientBoostingRegressor(**HISTGB), excluir=excluir)),
        ("ridge", modelo(Ridge(alpha=10.0), excluir=excluir)),
    ])


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


def pruebas_correccion(train: pd.DataFrame, cv, best_params: dict) -> pd.DataFrame:
    """Lo que se probó por la corrección de la Entrega 3 (que el modelo no achate tanto).

    Todo contra el Random Forest ajustado, con los mismos cortes temporales. Mejora > 0 = se equivoca menos.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, r2_score

    X, y = train[FEATURES], train["y"]

    def correr(fabrica, Xd=X, pesos=False, estirar=None):
        maes, r2s = [], []
        for a, b in cv.split(Xd):
            ya = y.iloc[a]
            kw = {"modelo__sample_weight": 1 + (ya - ya.median()).abs().values} if pesos else {}
            p = fabrica().fit(Xd.iloc[a], ya, **kw).predict(Xd.iloc[b])
            if estirar:
                p = ya.median() + estirar * (p - ya.median())
            maes.append(mean_absolute_error(y.iloc[b], p))
            r2s.append(r2_score(y.iloc[b], p))
        return np.array(maes), np.array(r2s)

    def con_trampa():
        m = rf_ajustado(best_params)
        m.named_steps["pre"].transformers.append(("trampa", "passthrough", ["en_saga"]))
        return m

    huber = lambda: modelo(GradientBoostingRegressor(loss="huber", n_estimators=400, learning_rate=0.04, max_depth=3,
                                                     subsample=0.8, min_samples_leaf=15, random_state=SEED))
    X_trampa = X.assign(en_saga=train["collection_id"].notna().astype(int).values)
    base_mae, base_r2 = correr(lambda: rf_ajustado(best_params))
    pruebas = [
        ("Promedio de 3 modelos (el final)", correr(lambda: modelo_final(best_params)),
         "Promedia Random Forest, HistGB y Ridge."),
        ("Pérdida Huber (GradientBoosting)", correr(huber),
         "Penaliza distinto los errores grandes, para no quedarse en la mediana."),
        ("Estirar las predicciones ×1,5", correr(lambda: rf_ajustado(best_params), estirar=1.5),
         "Aleja cada predicción de la mediana un 50%."),
        ("Más peso a los casos extremos", correr(lambda: rf_ajustado(best_params), pesos=True),
         "Al entrenar, cada película pesa 1 + su distancia a la mediana."),
        ("+ «pertenece a una saga» (trampa)", correr(con_trampa, Xd=X_trampa),
         "TMDb crea la saga cuando hay secuela, y eso pasa si a la primera le fue bien: se sabe después."),
    ]
    filas = [{"prueba": NOMBRE_RF + " (referencia)", "mae_cv": base_mae.mean(), "r2_cv": base_r2.mean(),
              "mejora": 0.0, "desvio_mejora": 0.0, "cortes_gana": "—", "supera_ruido": False,
              "que_es": "El modelo de la Entrega 3, con las columnas nuevas."}]
    for nombre, (maes, r2s), que in pruebas:
        dif = base_mae - maes
        filas.append({"prueba": nombre, "mae_cv": maes.mean(), "r2_cv": r2s.mean(), "mejora": dif.mean(),
                      "desvio_mejora": dif.std(ddof=0), "cortes_gana": f"{(dif > 0).sum()} de {len(dif)}",
                      "supera_ruido": bool(dif.mean() > dif.std(ddof=0)), "que_es": que})
    return pd.DataFrame(filas)
