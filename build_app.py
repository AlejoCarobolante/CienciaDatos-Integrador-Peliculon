"""De los datos a la app: entrena el modelo final y genera los artefactos que lee Streamlit.

    python build_app.py

1. Lee el CSV más reciente de include/output/movies_*.csv (o include/frozen/ultimo_ok.csv).
2. Aplica los filtros, las features, la partición temporal, el pipeline y la semilla de
   presentacion3.ipynb. El código es el mismo: los dos importan include/modelo/.
3. Entrena el Random Forest ajustado con el 80% de entrenamiento (como el notebook, no con todo).
4. Guarda en app/artefactos/:
   - modelo.joblib        el pipeline entrenado
   - referencias.joblib   medianas por género e historia por productora (features de un caso nuevo)
   - metricas.json        métricas, embudo, comparación de modelos, hipótesis de la Entrega 2, etc.
   - predicciones.parquet todas las películas del alcance con y real, y predicho, ROI y error
   - rangos.json          rangos y categorías vistas en entrenamiento (validación de «Probá el modelo»)
5. Control: el MAE de test tiene que coincidir con el que imprimió el notebook. Si no, falla.

Streamlit Cloud no corre el DAG ni el entrenamiento: los artefactos se commitean.
Si cambian los datos: volver a ejecutar el notebook y después este script.
"""
from __future__ import annotations

import json
import platform
import re
import sys
import time
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

from include.modelo import analisis as A  # noqa: E402
from include.modelo import entrenar as E  # noqa: E402
from include.modelo import features as F  # noqa: E402
from include.modelo.features import SEED  # noqa: E402

SALIDA = RAIZ / "app" / "artefactos"
NOTEBOOK = RAIZ / "presentacion3.ipynb"
FECHA_CONGELADO = "2026-10-24"  # corrida en modo full que generó include/frozen/ultimo_ok.csv


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fecha_de(csv: Path) -> str:
    m = re.search(r"movies_(\d{4}-\d{2}-\d{2})\.csv$", csv.name)
    return m.group(1) if m else FECHA_CONGELADO


def mae_test_del_notebook() -> float | None:
    """Lee de las salidas guardadas del notebook la línea «RF ajustado: MAE = ...» del test."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for celda in nb["cells"]:
        for salida in celda.get("outputs", []):
            texto = "".join(salida.get("text", ""))
            m = re.search(r"RF ajustado:\s+MAE = ([0-9.]+)", texto)
            if m:
                return float(m.group(1))
    return None


def tabla_folds(folds: pd.DataFrame) -> list[dict]:
    return [{"modelo": c, "fold": i + 1, "mae": float(v)} for c in folds.columns for i, v in enumerate(folds[c])]


def main() -> None:
    np.random.seed(SEED)
    SALIDA.mkdir(parents=True, exist_ok=True)

    # 1. Datos ---------------------------------------------------------------------------------
    csv = F.ruta_csv(RAIZ)
    df = F.cargar_dataset(csv)
    log(f"Dataset: {csv.relative_to(RAIZ)} -> {len(df):,} filas")
    data, pasos, ref = F.preparar_datos(df)
    train, test = E.particion_temporal(data)
    X_train, y_train = train[E.FEATURES], train["y"]
    X_test, y_test = test[E.FEATURES], test["y"]
    tss = E.cv_temporal()
    log(f"Alcance: {len(data):,} | train {len(train):,} | test {len(test):,}")

    folds_def = []
    for i, (idx_tr, idx_val) in enumerate(tss.split(train)):
        folds_def.append({"fold": i + 1,
                          "entrena": f"{train.iloc[idx_tr]['release_year'].min()}-{train.iloc[idx_tr]['release_year'].max()}",
                          "n_entrena": int(len(idx_tr)),
                          "valida": f"{train.iloc[idx_val]['release_year'].min()}-{train.iloc[idx_val]['release_year'].max()}",
                          "n_valida": int(len(idx_val))})

    # MAE sobre el ROI crudo (por qué no se usa)
    from sklearn.model_selection import cross_validate
    crudo = -cross_validate(DummyRegressor(strategy="median"), train, train["roi"], cv=tss,
                            scoring="neg_mean_absolute_error")["test_score"]

    # 2. Comparación de modelos y grilla --------------------------------------------------------
    log("Validación cruzada temporal de los candidatos...")
    candidatos = E.candidatos()
    folds = E.evaluar_mae_folds(candidatos, X_train, y_train, tss)

    log("Grilla de Random Forest...")
    grilla = E.grilla_rf(tss).fit(X_train, y_train)
    mejor = grilla.best_index_
    folds[E.NOMBRE_FINAL] = [-grilla.cv_results_[f"split{i}_test_score"][mejor] for i in range(E.N_FOLDS)]
    tabla_grilla = (pd.DataFrame(grilla.cv_results_)[["param_modelo__min_samples_leaf", "param_modelo__max_features",
                                                      "mean_test_score", "std_test_score"]]
                    .rename(columns={"param_modelo__min_samples_leaf": "min_samples_leaf",
                                     "param_modelo__max_features": "max_features",
                                     "mean_test_score": "mae", "std_test_score": "desvio"}))
    tabla_grilla["mae"] *= -1

    log("KFold al azar (contraste)...")
    folds_kf = E.evaluar_mae_folds(candidatos, X_train, y_train, KFold(5, shuffle=True, random_state=SEED))

    # 3. Modelo final y test (una sola medición, igual que el notebook) -------------------------
    log("Entrenando el modelo final con el 80% de entrenamiento...")
    modelo_final = E.rf_ajustado(grilla.best_params_).fit(X_train, y_train)
    piso = DummyRegressor(strategy="median").fit(X_train, y_train)
    pred_test, piso_test = modelo_final.predict(X_test), piso.predict(X_test)
    mae_final, mae_piso = mean_absolute_error(y_test, pred_test), mean_absolute_error(y_test, piso_test)

    # Test de todos los candidatos: informativo (la elección se hizo con validación cruzada)
    test_todos = {}
    for nombre, m in {**candidatos, E.NOMBRE_FINAL: E.rf_ajustado(grilla.best_params_)}.items():
        p = modelo_final.predict(X_test) if nombre == E.NOMBRE_FINAL else m.fit(X_train, y_train).predict(X_test)
        test_todos[nombre] = {"mae": float(mean_absolute_error(y_test, p)), "r2": float(r2_score(y_test, p))}

    modelos = []
    for nombre in folds.columns:
        f = folds[nombre]
        fila = {"modelo": nombre, "mae_cv": float(f.mean()), "desvio_cv": float(f.std(ddof=0)),
                "factor_cv": float(10 ** f.mean()),
                "mae_kfold_azar": float(folds_kf[nombre].mean()) if nombre in folds_kf else None,
                "mae_test": test_todos[nombre]["mae"], "r2_test": test_todos[nombre]["r2"],
                "es_final": nombre == E.NOMBRE_FINAL}
        for ref_nombre, clave in [("Piso (mediana)", "piso"), ("Ridge", "ridge")]:
            if nombre != ref_nombre:
                c = E.comparar(folds, ref_nombre, nombre)
                fila[f"mejora_vs_{clave}"] = float(c["mejora media"])
                fila[f"desvio_mejora_vs_{clave}"] = float(c["desvío de la mejora"])
                fila[f"folds_gana_vs_{clave}"] = c["folds que gana"]
                fila[f"supera_ruido_vs_{clave}"] = bool(c["supera el ruido"])
        modelos.append(fila)
    hist_vs_rf = E.comparar(folds, E.NOMBRE_FINAL, "HistGB (absolute_error)")

    # 4. Lectura del modelo: tramos, errores, importancia, popularidades, ablación ---------------
    res = test[["movie_id", "title", "release_year", "budget", "revenue", "roi"]].copy()
    res["y_real"], res["y_pred"] = y_test.values, pred_test
    res["roi_pred"] = 10 ** res["y_pred"] - 1
    res["error_abs"] = (res["y_real"] - res["y_pred"]).abs()
    res["tramo"] = E.tramo_roi(res["roi"])
    por_tramo = res.groupby("tramo", observed=True).agg(
        peliculas=("y_real", "size"), mae=("error_abs", "mean"),
        roi_real_mediano=("roi", "median"), roi_pred_mediano=("roi_pred", "median"),
        sesgo=("y_pred", lambda s: (s - res.loc[s.index, "y_real"]).mean())).reset_index()
    cols_err = ["title", "release_year", "budget", "revenue", "roi", "roi_pred"]
    fracasos = res.assign(e=res["y_pred"] - res["y_real"]).nlargest(5, "e")[cols_err]
    exitos = res.assign(e=res["y_real"] - res["y_pred"]).nlargest(5, "e")[cols_err]

    log("Importancia por permutación (test)...")
    imp = permutation_importance(modelo_final, X_test, y_test, scoring="neg_mean_absolute_error",
                                 n_repeats=20, random_state=SEED, n_jobs=-1)
    importancia = (pd.DataFrame({"variable": E.FEATURES, "aumento_mae": imp.importances_mean,
                                 "desvio": imp.importances_std})
                   .sort_values("aumento_mae", ascending=False).reset_index(drop=True))

    cols_pop = ["director_popularity", "lead_actor_popularity", "co_star_popularity"]
    spearman_pop = train[cols_pop + ["vote_count"]].corr(method="spearman")["vote_count"][cols_pop]
    sin_pop = E.rf_ajustado(grilla.best_params_, excluir=cols_pop).fit(X_train, y_train)
    mae_sin_pop = mean_absolute_error(y_test, sin_pop.predict(X_test))

    log("Ablación por grupos de columnas...")
    grupos = {"(todas las columnas)": [], "sin budget": ["budget"],
              "sin popularidades (director y elenco)": cols_pop, "sin productoras": E.COLS_PROD,
              "sin calendario (mes y día)": ["release_month", "release_day_of_week"],
              "sin idioma": ["original_language"], "sin géneros": ["primary_genre", "secondary_genre"],
              "sin runtime": ["runtime"], "sin release_year": ["release_year"]}
    ablacion = []
    for nombre, excluir in grupos.items():
        r = cross_validate(E.rf_ajustado(grilla.best_params_, excluir=excluir), X_train, y_train,
                           cv=tss, scoring="neg_mean_absolute_error")
        ablacion.append({"variante": nombre, "mae": float(-r["test_score"].mean()),
                         "desvio": float(r["test_score"].std())})
    for a in ablacion:
        a["cambio"] = a["mae"] - ablacion[0]["mae"]

    # 5. Predicciones de todo el alcance ---------------------------------------------------------
    pred = pd.concat([train.assign(conjunto="entrenamiento"), test.assign(conjunto="test")], ignore_index=True)
    pred["y_pred"] = modelo_final.predict(pred[E.FEATURES])
    pred.loc[pred["conjunto"] == "test", "y_pred"] = pred_test  # idénticas; se fija por las dudas
    pred["multiplo_real"] = 10 ** pred["y"]
    pred["multiplo_pred"] = 10 ** pred["y_pred"]
    pred["roi_pred"] = pred["multiplo_pred"] - 1
    pred["error_log"] = pred["y_pred"] - pred["y"]       # > 0: el modelo predijo de más
    pred["error_abs"] = pred["error_log"].abs()
    pred["factor_error"] = 10 ** pred["error_abs"]
    pred["tramo_roi"] = E.tramo_roi(pred["roi"]).astype(str)
    columnas = (["movie_id", "title"] + E.COLS_BASE + ["revenue", "vote_average", "vote_count"]
                + [c for c in E.FEATURES if c not in E.COLS_BASE]
                + ["roi", "y", "y_pred", "multiplo_real", "multiplo_pred", "roi_pred",
                   "error_log", "error_abs", "factor_error", "tramo_roi", "conjunto"])
    pred[columnas].to_parquet(SALIDA / "predicciones.parquet", index=False)

    # 6. Rangos y categorías vistas en entrenamiento ---------------------------------------------
    rangos = {"numericas": {}, "categoricas": {}}
    for c in ["budget", "runtime", "release_year", "director_popularity", "lead_actor_popularity",
              "co_star_popularity"]:
        s = train[c]
        rangos["numericas"][c] = {"min": float(s.min()), "max": float(s.max()), "p01": float(s.quantile(0.01)),
                                  "p99": float(s.quantile(0.99)), "mediana": float(s.median())}
    minimo_frecuencia = {**{c: 20 for c in E.COLS_CAT}, **{c: 30 for c in E.COLS_PROD}}
    for c in E.COLS_CAT + E.COLS_PROD:
        conteo = train[c].astype(str).value_counts()
        rangos["categoricas"][c] = {"categorias": conteo.index.tolist(), "conteo": conteo.astype(int).tolist(),
                                    "min_frecuencia": minimo_frecuencia[c]}
    (SALIDA / "rangos.json").write_text(json.dumps(rangos, ensure_ascii=False, indent=1), encoding="utf-8")

    # 7. Hallazgo de la Entrega 2: múltiplo mediano por escala de presupuesto ---------------------
    escalas = pd.cut(data["budget"], [0, 1e6, 1e7, 1e8, np.inf],
                     labels=["Menos de 1 M", "1 a 10 M", "10 a 100 M", "Más de 100 M"], right=False)
    por_escala = (data.assign(multiplo=10 ** data["y"], escala=escalas).groupby("escala", observed=True)
                  .agg(peliculas=("y", "size"), multiplo_mediano=("multiplo", "median"),
                       p25=("multiplo", lambda s: s.quantile(0.25)), p75=("multiplo", lambda s: s.quantile(0.75)),
                       pierde=("roi", lambda s: float((s < 0).mean()))).reset_index())
    por_escala["escala"] = por_escala["escala"].astype(str)
    spearman_budget_y = float(np.log10(data["budget"]).corr(data["y"], method="spearman"))

    # 8. metricas.json ---------------------------------------------------------------------------
    metricas = {
        "generado": date.today().isoformat(),
        "datos": {"csv": csv.relative_to(RAIZ).as_posix(), "fecha": fecha_de(csv), "filas": int(len(df)),
                  "columnas": int(df.shape[1]), "columnas_fuga": F.COLUMNAS_FUGA},
        "entorno": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                    "pandas": pd.__version__, "seed": SEED},
        "embudo": pasos,
        "particion": {"n_alcance": int(len(data)), "n_train": int(len(train)), "n_test": int(len(test)),
                      "anios_train": [int(train["release_year"].min()), int(train["release_year"].max())],
                      "anios_test": [int(test["release_year"].min()), int(test["release_year"].max())],
                      "folds": folds_def},
        "roi": {"mediana": float(data["roi"].median()), "p99": float(data["roi"].quantile(0.99)),
                "max": float(data["roi"].max()), "titulo_max": str(data.loc[data["roi"].idxmax(), "title"]),
                "mae_crudo_por_fold": [float(x) for x in crudo]},
        "modelos": modelos,
        "folds": tabla_folds(folds),
        "histgb_vs_final": {k: (v if isinstance(v, str) else float(v)) for k, v in hist_vs_rf.items()},
        "grilla": {"mejores": {k.replace("modelo__", ""): v for k, v in grilla.best_params_.items()},
                   "tabla": tabla_grilla.sort_values("mae").to_dict("records")},
        "final": {
            "nombre": E.NOMBRE_FINAL,
            "params": {k: v for k, v in E.params_rf(grilla.best_params_).items() if k != "n_jobs"},
            "mae_cv": float(folds[E.NOMBRE_FINAL].mean()), "desvio_cv": float(folds[E.NOMBRE_FINAL].std(ddof=0)),
            "mae_piso_cv": float(folds["Piso (mediana)"].mean()),
            "mae_test": float(mae_final), "factor_test": float(10 ** mae_final),
            "rmse_test": float(root_mean_squared_error(y_test, pred_test)),
            "r2_test": float(r2_score(y_test, pred_test)),
            "mae_piso_test": float(mae_piso), "factor_piso_test": float(10 ** mae_piso),
            "r2_piso_test": float(r2_score(y_test, piso_test)),
            "mejora_test_pct": float((mae_piso - mae_final) / mae_piso),
            "mejora_cv_pct": float((folds["Piso (mediana)"].mean() - folds[E.NOMBRE_FINAL].mean())
                                   / folds["Piso (mediana)"].mean()),
            "mediana_piso_y": float(piso.predict(X_test[:1])[0]),
        },
        "por_tramo": por_tramo.assign(tramo=por_tramo["tramo"].astype(str)).to_dict("records"),
        "fracasos_no_anticipados": fracasos.to_dict("records"),
        "exitos_no_anticipados": exitos.to_dict("records"),
        "importancia": importancia.to_dict("records"),
        "popularidades": {"spearman_vote_count": spearman_pop.round(4).to_dict(),
                          "mae_test_con": float(mae_final), "mae_test_sin": float(mae_sin_pop)},
        "ablacion": ablacion,
        "entrega2": A.fichas(df),
        "hallazgo_presupuesto": {"por_escala": por_escala.to_dict("records"),
                                 "spearman_logbudget_y": spearman_budget_y},
    }
    (SALIDA / "metricas.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=1, default=float),
                                         encoding="utf-8")
    joblib.dump(modelo_final, SALIDA / "modelo.joblib", compress=3)
    joblib.dump(ref, SALIDA / "referencias.joblib", compress=3)
    log(f"Artefactos en {SALIDA.relative_to(RAIZ)}: "
        + ", ".join(f"{p.name} ({p.stat().st_size / 1e6:.1f} MB)" for p in sorted(SALIDA.iterdir())))

    # 9. Control: tiene que coincidir con el notebook ---------------------------------------------
    mae_nb = mae_test_del_notebook()
    print(f"\nMAE test notebook = {mae_nb if mae_nb is None else f'{mae_nb:.4f}'} | "
          f"MAE test build_app = {mae_final:.4f}")
    if mae_nb is None or f"{mae_nb:.4f}" != f"{mae_final:.4f}":
        raise SystemExit("ERROR: el MAE de test no coincide con el del notebook. Volvé a ejecutar "
                         "presentacion3.ipynb con el mismo CSV y después este script.")
    print("OK: la app va a mostrar lo mismo que el notebook.")


if __name__ == "__main__":
    main()
