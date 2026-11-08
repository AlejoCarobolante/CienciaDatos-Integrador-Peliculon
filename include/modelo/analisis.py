"""Números de las fichas de hipótesis de la Entrega 2 (presentacion2.ipynb), recalculados.

Mismo método que el notebook: películas con budget > 0 y revenue > 0, ROI recortado al percentil 99.
build_app.py los recalcula con el CSV actual y los guarda para la sección «Lo que encontramos».
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def base_entrega2(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(df_fin, df_recortado): los dos universos que usan las fichas."""
    df_fin = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()
    df_fin["roi"] = (df_fin["revenue"] - df_fin["budget"]) / df_fin["budget"]
    q99 = df_fin["roi"].quantile(0.99)
    return df_fin, df_fin[df_fin["roi"] <= q99].copy()


def eta2(d: pd.DataFrame, col_cat: str, col_num: str = "roi") -> float:
    """Razón de correlación: SS_between / SS_total."""
    media = d[col_num].mean()
    ss_total = ((d[col_num] - media) ** 2).sum()
    ss_between = sum(len(g) * (g[col_num].mean() - media) ** 2 for _, g in d.groupby(col_cat))
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def fichas(df: pd.DataFrame) -> dict:
    df_fin, d = base_entrega2(df)
    log_b, log_r = np.log10(d["budget"]), np.log10(d["revenue"])

    # Ficha 1: presupuesto y recaudación
    f1 = {
        "pearson_logbudget_logrevenue": float(log_b.corr(log_r)),
        "pearson_logbudget_roi": float(log_b.corr(d["roi"])),
        "spearman_logbudget_roi": float(log_b.corr(d["roi"], method="spearman")),
    }
    # Ficha 2: idioma inglés vs. resto
    en, resto = d.loc[d["original_language"] == "en", "roi"], d.loc[d["original_language"] != "en", "roi"]
    s_p = np.sqrt(((len(en) - 1) * en.var() + (len(resto) - 1) * resto.var()) / (len(en) + len(resto) - 2))
    f2 = {"n_en": int(len(en)), "n_resto": int(len(resto)),
          "media_en": float(en.mean()), "media_resto": float(resto.mean()),
          "mediana_en": float(en.median()), "mediana_resto": float(resto.median()),
          "separacion_estandarizada": float((en.mean() - resto.mean()) / s_p)}
    # Ficha 3: género primario (géneros con 40 o más películas)
    conteo = d["primary_genre"].value_counts()
    dg = d[d["primary_genre"].isin(conteo[conteo >= 40].index)]
    f3 = {"eta2": eta2(dg, "primary_genre"),
          "mediana_por_genero": dg.groupby("primary_genre")["roi"].median().sort_values().round(4).to_dict()}
    # Ficha 4: popularidad del reparto
    log_cast = np.log10(d["lead_actor_popularity"] + d["co_star_popularity"] + 1)
    f4 = {"pearson": float(log_cast.corr(d["roi"])), "spearman": float(log_cast.corr(d["roi"], method="spearman"))}
    # Ficha 5: estacionalidad
    f5 = {"eta2_mes": eta2(d, "release_month"), "eta2_dia": eta2(d, "release_day_of_week")}

    return {
        "universo": {"con_montos": int(len(df_fin)), "recortado_p99": int(len(d)), "total": int(len(df))},
        "fichas": [
            {"id": 1, "nombre": "Presupuesto y recaudación",
             "afirmacion": "A mayor presupuesto, mayor recaudación.",
             "esperado": "Pearson ≥ 0,60 entre presupuesto y recaudación",
             "metrica": "Pearson entre log10(budget) y log10(revenue)",
             "valor": f1["pearson_logbudget_logrevenue"], "umbral": 0.60,
             "se_sostiene": f1["pearson_logbudget_logrevenue"] >= 0.60, **f1},
            {"id": 2, "nombre": "Idioma original",
             "afirmacion": "Las películas en inglés recuperan más que las del resto del mundo.",
             "esperado": "Separación estandarizada > 0,80 a favor del inglés",
             "metrica": "Separación estandarizada del ROI (inglés − resto) / s_p",
             "valor": f2["separacion_estandarizada"], "umbral": 0.80,
             "se_sostiene": f2["separacion_estandarizada"] > 0.80, **f2},
            {"id": 3, "nombre": "Género primario",
             "afirmacion": "El género separa fuerte el ROI.",
             "esperado": "η² > 0,25", "metrica": "η² del ROI por género primario",
             "valor": f3["eta2"], "umbral": 0.25, "se_sostiene": f3["eta2"] > 0.25, **f3},
            {"id": 4, "nombre": "Popularidad del reparto",
             "afirmacion": "Un reparto más popular se asocia con más ROI.",
             "esperado": "Pearson > 0,30", "metrica": "Pearson entre log10(popularidad del reparto + 1) y ROI",
             "valor": f4["pearson"], "umbral": 0.30, "se_sostiene": f4["pearson"] > 0.30, **f4},
            {"id": 5, "nombre": "Fecha de estreno",
             "afirmacion": "El mes o el día del estreno cambian mucho el ROI.",
             "esperado": "η² > 0,10", "metrica": "η² del ROI por mes de estreno",
             "valor": f5["eta2_mes"], "umbral": 0.10, "se_sostiene": f5["eta2_mes"] > 0.10, **f5},
        ],
    }
