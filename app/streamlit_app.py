"""Peliculón · ¿Cuántas veces recupera una película lo que costó?

App de la Entrega 4 (Ciencia de Datos, UTN FRM 2026). Forma de copa de martini:
  Portada -> Tallo (Los datos, Lo que encontramos, El modelo) -> Copa (Explorá, Probá el modelo).

Todo número que se muestra sale de app/artefactos/, que genera `python build_app.py`.
Todos los gráficos son Altair; colores y escalas vienen de app/estilo.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

APP = Path(__file__).resolve().parent
RAIZ = APP.parent
for p in (str(RAIZ), str(APP)):  # include.modelo (features) y estilo
    if p not in sys.path:
        sys.path.insert(0, p)

import estilo as S  # noqa: E402
from include.modelo import entrenar as E  # noqa: E402
from include.modelo import features as F  # noqa: E402

ART = APP / "artefactos"

st.set_page_config(page_title="Peliculón · ROI de películas", page_icon="🎬", layout="wide")
alt.data_transformers.disable_max_rows()


# =============================================================================================
# Carga de artefactos
# =============================================================================================
@st.cache_resource
def cargar_modelo():
    return joblib.load(ART / "modelo.joblib"), joblib.load(ART / "referencias.joblib")


@st.cache_data
def cargar_metricas() -> dict:
    return json.loads((ART / "metricas.json").read_text(encoding="utf-8"))


@st.cache_data
def cargar_rangos() -> dict:
    return json.loads((ART / "rangos.json").read_text(encoding="utf-8"))


@st.cache_data
def cargar_predicciones() -> pd.DataFrame:
    p = pd.read_parquet(ART / "predicciones.parquet")
    # Columnas de presentación (texto en formato rioplatense para los tooltips)
    for c in ["director_name", "lead_actor_name", "collection_name"]:
        p[c] = p[c].fillna("").astype(str)
    p["genero_es"] = p["primary_genre"].map(S.genero_es)
    p["genero_color"] = p["primary_genre"].map(S.genero_grupo)
    p["idioma_es"] = p["original_language"].map(S.idioma_es)
    p["presupuesto_txt"] = p["budget"].map(S.usd)
    p["recaudacion_txt"] = p["revenue"].map(S.usd)
    p["multiplo_real_txt"] = p["multiplo_real"].map(lambda v: S.mult(v, 2))
    p["multiplo_pred_txt"] = p["multiplo_pred"].map(lambda v: S.mult(v, 2))
    p["roi_real_txt"] = p["roi"].map(lambda v: S.pct(v, 0))
    p["roi_pred_txt"] = p["roi_pred"].map(lambda v: S.pct(v, 0))
    p["error_txt"] = np.where(p["error_log"] > 0,
                              "predijo " + p["factor_error"].map(lambda v: S.mult(v, 1)) + " de más",
                              "predijo " + p["factor_error"].map(lambda v: S.mult(v, 1)) + " de menos")
    return p


modelo, ref = cargar_modelo()
M = cargar_metricas()
R = cargar_rangos()
P = cargar_predicciones()
FIN = M["final"]

if sklearn.__version__ != M["entorno"]["scikit_learn"]:
    st.warning(f"El modelo se guardó con scikit-learn {M['entorno']['scikit_learn']} y esta app corre "
               f"{sklearn.__version__}. Las predicciones pueden no coincidir con el notebook.")


def como_lo_sabemos():
    return st.expander("Cómo lo sabemos", expanded=False)


def mostrar(chart, alto=None, **kw):
    return st.altair_chart(S.configurar(chart, alto), width="stretch", theme=None, **kw)


VAR_ES = {
    "budget": "Presupuesto", "budget_vs_genre_median": "Presupuesto / mediana del género",
    "prod_hist_roi": "Historia de la productora", "runtime": "Duración",
    "primary_production_company": "Productora principal", "original_language": "Idioma",
    "director_leverage": "Popularidad del director / presupuesto", "primary_genre": "Género principal",
    "director_vs_cast": "Director vs. protagonista", "lead_actor_popularity": "Popularidad del protagonista",
    "secondary_genre": "Género secundario", "co_production_company": "Coproductora",
    "star_power_ratio": "Popularidad del elenco / presupuesto", "release_year": "Año de estreno",
    "co_star_popularity": "Popularidad del coprotagonista", "director_popularity": "Popularidad del director",
    "power_pack": "Director × protagonista", "release_month": "Mes de estreno",
    "release_day_of_week": "Día de estreno", "is_horror_low_budget": "Terror de bajo presupuesto",
    "director_hist": "Historia del director", "director_n_prev": "Películas previas del director",
    "actor_hist": "Historia del protagonista", "actor_n_prev": "Películas previas del protagonista",
    "saga_hist": "Historia de la saga", "es_secuela": "Es secuela", "kw_secuela": "Palabra clave: secuela",
    "kw_remake": "Palabra clave: remake", "kw_libro": "Basada en un libro", "kw_comic": "Cómic o superhéroes",
    "kw_historia_real": "Historia real o biografía", "kw_videojuego": "Basada en un videojuego",
    "kw_navidad": "Navidad", "kw_musical": "Musical", "kw_found_footage": "Found footage",
    "paises_n": "Cantidad de países", "es_eeuu": "Producida en EE. UU.",
}

# Números que se repiten en varias secciones
embudo = pd.DataFrame(M["embudo"])
n_total, n_alcance = int(embudo["filas"].iloc[0]), int(embudo["filas"].iloc[-1])
imp = pd.DataFrame(M["importancia"])
peso_budget = imp.loc[0, "aumento_mae"] / imp.loc[1, "aumento_mae"]
modelos = pd.DataFrame(M["modelos"]).set_index("modelo")
anios_test = M["particion"]["anios_test"]

# Índice lateral
with st.sidebar:
    st.markdown("### Recorrido")
    st.markdown("- [El proyecto](#portada)\n- [Los datos](#datos)\n- [Lo que encontramos](#hallazgo)\n"
                "- [El modelo](#modelo)\n- [Explorá y probá](#copa)")
    st.caption(f"Datos de TMDb del {pd.Timestamp(M['datos']['fecha']).strftime('%d/%m/%Y')} · "
               f"scikit-learn {M['entorno']['scikit_learn']} · semilla {M['entorno']['seed']}")
    st.caption("[Repositorio en GitHub](https://github.com/AlejoCarobolante/CienciaDatos-Integrador-Peliculon)")


# =============================================================================================
# 1. PORTADA
# =============================================================================================
st.title("¿Cuántas veces recupera una película lo que costó?", anchor="portada")
st.markdown("**La pregunta:** con lo que se sabe *antes del estreno* (presupuesto, género, idioma, duración, "
            "productora, director, elenco, saga y fecha), ¿podemos anticipar cuántas veces recupera una película "
            "lo que costó?")

st.markdown(
    f"#### Con lo que se sabe antes del estreno, el modelo anticipa el múltiplo de recuperación con un error "
    f"típico de {S.mult(FIN['factor_test'])}: se equivoca un {S.pct(FIN['mejora_test_pct'], 1)} menos que adivinar "
    f"siempre la mediana. Es una mejora modesta, y lo que más pesa es el presupuesto.")

c1, c2, c3 = st.columns(3)
c1.metric("Error típico del modelo", S.mult(FIN["factor_test"]),
          help="10^MAE en el test: películas de 2017–2025 que el modelo nunca vio.")
c1.caption(f"Arriba o abajo del múltiplo real, en {S.num(M['particion']['n_test'])} películas de "
           f"{anios_test[0]}–{anios_test[1]} que el modelo nunca vio.")
c2.metric("Menos error que adivinar la mediana", S.pct(FIN["mejora_test_pct"], 1),
          help="(MAE del piso − MAE del modelo) / MAE del piso, en el test.")
c2.caption(f"El piso (predecir siempre la mediana) se equivoca {S.mult(FIN['factor_piso_test'])}; "
           f"el modelo, {S.mult(FIN['factor_test'])}.")
c3.metric("Cuánto más pesa el presupuesto", f"{S.num(peso_budget, 1)} veces",
          help="Importancia por permutación en el test: cuánto empeora el error al desordenar cada columna.")
c3.caption("que la segunda columna más importante: si se desordena el presupuesto, el error sube "
           f"{S.num(peso_budget, 1)} veces más que con cualquier otra.")
st.caption(f"Un error típico de {S.mult(FIN['factor_test'])} quiere decir que, si una película recuperó ×2, el modelo "
           f"suele decir algo entre {S.mult(2 / FIN['factor_test'], 1)} y {S.mult(2 * FIN['factor_test'], 1)}. Sirve "
           "para separar proyectos de riesgo muy distinto, no para afinar una cifra.")

st.divider()

# =============================================================================================
# 2. TALLO · LOS DATOS (Entrega 1)
# =============================================================================================
st.header(f"De {S.num(n_total)} películas de TMDb, solo {S.num(n_alcance)} "
          f"({S.pct(n_alcance / n_total)}) tienen presupuesto y recaudación creíbles", anchor="datos")

caida = embudo["filas"].shift(1) - embudo["filas"]
embudo["sale"] = caida.fillna(0).astype(int)
embudo["etiqueta"] = [f"{S.num(f)} ({S.pct(f / n_total)})" for f in embudo["filas"]]
embudo["orden"] = range(len(embudo))
mayor = embudo.iloc[1:]["sale"].idxmax()

# Gráfico: embudo de filtros.
#  datos: metricas.json["embudo"] (filas que quedan después de cada filtro)
#  marca: barra horizontal; mapeo: x = filas (cuantitativa, escala lineal desde 0), y = paso (ordinal, en orden)
#  anotación: texto con filas y % del total al final de cada barra, y la caída más grande marcada en tinta
#  (primera versión: orden alfabético y anotación en rojo; ver la guía de defensa)
base = alt.Chart(embudo)
y_paso = alt.Y("paso:N", sort=list(embudo["paso"]), title=None, axis=alt.Axis(labelLimit=260))
barras = base.mark_bar(color=S.REAL, cornerRadiusEnd=4, height=26).encode(
    x=alt.X("filas:Q", title="Películas que quedan", scale=alt.Scale(domain=[0, n_total * 1.18]),
            axis=alt.Axis(labelExpr="replace(format(datum.value, ','), ',', '.')")),
    y=y_paso,
    tooltip=[alt.Tooltip("paso:N", title="Paso"), alt.Tooltip("etiqueta:N", title="Quedan"),
             alt.Tooltip("sale:Q", title="Saca este paso", format=","), alt.Tooltip("motivo:N", title="Por qué")])
textos = base.mark_text(align="left", dx=6, color=S.TINTA, fontSize=12).encode(
    x="filas:Q", y=y_paso, text="etiqueta:N")
nota = alt.Chart(embudo.loc[[mayor]]).mark_text(align="left", dx=6, dy=16, color=S.TINTA, fontSize=12,
                                                fontWeight="bold").encode(
    x="filas:Q", y=y_paso, text=alt.value(f"−{S.num(embudo.loc[mayor, 'sale'])}: sin presupuesto o sin recaudación"))
mostrar(alt.layer(barras, textos, nota), alto=230)

st.markdown(
    f"Mirá el primer paso: **{S.num(embudo.loc[mayor, 'sale'])} películas** ({S.pct(embudo.loc[mayor, 'sale'] / n_total)}) "
    "no tienen presupuesto o recaudación en TMDb, y sin los dos montos no hay ROI. Los otros tres filtros sacan pocas: "
    + ", ".join(f"{S.num(s)}" for s in embudo["sale"].iloc[2:]) + " películas. "
    "Todo lo que sigue (el análisis, el modelo y la copa) trabaja con esas "
    f"{S.num(n_alcance)}.")

with como_lo_sabemos():
    st.markdown(f"""
**De dónde salen.** De la API oficial de [TMDb](https://www.themoviedb.org/) (The Movie Database), con un DAG de
Airflow (`dags/tmdb_ingest.py`, corrido con Astronomer) en modo `full`:

1. **Muestreo estratificado** (`discover_batches`): para no quedarnos solo con éxitos, se piden tres listas
   distintas, todas con al menos 100 votos: 4.000 películas **más populares**, 3.000 **peor calificadas** y
   3.000 de **mayor taquilla**. Los IDs repetidos se cuentan una vez.
2. **Capa bronce** (`land_bronze`): el JSON crudo de cada película, con sus créditos, comprimido en gzip.
3. **Capa plata** (`refine_silver`, `consolidate`, `validate`): una fila tipada de 18 columnas por película,
   sin duplicados y con controles de esquema (columnas, nulos, rango de votos).
4. **Respaldo**: si la API no responde, el DAG usa la última corrida buena (`include/frozen/ultimo_ok.csv`).

**Fecha de los datos:** corrida del {pd.Timestamp(M['datos']['fecha']).strftime('%d/%m/%Y')}
(`{M['datos']['csv']}`): {S.num(M['datos']['filas'])} películas × {M['datos']['columnas']} columnas. Se pidieron
10.000; las que faltan no se pudieron bajar por cortes de conexión con TMDb.

**Qué es una fila:** una película (`movie_id`), con su presupuesto y recaudación en USD, fecha de estreno
(año, mes y día de la semana), duración, idioma original, dos géneros, dos productoras, la popularidad del
director y de los dos primeros actores, y el promedio y la cantidad de votos.

**El filtrado, paso a paso:**
""")
    st.dataframe(embudo[["paso", "filas", "sale", "motivo"]].rename(
        columns={"paso": "Paso", "filas": "Quedan", "sale": "Saca", "motivo": "Por qué"}),
        hide_index=True, width="stretch")
    st.markdown(f"""
**Columnas descartadas por fuga:**
- `revenue`: es parte del objetivo (múltiplo = revenue / budget). Usarla sería darle la respuesta al modelo.
- `vote_average` y `vote_count`: existen recién después del estreno. No entran al modelo; se usan solo
  para auditar sus predicciones.

**El objetivo tiene una cola enorme:** la mediana del ROI es {S.num(M['roi']['mediana'], 2)} (recupera
{S.mult(M['roi']['mediana'] + 1, 1)} lo invertido), pero el máximo es {S.num(M['roi']['max'], 0)}
(*{M['roi']['titulo_max']}*). Por eso todo el proyecto lo mira en escala logarítmica (ver la sección siguiente).
""")

st.divider()

# =============================================================================================
# 3. TALLO · LO QUE ENCONTRAMOS (Entrega 2)
# =============================================================================================
esc = pd.DataFrame(M["hallazgo_presupuesto"]["por_escala"])
esc["orden"] = range(len(esc))
esc["mediana_txt"] = esc["multiplo_mediano"].map(lambda v: S.mult(v, 1))
esc["rango_txt"] = [f"{S.mult(a, 1)} a {S.mult(b, 1)}" for a, b in zip(esc["p25"], esc["p75"])]
esc["pierde_txt"] = esc["pierde"].map(S.pct)
esc["n_txt"] = esc["peliculas"].map(lambda v: f"{S.num(v)} películas")
barato, medio, caro = esc.iloc[0], esc.iloc[2], esc.iloc[3]
st.header(f"Las películas baratas multiplican mucho más lo que cuestan: {S.mult(barato['multiplo_mediano'], 0)} "
          f"con menos de 1 millón de dólares, {S.mult(medio['multiplo_mediano'], 0)} entre 10 y 100 millones",
          anchor="hallazgo")

# Gráfico: múltiplo mediano por escala de presupuesto.
#  datos: metricas.json["hallazgo_presupuesto"] (build_app agrupa las 6.391 películas por tramo de presupuesto
#         y calcula mediana y cuartiles del múltiplo revenue / budget con pandas)
#  mapeo: x = tramo de presupuesto (ordinal), y = múltiplo (escala LOG: ×1 a ×10 ocupa lo mismo que ×10 a ×100)
#  marcas: regla vertical del cuartil 1 al 3 (la mitad central de las películas) + punto en la mediana
#          + texto con la mediana; regla horizontal punteada en ×1 = recupera justo lo invertido
x_esc = alt.X("escala:N", sort=alt.SortField("orden"), title="Presupuesto de la película (USD)",
              axis=alt.Axis(labelAngle=0, labelFontSize=13))
y_esc = alt.Scale(type="log", domain=[0.5, 150], nice=False)
b = alt.Chart(esc)
tip = [alt.Tooltip("escala:N", title="Presupuesto"), alt.Tooltip("n_txt:N", title="Películas"),
       alt.Tooltip("mediana_txt:N", title="Múltiplo mediano"), alt.Tooltip("rango_txt:N", title="Mitad central"),
       alt.Tooltip("pierde_txt:N", title="Pierden plata")]
rango = b.mark_rule(color=S.REAL, strokeWidth=3, opacity=0.45).encode(
    x=x_esc, y=alt.Y("p25:Q", scale=y_esc, axis=S.eje_multiplo("Recaudación / presupuesto (escala log)")),
    y2="p75:Q", tooltip=tip)
punto = b.mark_circle(color=S.REAL, size=220, opacity=1).encode(x=x_esc, y=alt.Y("multiplo_mediano:Q", scale=y_esc),
                                                             tooltip=tip)
etiq = b.mark_text(align="left", dx=14, fontSize=15, fontWeight="bold", color=S.TINTA).encode(
    x=x_esc, y=alt.Y("multiplo_mediano:Q", scale=y_esc), text="mediana_txt:N")
cuenta = b.mark_text(dy=0, color=S.TINTA_SUAVE, fontSize=11, baseline="top").encode(
    x=x_esc, y=alt.value(0), text="n_txt:N")
uno = alt.Chart(pd.DataFrame({"y": [1], "t": ["×1: recupera justo lo invertido"]}))
linea_uno = uno.mark_rule(strokeDash=[4, 4], color=S.PISO).encode(y=alt.Y("y:Q", scale=y_esc))
texto_uno = uno.mark_text(align="left", dy=-7, color=S.TINTA_SUAVE, fontSize=11).encode(
    y=alt.Y("y:Q", scale=y_esc), x=alt.value(4), text="t:N")
mostrar(alt.layer(linea_uno, texto_uno, rango, punto, etiq, cuenta), alto=330)

st.markdown(
    f"Cada punto es la **mediana** del múltiplo (recaudación / presupuesto) en ese tramo de presupuesto, y la barra "
    f"cubre la mitad central de las películas. El eje es logarítmico. Las de menos de 1 millón son pocas "
    f"({S.num(barato['peliculas'])}) y muy dispersas: la mitad multiplica entre {barato['rango_txt']}. Las de más de "
    f"100 millones no siguen bajando ({S.mult(caro['multiplo_mediano'], 1)}) y pierden plata menos seguido "
    f"({S.pct(caro['pierde'])}) que las de 10 a 100 millones ({S.pct(medio['pierde'])}).")

fichas = {f["id"]: f for f in M["entrega2"]["fichas"]}
f1, f2 = fichas[1], fichas[2]
with como_lo_sabemos():
    st.markdown(f"""
En la Entrega 2 pusimos a prueba cinco hipótesis, cada una con un umbral fijado antes de mirar.
Acá están recalculadas con el dataset actual, con el mismo método del notebook `presentacion2.ipynb`:
las {S.num(M['entrega2']['universo']['con_montos'])} películas con los dos montos mayores a 0, recortando el 1%
de ROI más alto ({S.num(M['entrega2']['universo']['recortado_p99'])} películas).

**Una que se sostuvo: a más presupuesto, más recaudación.** Esperábamos un Pearson de al menos 0,60 y dio
**{S.num(f1['pearson_logbudget_logrevenue'], 2)}** entre log10(presupuesto) y log10(recaudación). Pero el
mismo cruce contra el ROI da **{S.num(f1['spearman_logbudget_roi'], 2)}** (Spearman): las caras recaudan más
en volumen y recuperan menos veces lo que costaron. Es lo que muestra el gráfico.

**Una que no: las películas en inglés recuperan más.** Esperábamos una separación estandarizada mayor a 0,80
a favor del inglés y dio **{S.num(f2['separacion_estandarizada'], 2)}**: el signo es el contrario. La mediana del
ROI es {S.num(f2['mediana_en'], 2)} en inglés y {S.num(f2['mediana_resto'], 2)} en el resto del mundo. Leído
junto con el gráfico, tiene sentido: Hollywood filma más caro.
""")
    tabla_f = pd.DataFrame([{"Hipótesis": f["afirmacion"], "Métrica": f["metrica"], "Esperábamos": f["esperado"],
                             "Dio": S.num(f["valor"], 4), "¿Se sostiene?": "Sí" if f["se_sostiene"] else "No"}
                            for f in fichas.values()])
    st.dataframe(tabla_f, hide_index=True, width="stretch")
    st.markdown(f"""
Las otras tres también cayeron: el género explica el {S.pct(fichas[3]['eta2'], 1)} de la variación del ROI
(η²), la popularidad del reparto se asocia levemente en contra ({S.num(fichas[4]['pearson'], 2)}) y el mes de
estreno explica el {S.pct(fichas[5]['eta2_mes'], 1)}. Por eso ninguna de esas columnas entra al modelo sola:
entran junto con el presupuesto, para que el modelo encuentre las combinaciones.

**Cómo se construyó el gráfico.** Con las {S.num(n_alcance)} películas del alcance del modelo (sección anterior).
`build_app.py` corta el presupuesto en cuatro tramos por potencias de 10 (1, 10 y 100 millones) y calcula, en
cada uno, la mediana y los cuartiles 1 y 3 de recaudación / presupuesto. Se usa la mediana y no el promedio
porque unos pocos fenómenos (×1.000 o más) arrastrarían el promedio.

**La transformación no obvia: log10.** El múltiplo va de ×0,001 a más de ×10.000. En escala lineal, unas pocas
películas aplastan a todas las demás contra el cero. En escala log10, cada salto de 10 veces ocupa lo mismo:
0 = recupera justo, 1 = ×10, −1 = pierde el 90%. El modelo predice exactamente eso:
y = log10(revenue / budget) = log10(1 + ROI). Así queda la distribución del objetivo:
""")
    # Gráfico: distribución del objetivo y.
    #  datos: predicciones.parquet (columna y)
    #  transformación: transform_bin agrupa y en intervalos de 0,1; mapeo: x = intervalo, y = count() de películas
    #  marca: barra; regla roja punteada en y = 0 (recupera justo lo invertido)
    hist = alt.Chart(P[["y"]]).transform_bin("y_bin", "y", bin=alt.Bin(step=0.1)).mark_bar(
        color=S.REAL, opacity=0.8).encode(
        x=alt.X("y_bin:Q", title="y = log10(recaudación / presupuesto)", bin="binned"),
        x2="y_bin_end:Q", y=alt.Y("count():Q", title="Películas"),
        tooltip=[alt.Tooltip("y_bin:Q", title="Desde", format=".1f"), alt.Tooltip("count():Q", title="Películas")])
    cero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=S.PISO, strokeDash=[4, 4]).encode(x="x:Q")
    mostrar(alt.layer(hist, cero), alto=220)

st.divider()

# =============================================================================================
# 4. TALLO · EL MODELO (Entrega 3)
# =============================================================================================
st.header(f"En películas que nunca vio, el modelo se equivoca un {S.pct(FIN['mejora_test_pct'], 1)} menos que "
          f"adivinar la mediana: una mejora real, pero chica", anchor="modelo")

orden_modelos = ["Piso (mediana)", "Ridge", "Random Forest", "HistGB (absolute_error)", E.NOMBRE_RF, E.NOMBRE_FINAL]
nombres_cortos = {"Piso (mediana)": "Piso: siempre la mediana", "Ridge": "Ridge (lineal)",
                  "Random Forest": "Random Forest sin ajustar", "HistGB (absolute_error)": "HistGB (boosting)",
                  E.NOMBRE_RF: "Random Forest ajustado",
                  E.NOMBRE_FINAL: "Promedio de los 3 (final)"}
mm = modelos.loc[orden_modelos].reset_index()
mm["nombre"] = mm["modelo"].map(nombres_cortos)
mm["rol"] = np.select([mm["es_final"], mm["modelo"] == "Piso (mediana)"], ["Modelo final", "Piso"], "Otros modelos")
folds_def = M["particion"]["folds"]
periodo_cv = f"{folds_def[0]['valida'].split('-')[0]}–{folds_def[-1]['valida'].split('-')[1]}"
largo = pd.concat([
    mm.assign(medida=f"Validación {periodo_cv} (media ± desvío)", mae=mm["mae_cv"], lo=mm["mae_cv"] - mm["desvio_cv"],
              hi=mm["mae_cv"] + mm["desvio_cv"]),
    mm.assign(medida=f"Test {anios_test[0]}–{anios_test[1]} (una sola vez)", mae=mm["mae_test"], lo=np.nan, hi=np.nan),
])
largo["factor_txt"] = largo["mae"].map(lambda v: S.mult(10 ** v, 2))
largo["mae_txt"] = largo["mae"].map(lambda v: S.num(v, 4))

# Gráfico: error de cada modelo.
#  datos: metricas.json["modelos"] en formato largo (una fila por modelo y medida)
#  mapeo: y = modelo (nominal, orden fijo del más simple al final), x = MAE en log10 (lineal, no arranca en 0
#         porque se comparan diferencias chicas), color = rol (final violeta, piso gris, otros gris claro),
#         forma = medida (círculo = validación, rombo = test)
#  marcas: regla horizontal = ± un desvío entre folds; punto = MAE; etiqueta del eje = MAE y factor ×
escala_rol = alt.Scale(domain=["Modelo final", "Piso", "Otros modelos"], range=[S.MODELO, S.PISO, "#a8a69c"])
escala_medida = alt.Scale(domain=list(largo["medida"].unique()), range=["circle", "diamond"])
y_mod = alt.Y("nombre:N", sort=[nombres_cortos[m] for m in orden_modelos], title=None,
              axis=alt.Axis(labelLimit=240, labelFontSize=13))
eje_mae = alt.Axis(title="Error medio (MAE sobre log10) y su factor ×. Menos es mejor", values=[0.36, 0.40, 0.44, 0.48],
                   labelExpr="replace(format(datum.value, '.2f'), '.', ',') + '  (×' + "
                             "replace(format(pow(10, datum.value), '.2f'), '.', ',') + ')'")
bm = alt.Chart(largo)
tip_m = [alt.Tooltip("modelo:N", title="Modelo"), alt.Tooltip("medida:N", title="Medida"),
         alt.Tooltip("mae_txt:N", title="MAE"), alt.Tooltip("factor_txt:N", title="Factor de error")]
desvio = bm.transform_filter(alt.datum.lo > 0).mark_rule(strokeWidth=2.5).encode(
    x=alt.X("lo:Q", scale=alt.Scale(domain=[0.35, 0.51], nice=False), axis=eje_mae), x2="hi:Q", y=y_mod,
    color=alt.Color("rol:N", scale=escala_rol, legend=alt.Legend(title=None)))
puntos_m = bm.mark_point(size=170, filled=True, opacity=1, stroke=S.SUPERFICIE, strokeWidth=1.5).encode(
    x=alt.X("mae:Q", scale=alt.Scale(domain=[0.35, 0.51], nice=False)), y=y_mod,
    color=alt.Color("rol:N", scale=escala_rol, legend=alt.Legend(title=None)),
    shape=alt.Shape("medida:N", scale=escala_medida, legend=alt.Legend(title=None, labelLimit=320)), tooltip=tip_m)
mejora_txt = (f"Test: {S.mult(FIN['factor_test'])} contra {S.mult(FIN['factor_piso_test'])} del piso "
              f"({S.pct(FIN['mejora_test_pct'], 1)} menos error)")
nota_m = alt.Chart(pd.DataFrame({"x": [FIN["mae_test"]], "nombre": [nombres_cortos[E.NOMBRE_FINAL]], "t": [mejora_txt]})
                   ).mark_text(align="right", dx=-4, dy=-16, color=S.MODELO, fontSize=12, fontWeight="bold").encode(
    x=alt.X("x:Q", scale=alt.Scale(domain=[0.35, 0.51])), y=y_mod, text="t:N")
mostrar(alt.layer(desvio, puntos_m, nota_m), alto=330)

rf, ridge = modelos.loc[E.NOMBRE_FINAL], modelos.loc["Ridge"]
rf_aj = modelos.loc[E.NOMBRE_RF]
texto_test = (f"En el test también le gana a Ridge ({S.num(rf['mae_test'], 3)} contra {S.num(ridge['mae_test'], 3)})."
              if ridge["mae_test"] - rf["mae_test"] > 0.01 else
              f"En el test, en cambio, quedan cerca (Ridge {S.num(ridge['mae_test'], 3)}, modelo final "
              f"{S.num(rf['mae_test'], 3)}).")
st.markdown(
    f"El círculo es el error medio en los 5 cortes de validación temporal y la línea, ± un desvío. El rombo es el "
    f"**test** ({anios_test[0]}–{anios_test[1]}), medido una vez al final. El modelo final es el promedio del Random "
    f"Forest ajustado, HistGB y Ridge. En validación le gana a Ridge en {rf['folds_gana_vs_ridge']} cortes. "
    f"{texto_test} Lo que el modelo sí logra, siempre, es ganarle a adivinar.")

tramos = pd.DataFrame(M["por_tramo"])
with como_lo_sabemos():
    st.markdown(f"""
**La métrica: MAE sobre log10(revenue / budget).** Lo caro es errarle al *múltiplo*: prever que una película
triplica su costo y que lo recupere a medias es el mismo error en una de 1 millón que en una de 200. Un MAE de
{S.num(FIN['mae_test'], 2)} significa que la predicción está errada, en promedio, por un factor de
10^{S.num(FIN['mae_test'], 2)} = {S.mult(FIN['factor_test'])} para arriba o para abajo.
**¿Por qué no el ROI crudo?** Lo probamos: el MAE del piso sobre el ROI crudo da, por corte,
{", ".join(S.num(x, 1) for x in M['roi']['mae_crudo_por_fold'])}. El corte 4 se dispara porque cae *Paranormal
Activity* (ROI ≈ 12.900). Con ese desvío entre cortes no se distingue ningún modelo del piso: la métrica mide
dónde cayó el outlier, no qué modelo es mejor.

**El piso:** `DummyRegressor(strategy="median")`, que predice siempre la mediana del entrenamiento
(y = {S.num(FIN['mediana_piso_y'], 2)}, o sea {S.mult(10 ** FIN['mediana_piso_y'])}) sin mirar ninguna columna.

**La validación imita el uso:** el modelo se usaría para películas futuras con lo aprendido de las pasadas.
1. Las películas se ordenan por fecha de estreno.
2. El **20% más reciente** ({S.num(M['particion']['n_test'])} películas, {anios_test[0]}–{anios_test[1]}) es el test.
3. Sobre el 80% restante ({S.num(M['particion']['n_train'])} películas,
   {M['particion']['anios_train'][0]}–{M['particion']['anios_train'][1]}), `TimeSeriesSplit(5)`: cada corte
   entrena con el pasado y valida con el período siguiente.

Todo el preprocesamiento (logaritmos, imputación, escalado y one-hot) está dentro de un `Pipeline` que se
ajusta en cada corte solo con su entrenamiento. Semilla {M['entorno']['seed']} en todo.
""")
    st.dataframe(pd.DataFrame(M["particion"]["folds"]).rename(columns={
        "fold": "Corte", "entrena": "Entrena con", "n_entrena": "Películas", "valida": "Valida con",
        "n_valida": "Películas "}), hide_index=True, width="stretch")
    rf_simple = modelos.loc["Random Forest"]
    grilla = pd.DataFrame(M["grilla"]["tabla"])
    st.markdown(f"""
Si los cortes se armaran al azar (`KFold` mezclando épocas), el Random Forest daría un MAE de
{S.num(modelos.loc['Random Forest', 'mae_kfold_azar'], 4)} en vez de {S.num(rf_simple['mae_cv'], 4)}: más
optimista, porque mete películas del futuro y secuelas de los dos lados.

**La comparación de modelos.** Todos con el mismo pipeline y los mismos 5 cortes. La mejora se mide **corte
contra corte** (algunos períodos son más difíciles para todos): la mejora media tiene que superar su desvío.
""")
    tabla_m = pd.DataFrame({
        "Modelo": [nombres_cortos[m] for m in orden_modelos],
        "MAE validación": [S.num(modelos.loc[m, "mae_cv"], 4) for m in orden_modelos],
        "Desvío": [S.num(modelos.loc[m, "desvio_cv"], 4) for m in orden_modelos],
        "Factor": [S.mult(modelos.loc[m, "factor_cv"]) for m in orden_modelos],
        "Mejora vs. piso (desvío)": ["—"] + [
            f"{S.num(modelos.loc[m, 'mejora_vs_piso'], 4)} ({S.num(modelos.loc[m, 'desvio_mejora_vs_piso'], 4)}) · "
            f"{modelos.loc[m, 'folds_gana_vs_piso']} {'✓' if modelos.loc[m, 'supera_ruido_vs_piso'] else '✗'}"
            for m in orden_modelos[1:]],
        "Mejora vs. Ridge (desvío)": ["—", "—"] + [
            f"{S.num(modelos.loc[m, 'mejora_vs_ridge'], 4)} ({S.num(modelos.loc[m, 'desvio_mejora_vs_ridge'], 4)}) · "
            f"{modelos.loc[m, 'folds_gana_vs_ridge']} {'✓' if modelos.loc[m, 'supera_ruido_vs_ridge'] else '✗'}"
            for m in orden_modelos[2:]],
        "MAE test (informativo)": [S.num(modelos.loc[m, "mae_test"], 4) for m in orden_modelos],
    })
    st.dataframe(tabla_m, hide_index=True, width="stretch")
    hv, fv = M["histgb_vs_final"], M["final_vs_rf"]
    st.markdown(f"""
✓ = la mejora media supera su desvío. **Por qué un promedio de tres modelos:** entre el Random Forest ajustado,
HistGB y el promedio, la diferencia es ruido (el promedio mejora {S.num(fv['mejora media'], 4)} al Random Forest con
un desvío de {S.num(fv['desvío de la mejora'], 4)}; HistGB, {S.num(hv['mejora media'], 4)} con un desvío de
{S.num(hv['desvío de la mejora'], 4)}). Elegimos el promedio por estabilidad: es el que le gana a Ridge de forma más
pareja ({S.num(rf['mejora_vs_ridge'], 4)} ± {S.num(rf['desvio_mejora_vs_ridge'], 4)}, {rf['folds_gana_vs_ridge']}
cortes) y no depende de un solo modelo. Es el mismo criterio que usamos en la Entrega 3 entre modelos empatados.

**El ajuste del elegido.** Grilla sobre `min_samples_leaf` (1, 5, 10, 20, 40) y `max_features` (0,3, 0,6, 1,0),
con los mismos 5 cortes. Ganó `min_samples_leaf = {M['grilla']['mejores']['min_samples_leaf']}` y
`max_features = {S.num(M['grilla']['mejores']['max_features'], 1)}`, con 300 árboles. Entre la mejor y la peor de
las {len(grilla)} combinaciones hay {S.num(grilla['mae'].max() - grilla['mae'].min(), 4)} de MAE, menos que el
desvío entre cortes ({S.num(grilla['desvio'].min(), 4)} o más): el modelo es poco sensible a estos hiperparámetros.
El ajuste mejora {S.num(rf_simple['mae_cv'] - rf_aj['mae_cv'], 4)} al Random Forest sin ajustar.

**El resultado en el test** ({S.num(M['particion']['n_test'])} películas, mirado una vez): MAE
{S.num(FIN['mae_test'], 4)} ({S.mult(FIN['factor_test'])}), contra {S.num(FIN['mae_piso_test'], 4)}
({S.mult(FIN['factor_piso_test'])}) del piso. R² = {S.num(FIN['r2_test'], 2)}. El test da peor que la validación
para todos los modelos, también para el piso ({S.num(FIN['mae_piso_cv'], 4)} en validación): {anios_test[0]}–{anios_test[1]}
es un período más difícil de anticipar (probablemente influyen la pandemia y los estrenos en streaming).

**Dónde se equivoca: el modelo achata.** Cerca de la mediana acierta bastante; en las puntas, no:
""")
    # Gráfico: múltiplo mediano real vs. predicho por tramo de ROI (test).
    #  datos: metricas.json["por_tramo"], en formato largo (real / predicho)
    #  mapeo: y = tramo (ordinal), x = múltiplo (escala log), color = real (gris tinta) o predicho (violeta)
    #  marcas: regla entre los dos valores (la distancia es el error) + punto para cada uno
    tramos["real"] = tramos["roi_real_mediano"] + 1
    tramos["predicho"] = tramos["roi_pred_mediano"] + 1
    tl = tramos.melt(id_vars=["tramo", "peliculas", "mae"], value_vars=["real", "predicho"], var_name="valor",
                     value_name="multiplo")
    tl["valor"] = tl["valor"].map({"real": "Real", "predicho": "Predicho por el modelo"})
    tl["multiplo_txt"] = tl["multiplo"].map(lambda v: S.mult(v, 2))
    y_tr = alt.Y("tramo:N", sort=list(tramos["tramo"]), title="Tramo de ROI real", axis=alt.Axis(labelLimit=200))
    x_tr = alt.X("multiplo:Q", scale=alt.Scale(type="log", domain=[0.2, 30], nice=False),
                 axis=S.eje_multiplo("Múltiplo mediano (escala log)"))
    une = alt.Chart(tramos).mark_rule(color=S.EJE, strokeWidth=2).encode(
        y=y_tr, x=alt.X("real:Q", scale=alt.Scale(type="log", domain=[0.2, 30], nice=False)), x2="predicho:Q")
    pts = alt.Chart(tl).mark_circle(size=150, opacity=1).encode(
        y=y_tr, x=x_tr,
        color=alt.Color("valor:N", scale=alt.Scale(domain=["Real", "Predicho por el modelo"],
                                                   range=[S.REAL, S.MODELO]), legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip("tramo:N", title="Tramo"), alt.Tooltip("valor:N", title=" "),
                 alt.Tooltip("multiplo_txt:N", title="Múltiplo mediano"), alt.Tooltip("peliculas:Q", title="Películas")])
    mostrar(alt.layer(une, pts), alto=230)
    perdida, top = tramos.iloc[0], tramos.iloc[-1]
    st.markdown(f"""
A las películas que perdieron plata ({S.num(perdida['peliculas'])} en el test) les predice, en mediana,
{S.mult(perdida['predicho'], 2)}; a las que multiplicaron más de 11 veces, {S.mult(top['predicho'], 2)}.
Las predicciones cubren solo una parte de la dispersión real: el modelo no anticipa ni los desastres ni los fenómenos.
""")
    cols_e = {"title": "Película", "release_year": "Año", "budget": "Presupuesto", "revenue": "Recaudación",
              "roi": "ROI real", "roi_pred": "ROI predicho"}

    def tabla_errores(filas):
        t = pd.DataFrame(filas)
        t["release_year"] = t["release_year"].astype(int)
        t["budget"], t["revenue"] = t["budget"].map(S.usd), t["revenue"].map(S.usd)
        t["roi"], t["roi_pred"] = t["roi"].map(S.pct), t["roi_pred"].map(S.pct)
        return t.rename(columns=cols_e)

    ca, cb = st.columns(2)
    ca.markdown("**Fracasos no anticipados** (predijo mucho más)")
    ca.dataframe(tabla_errores(M["fracasos_no_anticipados"]), hide_index=True, width="stretch")
    cb.markdown("**Éxitos no anticipados** (predijo mucho menos)")
    cb.dataframe(tabla_errores(M["exitos_no_anticipados"]), hide_index=True, width="stretch")
    st.markdown("""
Varios «fracasos» del test son estrenos de streaming con una recaudación simbólica en TMDb
(*Red Notice*, *The Gray Man*, *The Killer*: Netflix). Son un problema de datos, no del modelo. No los
sacamos porque eso sería tocar el test después de mirarlo.

**En qué se apoya: importancia por permutación.** Se desordena una columna por vez en el test y se mide cuánto
sube el error. Las que más pesan:
""")
    # Gráfico: importancia por permutación (12 columnas que más pesan).
    #  datos: metricas.json["importancia"]; mapeo: y = columna (ordenada por importancia), x = aumento del MAE
    #  marcas: barra violeta (es el modelo) + regla gris = ± un desvío entre las 20 repeticiones
    im = imp.head(12).copy()
    im["nombre"] = im["variable"].map(VAR_ES)
    im["txt"] = im["aumento_mae"].map(lambda v: S.num(v, 4))
    im["lo"], im["hi"] = im["aumento_mae"] - im["desvio"], im["aumento_mae"] + im["desvio"]
    y_im = alt.Y("nombre:N", sort=list(im["nombre"]), title=None, axis=alt.Axis(labelLimit=280))
    barra_im = alt.Chart(im).mark_bar(color=S.MODELO, cornerRadiusEnd=4, height=14).encode(
        x=alt.X("aumento_mae:Q", title="Cuánto sube el MAE al desordenar la columna"), y=y_im,
        tooltip=[alt.Tooltip("nombre:N", title="Columna"), alt.Tooltip("variable:N", title="En el código"),
                 alt.Tooltip("txt:N", title="Aumento del MAE")])
    err_im = alt.Chart(im).mark_rule(color=S.TINTA_2).encode(x="lo:Q", x2="hi:Q", y=y_im)
    mostrar(alt.layer(barra_im, err_im), alto=320)
    abl = pd.DataFrame(M["ablacion"])
    pop = M["popularidades"]
    st.markdown(f"""
La prueba al revés (sacar un grupo de columnas y volver a validar) confirma lo mismo: sin el presupuesto el error
sube {S.num(abl.loc[abl['variante'] == 'sin budget', 'cambio'].iloc[0], 4)}; sin el contexto nuevo (director,
actor, saga, palabras clave y países), sube {S.num(abl.loc[abl['variante'].str.startswith('sin el contexto'), 'cambio'].iloc[0], 4)};
sin las popularidades cambia {S.num(abl.loc[abl['variante'].str.startswith('sin popularidades'), 'cambio'].iloc[0], 4)}.

**La limitación de las popularidades.** TMDb da la popularidad de la persona **hoy**, no la que tenía antes del
estreno: si una película fue un éxito, su director y su elenco quedan más populares. Su correlación con
`vote_count` (éxito posterior) es de {S.num(min(pop['spearman_vote_count'].values()), 2)} a
{S.num(max(pop['spearman_vote_count'].values()), 2)}. Las dejamos porque el modelo casi no las usa: sin ellas el
MAE de test da {S.num(pop['mae_test_sin'], 4)} contra {S.num(pop['mae_test_con'], 4)} con ellas.

**Lo que marcó el profe en la Entrega 3: el modelo achata.** Acertaba cerca de la mediana y no se animaba en los
extremos. Sugirió probar una penalización que lo aleje de la mediana y mirar también el R². Lo que hicimos:
- **Más información antes del estreno.** El DAG ahora baja el director, el actor principal, la saga, las palabras
  clave y los países. Se agregaron a la misma corrida del 24/10 (mismas películas). Con eso se arman la historia del
  director, del actor y de la saga (mediana del múltiplo en sus películas de años anteriores), si es secuela, de qué
  tipo de historia se trata (libro, cómic, historia real, remake...) y de dónde es.
- **Modelo final: el promedio de tres modelos** (Random Forest ajustado, HistGB y Ridge), el que le gana a Ridge
  de forma más estable.
- **Resultado:** en el test, el error pasa de ×2,86 (Entrega 3) a {S.mult(FIN['factor_test'])}, y la mejora contra el
  piso, de 7% a {S.pct(FIN['mejora_test_pct'], 1)}. Lo que más aporta son las columnas nuevas: sin ellas, el Random Forest
  empeora {S.num(abl.loc[abl['variante'].str.startswith('sin el contexto'), 'cambio'].iloc[0], 4)}.
- **Las estrategias que pidió probar**, todas contra el Random Forest ajustado con los mismos cortes:
""")
    pc = pd.DataFrame(M["pruebas_correccion"])
    st.dataframe(pd.DataFrame({
        "Prueba": pc["prueba"], "Qué hace": pc["que_es"], "MAE": pc["mae_cv"].map(lambda v: S.num(v, 4)),
        "R²": pc["r2_cv"].map(lambda v: S.num(v, 3)),
        "Mejora (desvío) · cortes": ["—"] + [f"{S.num(a, 4)} ({S.num(b, 4)}) · {c} {'✓' if d else '✗'}"
                                             for a, b, c, d in zip(pc["mejora"][1:], pc["desvio_mejora"][1:],
                                                                   pc["cortes_gana"][1:], pc["supera_ruido"][1:])]}),
        hide_index=True, width="stretch")
    st.markdown(f"""
Huber empata con el Random Forest (la diferencia no supera el ruido) y empeora el R²; estirar y dar más peso a los
extremos **empeoran** en los 5 cortes. El modelo no achata por cómo está configurado, sino
porque con lo que se sabe antes del estreno es difícil distinguir un fenómeno de una película normal. Lo que sí
mejora es **darle más información** y promediar modelos. «Pertenece a una saga» parece la mejor de todas, pero es
una trampa: TMDb crea la saga cuando hay secuela, y eso pasa cuando a la primera le fue bien. Por eso solo usamos
«es secuela», que sí se sabe antes.

**Las correcciones anteriores, de la misma entrega.**
- El modelo final es uno solo en todo el notebook. La versión de las correcciones describía un Random Forest y
  entrenaba como final un HistGB.
- Ese HistGB usaba `loss="huber"`, que **no existe** en `HistGradientBoostingRegressor` (y `quantile` solo aplica
  con `loss="quantile"`): el notebook no corría. El candidato de boosting quedó con `loss="absolute_error"`.
- `prod_hist_roi` (historia de la productora) usa solo años **estrictamente anteriores** al estreno.
- `budget_vs_genre_median` divide por la mediana del género calculada sobre todo el alcance, test incluido. No usa
  el objetivo; lo dejamos así para no volver a tocar el test, y queda declarado.
- El test se miró dos veces en la Entrega 3 (para limpiar recaudaciones imposibles) y, al corregir, se reportó el
  test de todos los candidatos como información. La elección se hizo con la validación.

**Por qué Streamlit y no una página HTML.** El modelo final promedia un bosque de 300 árboles, HistGB y Ridge: no
hay una fórmula corta que se pueda escribir dentro de una página para calcular un caso nuevo. Streamlit corre Python, así que
carga el mismo `modelo.joblib` que generó `build_app.py` y aplica las mismas features.
La otra salida era llevar al navegador un modelo cuya fórmula se puede escribir: Ridge. Queda a
{S.num(ridge['mae_cv'] - FIN['mae_cv'], 4)} de MAE del elegido, menos que un desvío entre cortes
({S.num(FIN['desvio_cv'], 4)}), así que la consigna lo habría permitido. No lo elegimos porque, comparado corte
contra corte, pierde en los 5 cortes y por más que el desvío de esa diferencia
({S.num(rf['mejora_vs_ridge'], 4)} ± {S.num(rf['desvio_mejora_vs_ridge'], 4)}): preferimos mostrar el modelo que
defendemos y no uno peor.
""")

st.divider()

# =============================================================================================
# 5 y 6. COPA
# =============================================================================================
st.header("Ahora te toca a vos", anchor="copa")
st.markdown("Ya sabés qué mirar: el presupuesto manda, el modelo todavía achata las puntas y se equivoca, en promedio, "
            f"{S.mult(FIN['factor_test'])}. Recorré las películas o cargá una nueva.")
tab_explora, tab_proba = st.tabs(["Explorá las películas", "Probá el modelo"])

# ---------------------------------------------------------------------------------------------
# 5. Explorá
# ---------------------------------------------------------------------------------------------
with tab_explora:
    anio_min, anio_max = int(P["release_year"].min()), int(P["release_year"].max())
    generos_orden = P["genero_es"].value_counts().index.tolist()
    idiomas_orden = P["idioma_es"].value_counts().index.tolist()

    k1, k2, k3, k4 = st.columns([2, 2, 2, 1.4])
    rango_anios = k1.slider("Años de estreno", anio_min, anio_max, (anio_min, anio_max))
    generos_sel = k2.multiselect("Géneros", generos_orden, placeholder="Todos los géneros")
    idiomas_sel = k3.multiselect("Idioma original", idiomas_orden, placeholder="Todos los idiomas")
    conjunto = k4.radio("Películas", ["Todas", "Solo test", "Solo entrenamiento"],
                        help="Test: 2017–2025, el modelo nunca las vio. Entrenamiento: el modelo las vio al aprender, "
                             "así que su predicción es optimista.")
    k5, k6 = st.columns([2, 3])
    eje_y = k5.radio("Eje vertical", ["Múltiplo real", "Múltiplo predicho por el modelo"], horizontal=True)
    color_por = k6.radio("Color", ["Error del modelo", "Género"], horizontal=True)

    # Filtros: los mismos predicados se aplican en el gráfico (transform_filter) y en la tabla de conteo
    predicados = [alt.FieldRangePredicate(field="release_year", range=list(rango_anios))]
    vista = P[P["release_year"].between(*rango_anios)]
    if generos_sel:
        predicados.append(alt.FieldOneOfPredicate(field="genero_es", oneOf=generos_sel))
        vista = vista[vista["genero_es"].isin(generos_sel)]
    if idiomas_sel:
        predicados.append(alt.FieldOneOfPredicate(field="idioma_es", oneOf=idiomas_sel))
        vista = vista[vista["idioma_es"].isin(idiomas_sel)]
    if conjunto != "Todas":
        valor = "test" if conjunto == "Solo test" else "entrenamiento"
        predicados.append(alt.FieldEqualPredicate(field="conjunto", equal=valor))
        vista = vista[vista["conjunto"] == valor]

    campo_y = "multiplo_real" if eje_y == "Múltiplo real" else "multiplo_pred"
    if color_por == "Error del modelo":
        color = alt.Color("error_log:Q", scale=S.escala_error(),
                          legend=alt.Legend(title="Error del modelo", orient="top", direction="horizontal",
                                            gradientLength=320, titleLimit=400, values=[-1, 0, 1],
                                            labelExpr="datum.value == 0 ? 'acierta' : (datum.value < 0 ? "
                                                      "'predijo 10 veces menos' : 'predijo 10 veces más')"))
    else:
        color = alt.Color("genero_color:N", scale=S.escala_genero(),
                          legend=alt.Legend(title="Género principal", orient="top"))

    # Gráfico: panorama de todas las películas del alcance.
    #  datos: predicciones.parquet (6.391 películas, con lo que predijo el modelo)
    #  transformaciones: transform_filter con los controles de arriba (años, géneros, idiomas, train/test)
    #  mapeo: x = presupuesto (escala LOG), y = múltiplo real o predicho (escala LOG),
    #         color = error del modelo (divergente) o género (3 con color + otros en gris), tooltip = ficha corta
    #  marca: círculo chico semitransparente; selección por clic sobre movie_id (abre la ficha de abajo)
    sel = alt.selection_point(name="peli", fields=["movie_id"], on="click", clear="dblclick")
    puntos = alt.Chart(P).mark_circle(size=30, stroke=S.SUPERFICIE, strokeWidth=0.4).encode(
        x=alt.X("budget:Q", scale=S.escala_budget_log(), axis=S.eje_budget()),
        y=alt.Y(f"{campo_y}:Q", scale=S.escala_multiplo_log(), axis=S.eje_multiplo(f"{eje_y} (recaudación / presupuesto, log)")),
        color=color,
        opacity=alt.condition(sel, alt.value(0.75), alt.value(0.15)),  # al hacer clic, el resto se apaga
        tooltip=[alt.Tooltip("title:N", title="Película"), alt.Tooltip("release_year:Q", title="Año"),
                 alt.Tooltip("genero_es:N", title="Género"), alt.Tooltip("presupuesto_txt:N", title="Presupuesto"),
                 alt.Tooltip("recaudacion_txt:N", title="Recaudación"),
                 alt.Tooltip("multiplo_real_txt:N", title="Recuperó (real)"),
                 alt.Tooltip("multiplo_pred_txt:N", title="Predijo el modelo"),
                 alt.Tooltip("roi_real_txt:N", title="ROI real"), alt.Tooltip("roi_pred_txt:N", title="ROI predicho"),
                 alt.Tooltip("error_txt:N", title="Error"), alt.Tooltip("conjunto:N", title="Conjunto")],
    ).add_params(sel)
    for pr in predicados:
        puntos = puntos.transform_filter(pr)
    linea = alt.Chart(pd.DataFrame({"y": [1]})).mark_rule(strokeDash=[4, 4], color=S.PISO).encode(
        y=alt.Y("y:Q", scale=S.escala_multiplo_log()))
    evento = mostrar(alt.layer(linea, puntos), alto=520, on_select="rerun", selection_mode="peli", key="panorama")

    st.caption(f"{S.num(len(vista))} películas en la vista. La línea punteada es ×1: arriba recuperaron más de lo que "
               "costaron, abajo perdieron plata. Pasá el mouse por un punto para ver su ficha corta; hacé clic para "
               "abrirla abajo (doble clic para soltarla).")

    elegido = None
    seleccion = (evento.selection.get("peli") if evento is not None and hasattr(evento, "selection") else None) or []
    if seleccion and "movie_id" in seleccion[0]:
        elegido = int(seleccion[0]["movie_id"])
    opciones = vista.sort_values("vote_count", ascending=False)
    buscada = st.selectbox("…o buscá una película de la vista", [None] + opciones["movie_id"].tolist(),
                           format_func=lambda i: "Elegí una película" if i is None else
                           f"{P.loc[P['movie_id'] == i, 'title'].iloc[0]} "
                           f"({P.loc[P['movie_id'] == i, 'release_year'].iloc[0]})")
    if buscada is not None:
        elegido = int(buscada)

    if elegido is not None and (P["movie_id"] == elegido).any():
        f = P.loc[P["movie_id"] == elegido].iloc[0]
        st.subheader(f"{f['title']} ({f['release_year']})")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Presupuesto", S.usd_corto(f["budget"]))
        d2.metric("Recaudación", S.usd_corto(f["revenue"]))
        d3.metric("Recuperó (real)", S.mult(f["multiplo_real"]), help=f"ROI real: {f['roi_real_txt']}")
        d4.metric("Predijo el modelo", S.mult(f["multiplo_pred"]), help=f"ROI predicho: {f['roi_pred_txt']}")
        nota_conj = ("El modelo **nunca la vio**: es una predicción real." if f["conjunto"] == "test" else
                     "Está en el **entrenamiento**: el modelo la vio al aprender, así que la predicción es optimista.")
        st.markdown(f"{f['genero_es']} · {f['idioma_es']} · {int(f['runtime'])} min · "
                    f"{S.MESES[int(f['release_month']) - 1].lower()}, {S.DIAS[int(f['release_day_of_week'])].lower()} · "
                    f"{f['primary_production_company']}  \n"
                    f"Dirige {f['director_name'] or 'sin dato'} · protagoniza {f['lead_actor_name'] or 'sin dato'}"
                    + (f" · {f['collection_name']}" + (" (secuela)" if f['es_secuela'] else " (primera de la saga)")
                       if f['collection_name'] else "") + "  \n"
                    f"**Error:** {f['error_txt']} (tramo real: {f['tramo_roi']}). {nota_conj}  \n"
                    f"Después del estreno (no lo usa el modelo): {S.num(f['vote_average'], 1)} de 10 con "
                    f"{S.num(f['vote_count'])} votos.")

# ---------------------------------------------------------------------------------------------
# 6. Probá el modelo
# ---------------------------------------------------------------------------------------------
with tab_proba:
    st.markdown("Cargá una película que todavía no se estrenó. El modelo usa solo lo que se sabe antes del estreno. "
                "Los menús muestran únicamente lo que el modelo vio al entrenar.")
    num_r, cat_r = R["numericas"], R["categoricas"]
    OTRA = "Otra (el modelo no la conoce)"

    def opciones_cat(col):
        return cat_r[col]["categorias"]

    def conteo_cat(col, valor):
        c = cat_r[col]
        return dict(zip(c["categorias"], c["conteo"])).get(valor, 0)

    with st.form("caso"):
        a1, a2, a3 = st.columns(3)
        budget = a1.number_input("Presupuesto (USD)", min_value=10_000.0, max_value=5_000_000_000.0,
                                 value=float(num_r["budget"]["mediana"]), step=1_000_000.0, format="%.0f",
                                 help=f"El modelo vio de {S.usd(num_r['budget']['min'])} a {S.usd(num_r['budget']['max'])}.")
        runtime = a2.number_input("Duración (minutos)", min_value=1, max_value=600,
                                  value=int(num_r["runtime"]["mediana"]), step=1)
        anio = a3.number_input("Año de estreno", min_value=1900, max_value=2040,
                               value=int(num_r["release_year"]["p99"]), step=1,
                               help=f"El modelo se entrenó con películas de {int(num_r['release_year']['min'])} a "
                                    f"{int(num_r['release_year']['max'])}.")
        b1, b2, b3 = st.columns(3)
        mes = b1.selectbox("Mes de estreno", list(range(1, 13)), index=5, format_func=lambda m: S.MESES[m - 1])
        dia = b2.selectbox("Día de la semana", list(range(7)), index=4, format_func=lambda d: S.DIAS[d])
        idioma = b3.selectbox("Idioma original", opciones_cat("original_language") + [OTRA],
                              format_func=lambda c: c if c == OTRA else S.idioma_es(c))
        g1, g2 = st.columns(2)
        gen1 = g1.selectbox("Género principal", [g for g in opciones_cat("primary_genre")] + [OTRA],
                            format_func=lambda g: g if g == OTRA else S.genero_es(g))
        gen2 = g2.selectbox("Género secundario", opciones_cat("secondary_genre") + [OTRA],
                            format_func=lambda g: g if g == OTRA else S.genero_es(g))
        p1, p2 = st.columns(2)
        prod1 = p1.selectbox("Productora principal", opciones_cat("primary_production_company") + [OTRA],
                             help="Ordenadas por cantidad de películas en el entrenamiento.")
        prod2 = p2.selectbox("Coproductora", opciones_cat("co_production_company") + [OTRA],
                             format_func=lambda c: "Sin coproductora" if c == "Unknown" else c)
        q1, q2, q3 = st.columns(3)
        pop_dir = q1.number_input("Popularidad del director (TMDb)", min_value=0.0, max_value=1000.0,
                                  value=float(num_r["director_popularity"]["mediana"]), step=0.1, format="%.2f")
        pop_lead = q2.number_input("Popularidad del protagonista", min_value=0.0, max_value=1000.0,
                                   value=float(num_r["lead_actor_popularity"]["mediana"]), step=0.1, format="%.2f")
        pop_co = q3.number_input("Popularidad del coprotagonista", min_value=0.0, max_value=1000.0,
                                 value=float(num_r["co_star_popularity"]["mediana"]), step=0.1, format="%.2f")
        st.caption("Popularidad de TMDb: la mediana del entrenamiento es "
                   f"{S.num(num_r['director_popularity']['mediana'], 1)} (director), "
                   f"{S.num(num_r['lead_actor_popularity']['mediana'], 1)} (protagonista) y "
                   f"{S.num(num_r['co_star_popularity']['mediana'], 1)} (coprotagonista).")
        NUEVO = None

        def lista(clave):
            c = cat_r[clave]
            return [NUEVO] + c["ids"], dict(zip(c["ids"], c["nombres"])), dict(zip(c["ids"], c["conteo"]))

        dir_ids, dir_nom, dir_cnt = lista("director_id")
        act_ids, act_nom, act_cnt = lista("lead_actor_id")
        saga_ids, saga_nom, saga_cnt = lista("collection_id")
        h1, h2 = st.columns(2)
        director = h1.selectbox("Director", dir_ids, format_func=lambda i: "Otro o primera película" if i is None
                                else f"{dir_nom[i]} ({dir_cnt[i]} películas)",
                                help="Directores con 2 o más películas en el dataset. El modelo usa cómo les fue antes.")
        actor = h2.selectbox("Actor o actriz principal", act_ids, format_func=lambda i: "Otro o sin historia" if i is None
                             else f"{act_nom[i]} ({act_cnt[i]} películas)")
        s1, s2 = st.columns([1, 2])
        secuela = s1.checkbox("Es una secuela", help="No es la primera película de su saga.")
        saga = s2.selectbox("Saga", saga_ids, format_func=lambda i: "Ninguna u otra" if i is None
                            else f"{saga_nom[i]} ({saga_cnt[i]} películas)",
                            help="Si es secuela de una saga conocida, el modelo usa cómo les fue a las anteriores.")
        GRUPOS_ES = {"kw_libro": "Basada en un libro", "kw_comic": "Cómic o superhéroes",
                     "kw_historia_real": "Historia real o biografía", "kw_remake": "Remake",
                     "kw_videojuego": "Basada en un videojuego", "kw_navidad": "Navidad", "kw_musical": "Musical",
                     "kw_found_footage": "Found footage"}
        t1, t2, t3 = st.columns([2, 1, 1])
        temas = t1.multiselect("De qué se trata", list(GRUPOS_ES), format_func=GRUPOS_ES.get,
                               placeholder="Ninguna de estas")
        eeuu = t2.checkbox("Producida en EE. UU.", value=True)
        n_paises = t3.number_input("Países que la producen", min_value=1, max_value=10, value=1, step=1)
        enviado = st.form_submit_button("Predecir", type="primary")

    # --- Validación: avisos claros si algo está fuera de lo que vio el modelo ---------------
    valores = {"budget": budget, "runtime": runtime, "release_year": anio, "director_popularity": pop_dir,
               "lead_actor_popularity": pop_lead, "co_star_popularity": pop_co}
    nombres_num = {"budget": "El presupuesto", "runtime": "La duración", "release_year": "El año",
                   "director_popularity": "La popularidad del director",
                   "lead_actor_popularity": "La popularidad del protagonista",
                   "co_star_popularity": "La popularidad del coprotagonista"}
    fmt_num = {"budget": S.usd, "release_year": lambda v: str(int(v))}
    avisos, notas = [], []
    for c, v in valores.items():
        r = num_r[c]
        f_ = fmt_num.get(c, lambda x: S.num(x, 1))
        if v < r["min"] or v > r["max"]:
            extra = ""
            if c == "release_year" and v > r["max"]:
                extra = (f" Para años posteriores, los árboles responden como si fuera {int(r['max'])}: así se midió "
                         f"el test {anios_test[0]}–{anios_test[1]}, con un error típico de {S.mult(FIN['factor_test'])}.")
            avisos.append(f"**{nombres_num[c]} ({f_(v)}) está fuera de lo que vio el modelo** "
                          f"(de {f_(r['min'])} a {f_(r['max'])}). El modelo nunca vio valores así: tomá la predicción "
                          f"con mucha cautela.{extra}")
        elif v < r["p01"] or v > r["p99"]:
            notas.append(f"{nombres_num[c]} ({f_(v)}) es poco común: solo el 1% de las películas del entrenamiento "
                         f"está por {'debajo' if v < r['p01'] else 'encima'}.")
    for col, valor, nombre in [("original_language", idioma, "El idioma"), ("primary_genre", gen1, "El género principal"),
                               ("secondary_genre", gen2, "El género secundario"),
                               ("primary_production_company", prod1, "La productora principal"),
                               ("co_production_company", prod2, "La coproductora")]:
        n = conteo_cat(col, valor)
        if valor == OTRA:
            avisos.append(f"**{nombre} es una categoría que el modelo nunca vio.** La trata como «otras» "
                          "(las categorías poco frecuentes).")
        elif n < cat_r[col]["min_frecuencia"]:
            notas.append(f"{nombre} aparece {n} {'vez' if n == 1 else 'veces'} en el entrenamiento: el modelo la "
                         f"agrupa con «otras» (necesita al menos {cat_r[col]['min_frecuencia']}).")

    # --- Predicción: mismas features que el notebook (include/modelo/features.py) ----------
    caso = pd.DataFrame([{
        "movie_id": -1, "title": "Tu película", "budget": budget, "runtime": runtime, "release_year": int(anio),
        "release_month": int(mes), "release_day_of_week": int(dia),
        "original_language": "__otra__" if idioma == OTRA else idioma,
        "primary_genre": "__otra__" if gen1 == OTRA else gen1,
        "secondary_genre": "__otra__" if gen2 == OTRA else gen2,
        "primary_production_company": "__otra__" if prod1 == OTRA else prod1,
        "co_production_company": "__otra__" if prod2 == OTRA else prod2,
        "director_popularity": pop_dir, "lead_actor_popularity": pop_lead, "co_star_popularity": pop_co,
    }])
    palabras = [F.GRUPOS_KEYWORDS[g][0] for g in temas] + (["sequel"] if secuela else [])
    paises = (["US"] if eeuu else []) + [f"P{i}" for i in range(int(n_paises) - (1 if eeuu else 0))]
    # La saga solo cuenta si es secuela: la primera película de una saga no tiene películas anteriores
    caso = caso.assign(director_id=director, lead_actor_id=actor, collection_id=saga if secuela else None,
                       saga_position=1 if secuela else 0, keywords="|".join(palabras),
                       production_countries="|".join(paises))
    caso = F.agregar_features(caso, ref)  # mismas features que el notebook, incluidas las historias
    if director is None or actor is None:
        notas.append("Sin director o actor con historia, el modelo completa ese dato con la mediana "
                     "(así lo trata también cuando entrena).")
    if saga is not None and not secuela:
        notas.append("Elegiste una saga sin marcar «Es una secuela»: si es la primera de la saga, no hay películas "
                     "anteriores, así que el modelo no usa la saga.")
    if secuela and saga is None:
        notas.append("Marcaste secuela sin elegir una saga conocida: el modelo usa que es secuela, pero no tiene "
                     "cómo les fue a las anteriores.")
    y_hat = float(modelo.predict(caso[E.FEATURES])[0])
    multiplo = 10 ** y_hat
    factor = FIN["factor_test"]
    bajo, alto = multiplo / factor, multiplo * factor

    for a in avisos:
        st.warning(a)
    for n_ in notas:
        st.info(n_)

    r1, r2 = st.columns([1.1, 2])
    with r1:
        verbo = "recuperaría" if multiplo >= 1 else "recuperaría solo"
        st.markdown(f"### {verbo.capitalize()} {S.mult(multiplo)} lo invertido")
        roi_hat = multiplo - 1
        st.markdown(f"ROI predicho: **{'+' if roi_hat >= 0 else ''}{S.pct(roi_hat)}**. Con un presupuesto de "
                    f"{S.usd(budget)}, eso es una recaudación de unos **{S.usd(budget * multiplo)}**.")
        st.markdown(f"**Rango típico de error:** entre {S.mult(bajo)} y {S.mult(alto)} "
                    f"(÷ y × {S.mult(factor)}, el error típico del modelo en el test). "
                    + ("Ni el extremo de abajo pierde plata." if bajo >= 1 else
                       "Ni el extremo de arriba recupera lo invertido." if alto < 1 else
                       "El rango cruza ×1: puede ganar o perder plata."))
        st.caption(f"El piso (adivinar la mediana) diría siempre {S.mult(10 ** FIN['mediana_piso_y'])}.")
        CALCULADAS = ["star_power_ratio", "director_leverage", "director_vs_cast", "power_pack",
                      "budget_vs_genre_median", "is_horror_low_budget", "prod_hist_roi", "director_hist",
                      "actor_hist", "saga_hist", "es_secuela"]
        with st.expander("Qué calculó la app con tus datos"):
            st.dataframe(pd.DataFrame({
                "Columna": [VAR_ES.get(c, c) for c in CALCULADAS],
                "Valor": ["sin historia" if pd.isna(caso[c].iloc[0]) else S.num(caso[c].iloc[0], 3)
                          for c in CALCULADAS]}), hide_index=True, width="stretch")
            st.caption("Con la misma función que el notebook (include/modelo/features.py). La mediana por género y las "
                       "historias de la productora, el director, el actor y la saga (mediana de log10 del múltiplo en "
                       "los años anteriores al estreno) salen de las películas con las que se construyó el modelo.")
    with r2:
        # Gráfico: dónde cae tu película en el panorama.
        #  datos: predicciones.parquet (todas, en gris) + tu caso (una fila)
        #  mapeo: x = presupuesto (log), y = múltiplo (log), las mismas escalas que «Explorá»
        #  marcas: círculos grises = múltiplo real de las 6.391 películas; regla violeta = rango típico de error;
        #          estrella violeta = lo que predice el modelo para tu película
        fondo = alt.Chart(P[["budget", "multiplo_real"]]).mark_circle(size=14, color=S.EJE, opacity=0.5).encode(
            x=alt.X("budget:Q", scale=S.escala_budget_log(), axis=S.eje_budget()),
            y=alt.Y("multiplo_real:Q", scale=S.escala_multiplo_log(),
                    axis=S.eje_multiplo("Recaudación / presupuesto (log)")))
        tu = pd.DataFrame({"budget": [budget], "multiplo": [multiplo], "bajo": [bajo], "alto": [alto],
                           "t": [f"Tu película: {S.mult(multiplo)}"], "rango": [f"{S.mult(bajo)} a {S.mult(alto)}"]})
        rango_tu = alt.Chart(tu).mark_rule(color=S.MODELO, strokeWidth=3).encode(
            x="budget:Q", y=alt.Y("bajo:Q", scale=S.escala_multiplo_log()), y2="alto:Q",
            tooltip=[alt.Tooltip("rango:N", title="Rango típico")])
        estrella = alt.Chart(tu).mark_point(shape="diamond", filled=True, size=320, color=S.MODELO,
                                            stroke=S.TINTA, strokeWidth=1.5).encode(
            x="budget:Q", y=alt.Y("multiplo:Q", scale=S.escala_multiplo_log()),
            tooltip=[alt.Tooltip("t:N", title=" "), alt.Tooltip("rango:N", title="Rango típico")])
        texto_tu = alt.Chart(tu).mark_text(align="left", dx=14, fontSize=13, fontWeight="bold", color=S.MODELO).encode(
            x="budget:Q", y=alt.Y("multiplo:Q", scale=S.escala_multiplo_log()), text="t:N")
        linea1 = alt.Chart(pd.DataFrame({"y": [1]})).mark_rule(strokeDash=[4, 4], color=S.PISO).encode(
            y=alt.Y("y:Q", scale=S.escala_multiplo_log()))
        mostrar(alt.layer(fondo, linea1, rango_tu, estrella, texto_tu), alto=420)
        st.caption("Los puntos grises son lo que recuperaron de verdad las películas del dataset. El rombo violeta es "
                   "lo que el modelo predice para la tuya; la línea, su rango típico de error.")

st.divider()
st.caption("Proyecto integrador de Ciencia de Datos · UTN FRM 2026 · Datos: TMDb (este producto usa la API de TMDb "
           "pero no está avalado ni certificado por TMDb). Los números salen de `python build_app.py`.")
