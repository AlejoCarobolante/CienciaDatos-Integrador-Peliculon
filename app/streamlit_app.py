"""Peliculón · ¿Cuántas veces recupera una película lo que costó?

App del proyecto integrador (Ciencia de Datos, UTN FRM 2026), contada como una historia (copa de martini):
  Portada -> Tallo (1 Los datos, 2 Lo que encontramos, 3 El modelo, 4 Cómo lo mejoramos) -> Copa (Explorá, Probá).
Cada sección del tallo: una frase que afirma algo, los gráficos que la prueban (visibles) y un «Cómo lo sabemos»
cerrado con el detalle técnico. Al final de cada sección, la pregunta que responde la siguiente.

Todo número que se muestra sale de app/artefactos/, que genera `python build_app.py`.
Todos los gráficos son Altair; colores y escalas vienen de app/estilo.py. Arriba de cada gráfico hay un
comentario con sus componentes (datos, transformaciones, marca, mapeo, escalas).
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

st.set_page_config(page_title="Peliculón · ¿Cuánto recupera una película?", page_icon="🎬", layout="wide")
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
    p["escala"] = pd.cut(p["budget"], [0, 1e6, 1e7, 1e8, np.inf], right=False,
                         labels=["Menos de 1 M", "1 a 10 M", "10 a 100 M", "Más de 100 M"]).astype(str)
    return p


modelo, ref = cargar_modelo()
M = cargar_metricas()
R = cargar_rangos()
P = cargar_predicciones()
FIN = M["final"]
CTX = M["contexto"]
E3 = M["entrega3_presentada"]

if sklearn.__version__ != M["entorno"]["scikit_learn"]:
    st.warning(f"El modelo se guardó con scikit-learn {M['entorno']['scikit_learn']} y esta app corre "
               f"{sklearn.__version__}. Las predicciones pueden no coincidir con el notebook.")


# =============================================================================================
# Piezas de página
# =============================================================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;750;850&display=swap');
html, body, .stMarkdown p, .stMarkdown li, h1, h2, h3 {{font-family: Inter, system-ui, sans-serif;}}
.block-container {{max-width: 1200px; padding-top: 2rem;}}
div[data-testid="stMarkdownContainer"] p.kicker, .kicker {{font-size: 1.35rem; font-weight: 800; letter-spacing: .1em;
    text-transform: uppercase; color: {S.DORADO}; margin: 0 0 .5rem 0;}}
div[data-testid="stMarkdownContainer"] p.titulo-seccion, .titulo-seccion {{font-size: 2.15rem; line-height: 1.15;
    font-weight: 850; color: {S.TINTA}; margin: 0 0 .7rem 0; letter-spacing: -.01em;}}
div[data-testid="stMarkdownContainer"] p.bajada, .bajada {{font-size: 1.1rem; color: {S.TINTA_2}; margin-bottom: 1.3rem;
    max-width: 62rem; line-height: 1.6;}}
div[data-testid="stMarkdownContainer"] p.grafico-titulo, .grafico-titulo {{font-size: 1.12rem; font-weight: 750;
    color: {S.TINTA}; margin: 1.6rem 0 .15rem 0;}}
div[data-testid="stMarkdownContainer"] p.grafico-bajada, .grafico-bajada {{font-size: .93rem; color: {S.TINTA_2};
    margin: 0 0 .4rem 0;}}
.hero {{position: relative; border-radius: 22px; padding: 2.4rem 2.4rem 1.6rem 2.4rem; margin-bottom: .6rem;
    background: radial-gradient(1200px 400px at 85% -10%, rgba(165,148,255,.28), transparent 60%),
                radial-gradient(800px 300px at -10% 110%, rgba(255,200,87,.16), transparent 60%), {S.PANEL};
    border: 1px solid {S.GRILLA};}}
div[data-testid="stMarkdownContainer"] .hero p.kicker {{font-size: .9rem; letter-spacing: .16em; margin-bottom: .6rem;}}
section[data-testid="stSidebar"] {{border-right: 1px solid {S.GRILLA};}}
section[data-testid="stSidebar"] h3 {{font-size: .8rem; font-weight: 750; letter-spacing: .16em; text-transform: uppercase;
    color: {S.DORADO}; padding: .4rem 0 .2rem .3rem;}}
section[data-testid="stSidebar"] ul {{list-style: none; padding: 0; margin: 0;}}
section[data-testid="stSidebar"] li {{margin: 0 0 .15rem 0;}}
section[data-testid="stSidebar"] li a {{display: block; padding: .5rem .8rem; border-radius: 10px; font-size: 1rem;
    font-weight: 600; color: {S.TINTA_2}; text-decoration: none; border-left: 3px solid transparent;
    transition: background .15s, color .15s, border-color .15s;}}
section[data-testid="stSidebar"] li a:hover {{background: {S.PANEL}; color: {S.TINTA}; border-left-color: {S.DORADO};}}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{padding: 0 .3rem; line-height: 1.5;}}
.hero h1 {{font-size: 3.1rem; line-height: 1.06; font-weight: 850; letter-spacing: -.02em; color: {S.TINTA}; margin: 0 0 1rem 0;
    padding: 0;}}
.hero h1 span {{background: linear-gradient(90deg, {S.MODELO}, {S.DORADO}); -webkit-background-clip: text;
    background-clip: text; color: transparent;}}
.hero p {{font-size: 1.15rem; color: {S.TINTA_2}; max-width: 54rem; line-height: 1.6; margin: 0;}}
.hero b {{color: {S.TINTA};}}
.pase {{border: 1px solid {S.GRILLA}; border-left: 4px solid {S.DORADO}; background: {S.PANEL}; padding: 1rem 1.2rem;
    border-radius: 0 14px 14px 0; margin: 2rem 0 .6rem 0; color: {S.TINTA}; font-size: 1.06rem;}}
.pase b {{color: {S.DORADO};}}
.tarjetas {{display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; margin: 1.2rem 0 .4rem 0;}}
.tarjeta {{background: {S.PANEL}; border: 1px solid {S.GRILLA}; border-radius: 16px; padding: 1.15rem 1.25rem;}}
.tarjeta .rotulo {{font-size: .82rem; color: {S.TINTA_SUAVE}; text-transform: uppercase; letter-spacing: .1em; font-weight: 600;}}
.tarjeta .valor {{font-size: 2.6rem; font-weight: 850; color: {S.TINTA}; line-height: 1.1; margin: .25rem 0 .35rem 0;
    letter-spacing: -.02em;}}
.tarjeta .valor.modelo {{color: {S.MODELO}; text-shadow: 0 0 28px rgba(165,148,255,.45);}}
.tarjeta .valor.texto {{font-size: 2.1rem;}}
.tarjeta .valor.oro {{color: {S.DORADO}; text-shadow: 0 0 28px rgba(255,200,87,.35);}}
.tarjeta .nota {{font-size: .9rem; color: {S.TINTA_2}; line-height: 1.5;}}
.flujo {{display: flex; gap: .45rem; align-items: stretch; margin: .7rem 0 .2rem 0;}}
.paso {{flex: 1 1 0; min-width: 0; background: {S.PANEL}; border: 1px solid {S.GRILLA}; border-radius: 12px; padding: .8rem .9rem;}}
.paso .n {{font-size: .72rem; font-weight: 750; color: {S.DORADO}; letter-spacing: .12em;}}
.paso .t {{font-weight: 750; color: {S.TINTA}; font-size: 1rem; margin: .1rem 0;}}
.paso .d {{font-size: .85rem; color: {S.TINTA_2}; line-height: 1.45;}}
.flecha {{align-self: center; color: {S.TINTA_SUAVE}; font-size: 1.3rem;}}
.capitulos {{display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .8rem; margin: .5rem 0 0 0;}}
.capitulo {{background: {S.PANEL}; border: 1px solid {S.GRILLA}; border-radius: 12px; padding: .8rem .9rem;}}
.capitulo a {{text-decoration: none; color: {S.TINTA}; font-weight: 750;}}
.capitulo .num {{font-size: 1.6rem; font-weight: 850; color: {S.MODELO}; line-height: 1;}}
.capitulo .q {{font-size: .88rem; color: {S.TINTA_2}; margin-top: .25rem;}}
div[data-testid="stExpander"] details {{background: {S.PANEL}; border-color: {S.GRILLA};}}
div[data-testid="stHorizontalBlock"]:has(div.pb-pegajoso) {{align-items: flex-start;}}
div[data-testid="stColumn"]:has(div.pb-pegajoso) {{position: sticky; top: 3.2rem; align-self: flex-start;}}
@media (max-width: 760px) {{
  .tarjetas, .capitulos {{grid-template-columns: 1fr;}} .flujo {{flex-direction: column;}} .flecha {{display: none;}}
  .hero {{padding: 1.4rem;}} .hero h1 {{font-size: 2rem;}}
  div[data-testid="stMarkdownContainer"] p.titulo-seccion {{font-size: 1.55rem;}}
}}
</style>
""", unsafe_allow_html=True)


def seccion(ancla: str, kicker: str, titulo: str, bajada: str | None = None) -> None:
    """Encabezado visible de cada sección: el nombre (kicker), la frase que afirma y una bajada."""
    st.markdown(f"<div id='{ancla}'></div><p class='kicker'>{kicker}</p><p class='titulo-seccion'>{titulo}</p>"
                + (f"<p class='bajada'>{bajada}</p>" if bajada else ""), unsafe_allow_html=True)


def titulo_grafico(titulo: str, bajada: str | None = None) -> None:
    st.markdown(f"<p class='grafico-titulo'>{titulo}</p>" + (f"<p class='grafico-bajada'>{bajada}</p>" if bajada else ""),
                unsafe_allow_html=True)


def pase(texto: str) -> None:
    """Transición: la pregunta que deja una sección y responde la siguiente."""
    st.markdown(f"<div class='pase'>{texto}</div>", unsafe_allow_html=True)


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
    "power_pack": "Director × protagonista (popularidad)", "release_month": "Mes de estreno",
    "release_day_of_week": "Día de estreno", "is_horror_low_budget": "Terror de bajo presupuesto",
    "director_hist": "Historia del director", "director_n_prev": "Películas previas del director",
    "actor_hist": "Historia del protagonista", "actor_n_prev": "Películas previas del protagonista",
    "saga_hist": "Historia de la saga", "es_secuela": "Es secuela", "kw_secuela": "Palabra clave: secuela",
    "kw_remake": "Palabra clave: remake", "kw_libro": "Basada en un libro", "kw_comic": "Cómic o superhéroes",
    "kw_historia_real": "Historia real o biografía", "kw_videojuego": "Basada en un videojuego",
    "kw_navidad": "Navidad", "kw_musical": "Musical", "kw_found_footage": "Found footage",
    "paises_n": "Cantidad de países", "es_eeuu": "Producida en EE. UU.",
}
COLS_NEGOCIO = ["power_pack", "star_power_ratio", "director_leverage", "director_vs_cast", "budget_vs_genre_median",
                "is_horror_low_budget", "prod_hist_roi"]


def origen(col: str) -> str:
    if col in E.COLS_CTX:
        return "Nuevas de TMDb (2ª ronda)"
    if col in COLS_NEGOCIO:
        return "Combinadas (1ª ronda)"
    return "Originales"


ESCALA_ORIGEN = alt.Scale(domain=["Originales", "Combinadas (1ª ronda)", "Nuevas de TMDb (2ª ronda)"],
                          range=[S.GRIS_3, S.GRIS_1, S.MODELO])

# Resultados de una película, de peor a mejor (ordinal): gris = perdió plata, verdes = cuánto multiplicó
TRAMOS = list(E.TRAMOS_ROI[1])
TRAMOS_ES = {"Pérdida (ROI<0)": "Perdió plata", "Recupera hasta 2x": "Recuperó hasta ×2", "2x a 4x": "×2 a ×4",
             "4x a 11x": "×4 a ×11", "Más de 11x": "Más de ×11"}
ESCALA_RESULTADO = alt.Scale(domain=[TRAMOS_ES[t] for t in TRAMOS],
                             range=["#6b6878", "#24493b", "#2f7a57", "#47b47d", "#82f2ad"])

# Números que se repiten
embudo = pd.DataFrame(M["embudo"])
n_bajadas, n_alcance = int(embudo["filas"].iloc[0]), int(embudo["filas"].iloc[-1])
n_pedidas = int(M["pedidas"]["total"])
imp = pd.DataFrame(M["importancia"])
peso_budget = imp.loc[0, "aumento_mae"] / imp.loc[1, "aumento_mae"]
modelos = pd.DataFrame(M["modelos"]).set_index("modelo")
anios_test = M["particion"]["anios_test"]
tramos = pd.DataFrame(M["por_tramo"])
fac = lambda mae: 10 ** mae  # noqa: E731
mejora_ctx_pct = (CTX["sin"]["mae_test"] - CTX["con"]["mae_test"]) / CTX["sin"]["mae_test"]

with st.sidebar:
    st.markdown("### Recorrido")
    st.markdown("- [El proyecto](#portada)\n- [1 · Los datos](#datos)\n- [2 · Lo que encontramos](#hallazgo)\n"
                "- [3 · El modelo](#modelo)\n- [4 · Cómo lo mejoramos](#mejora)\n- [Explorá y probá](#copa)")
    st.caption(f"Datos de TMDb del {pd.Timestamp(M['datos']['fecha']).strftime('%d/%m/%Y')} · "
               f"scikit-learn {M['entorno']['scikit_learn']} · semilla {M['entorno']['seed']}")
    st.caption("[Repositorio en GitHub](https://github.com/AlejoCarobolante/CienciaDatos-Integrador-Peliculon/tree/mejoras)")


# =============================================================================================
# PORTADA
# =============================================================================================
st.markdown(f"""
<div id='portada'></div>
<div class='hero'>
  <p class='kicker'>Peliculón · Proyecto integrador de Ciencia de Datos · UTN FRM 2026</p>
  <h1>¿Se puede saber, antes del estreno, <span>cuántas veces va a recuperar</span> una película lo que costó?</h1>
  <p>Juntamos {S.num(n_alcance)} películas de TMDb y entrenamos un modelo con lo que se sabe antes del estreno. Se equivoca
  un <b>{S.pct(FIN['mejora_test_pct'], 1)} menos</b> que adivinar: no da la cifra exacta, pero separa un proyecto riesgoso
  de uno seguro. Y lo que más pesa, por lejos, es el <b>presupuesto</b>.</p>
</div>
""", unsafe_allow_html=True)

# Galaxia: todas las películas del alcance, con las famosas marcadas.
P["resultado"] = P["tramo_roi"].map(TRAMOS_ES)
FAMOSAS = ["Paranormal Activity", "The Blair Witch Project", "Rocky", "Star Wars", "Titanic", "Get Out", "Cats",
           "The Irishman", "Halloween"]
fam = P[P["title"].isin(FAMOSAS)].drop_duplicates("title").copy()
fam["etiqueta"] = fam["title"] + "  " + fam["multiplo_real"].map(lambda v: S.mult(v, 2 if v < 1 else (1 if v < 100 else 0)))
fam["lado"] = np.where(fam["budget"] < 3e7, "izq", "der")
fam.loc[fam["title"] == "Get Out", "lado"] = "der"  # para no pisar a Star Wars
titulo_grafico(f"Las {S.num(n_alcance)} películas, de un vistazo",
               "Cada punto es una película: a la derecha, más cara; arriba, más veces recuperó lo que costó (las dos "
               "escalas son logarítmicas). Debajo de la línea punteada perdieron plata. Tocá un color de la leyenda para "
               "resaltarlo.")
# Gráfico: la galaxia de películas.
#  datos: predicciones.parquet (las 6.391 del alcance) + las famosas (subconjunto)
#  mapeo: x = presupuesto (LOG), y = múltiplo real (LOG), color = resultado (ordinal: de perder plata a más de ×11)
#  marcas: círculos chicos semitransparentes; anillo y texto dorados para las famosas; regla punteada en ×1
#  interacción: clic en la leyenda resalta un resultado (selección ligada a la leyenda, cambia la opacidad)
sel_res = alt.selection_point(fields=["resultado"], bind="legend")
x_gal = alt.X("budget:Q", scale=S.escala_budget_log(), axis=S.eje_budget())
y_gal = alt.Y("multiplo_real:Q", scale=S.escala_multiplo_log(),
              axis=S.eje_multiplo("Cuántas veces recuperó lo que costó (log)"))
galaxia = alt.Chart(P).mark_circle(size=16).encode(
    x=x_gal, y=y_gal,
    color=alt.Color("resultado:N", scale=ESCALA_RESULTADO, sort=[TRAMOS_ES[t] for t in TRAMOS],
                    legend=alt.Legend(title=None, orient="top", symbolSize=140, labelFontSize=12.5)),
    opacity=alt.condition(sel_res, alt.value(0.75), alt.value(0.06)),
    tooltip=[alt.Tooltip("title:N", title="Película"), alt.Tooltip("release_year:Q", title="Año"),
             alt.Tooltip("presupuesto_txt:N", title="Costó"), alt.Tooltip("recaudacion_txt:N", title="Recaudó"),
             alt.Tooltip("multiplo_real_txt:N", title="Recuperó")]).add_params(sel_res)
anillos = alt.Chart(fam).mark_point(size=150, color=S.DORADO, strokeWidth=2, filled=False).encode(x=x_gal, y=y_gal)
textos_fam = [alt.Chart(fam[fam["lado"] == lado]).mark_text(
    align="left" if lado == "izq" else "right", dx=10 if lado == "izq" else -10, fontSize=12, fontWeight=600,
    color=S.DORADO).encode(x=x_gal, y=y_gal, text="etiqueta:N") for lado in ("izq", "der")]
uno = alt.Chart(pd.DataFrame({"y": [1], "x": [1.2e4], "t": ["×1: recupera justo lo que costó"]}))
mostrar(alt.layer(galaxia, uno.mark_rule(strokeDash=[5, 4], color=S.TINTA_SUAVE).encode(y=alt.Y("y:Q", scale=S.escala_multiplo_log())),
                  uno.mark_text(align="left", dy=-8, color=S.TINTA_SUAVE, fontSize=11.5).encode(
                      x=alt.X("x:Q", scale=S.escala_budget_log()), y=alt.Y("y:Q", scale=S.escala_multiplo_log()), text="t:N"),
                  anillos, *textos_fam), alto=560)
st.markdown(f"Arriba a la izquierda están los milagros de bajo presupuesto: *Paranormal Activity* costó 15 mil dólares y "
            f"recaudó casi 13 mil veces eso. Abajo a la derecha, los golpes caros: *The Irishman* figura con una recaudación "
            f"simbólica porque se estrenó en Netflix. La nube del medio es la mayoría: películas que recuperan entre ×1 y ×10.")

# Ejemplos: tres destinos (una que perdió, una típica, un fenómeno). Si el dataset cambia y no están,
# se eligen solas entre las más votadas.
def ejemplo(titulo: str, cond) -> pd.Series:
    f = P[P["title"] == titulo]
    if len(f):
        return f.iloc[0]
    return P[cond(P)].sort_values("vote_count", ascending=False).iloc[0]


ej = pd.DataFrame([
    ejemplo("Gattaca", lambda d: d["multiplo_real"].between(0.2, 0.6)),
    ejemplo("Batman Begins", lambda d: d["multiplo_real"].between(2.3, 2.7)),
    ejemplo("Get Out", lambda d: d["multiplo_real"] > 40),
])
ej["nombre"] = ej["title"] + " (" + ej["release_year"].astype(int).astype(str) + ")"
ej["destino"] = [f"{S.mult(m, 2)}: perdió el {S.pct(1 - m)} de lo invertido" if m < 1 else
                 f"{S.mult(m, 1)}: recuperó {S.num(m, 1)} veces lo que costó" for m in ej["multiplo_real"]]
ej["costo_txt"], ej["recaudo_txt"] = ej["budget"].map(S.usd), ej["revenue"].map(S.usd)

titulo_grafico("Cómo se lee: cuántas veces vuelve lo que se puso",
               "El punto hueco es lo que costó; el dorado, lo que recaudó en cines. La distancia entre los dos es el "
               "<b>múltiplo</b> (recaudación ÷ presupuesto), lo que el modelo intenta anticipar.")
# Gráfico: tres películas, tres destinos.
#  datos: tres filas de predicciones.parquet (presupuesto y recaudación reales)
#  mapeo: y = película (nominal), x = dólares (escala LOG, para que 4 millones y 400 millones entren juntos)
#  marcas: regla de lo que costó a lo que recaudó; círculo hueco = costó, círculo lleno = recaudó; texto = múltiplo
esc_usd = alt.Scale(type="log", domain=[1e6, 2.5e10], nice=False)
eje_usd = alt.Axis(title="Dólares (escala log)", values=[1e6, 1e7, 1e8, 1e9], labelExpr=S.ETIQUETA_USD)
y_ej = alt.Y("nombre:N", sort=list(ej["nombre"]), title=None, axis=alt.Axis(labelFontSize=13, labelLimit=260))
base_ej = alt.Chart(ej)
tip_ej = [alt.Tooltip("nombre:N", title="Película"), alt.Tooltip("costo_txt:N", title="Costó"),
          alt.Tooltip("recaudo_txt:N", title="Recaudó"), alt.Tooltip("destino:N", title="Resultado")]
g_ej = alt.layer(
    base_ej.mark_rule(color=S.EJE, strokeWidth=4).encode(x=alt.X("budget:Q", scale=esc_usd, axis=eje_usd),
                                                         x2="revenue:Q", y=y_ej, tooltip=tip_ej),
    base_ej.mark_point(size=260, filled=True, fill=S.SUPERFICIE, stroke=S.TINTA_2, strokeWidth=2.5).encode(
        x=alt.X("budget:Q", scale=esc_usd), y=y_ej, tooltip=tip_ej),
    base_ej.mark_circle(size=260, color=S.DORADO, opacity=1).encode(
        x=alt.X("revenue:Q", scale=esc_usd), y=y_ej, tooltip=tip_ej),
    base_ej.mark_text(align="left", dx=16, fontSize=13, fontWeight="bold", color=S.TINTA).encode(
        x=alt.X("max_v:Q", scale=esc_usd), y=y_ej, text="destino:N"),
).transform_calculate(max_v="max(datum.budget, datum.revenue)")
mostrar(g_ej, alto=190)

st.markdown(f"""
<div class='tarjetas'>
  <div class='tarjeta'><div class='rotulo'>Error típico del modelo</div><div class='valor modelo'>{S.mult(FIN['factor_test'])}</div>
    <div class='nota'>Arriba o abajo de lo real, en {S.num(M['particion']['n_test'])} películas de {anios_test[0]}–{anios_test[1]}
    que el modelo nunca vio. Si una recuperó ×2, suele decir algo entre {S.mult(2 / FIN['factor_test'], 1)} y
    {S.mult(2 * FIN['factor_test'], 1)}.</div></div>
  <div class='tarjeta'><div class='rotulo'>Menos error que adivinar</div><div class='valor oro'>{S.pct(FIN['mejora_test_pct'], 1)}</div>
    <div class='nota'>Adivinar siempre lo más común se equivoca {S.mult(FIN['factor_piso_test'])}. En la Entrega 3,
    el modelo mejoraba un {S.pct(E3['mejora_test_pct'], 0)}.</div></div>
  <div class='tarjeta'><div class='rotulo'>Lo que más pesa</div><div class='valor texto'>Presupuesto</div>
    <div class='nota'>Pesa {S.num(peso_budget, 1)} veces más que la segunda columna más importante del modelo.</div></div>
</div>
""", unsafe_allow_html=True)

titulo_grafico("Cómo está armado el proyecto")
st.markdown(f"""
<div class='flujo'>
  <div class='paso'><div class='n'>ENTREGA 1</div><div class='t'>Datos de TMDb</div>
    <div class='d'>Un DAG de Airflow pide {S.num(n_pedidas)} películas a la API pública de TMDb.</div></div>
  <div class='flecha'>→</div>
  <div class='paso'><div class='n'>ENTREGA 1</div><div class='t'>Limpieza</div>
    <div class='d'>Quedan {S.num(n_alcance)} con presupuesto y recaudación creíbles.</div></div>
  <div class='flecha'>→</div>
  <div class='paso'><div class='n'>ENTREGA 2</div><div class='t'>Análisis</div>
    <div class='d'>Hipótesis sobre qué hace que una película recupere más.</div></div>
  <div class='flecha'>→</div>
  <div class='paso'><div class='n'>ENTREGA 3</div><div class='t'>Modelo</div>
    <div class='d'>Promedio de tres modelos, probado con películas que nunca vio.</div></div>
  <div class='flecha'>→</div>
  <div class='paso'><div class='n'>ENTREGA 4</div><div class='t'>Esta app</div>
    <div class='d'>Cuenta lo que encontramos y deja probar el modelo.</div></div>
</div>
""", unsafe_allow_html=True)

titulo_grafico("El recorrido")
st.markdown("""
<div class='capitulos'>
  <div class='capitulo'><div class='num'>1</div><a href='#datos'>Los datos</a><div class='q'>¿De dónde salen las películas y cuántas sirven?</div></div>
  <div class='capitulo'><div class='num'>2</div><a href='#hallazgo'>Lo que encontramos</a><div class='q'>¿Les va mejor a las caras o a las baratas?</div></div>
  <div class='capitulo'><div class='num'>3</div><a href='#modelo'>El modelo</a><div class='q'>¿Se puede anticipar? ¿Dónde acierta y dónde falla?</div></div>
  <div class='capitulo'><div class='num'>4</div><a href='#mejora'>Cómo lo mejoramos</a><div class='q'>¿Qué cambió desde la Entrega 3?</div></div>
</div>
""", unsafe_allow_html=True)
st.caption("Al final, en «Explorá y probá», podés recorrer todas las películas y cargar una inventada.")

st.divider()

# =============================================================================================
# 1 · LOS DATOS (Entrega 1)
# =============================================================================================
seccion("datos", "1 · Los datos",
        f"De {S.num(n_pedidas)} películas que le pedimos a TMDb, {S.num(n_alcance)} tienen presupuesto y recaudación "
        "creíbles",
        "Para medir cuánto recupera una película hacen falta dos números: lo que costó y lo que recaudó. "
        "A una de cada tres le falta alguno, así que el camino de los datos es, sobre todo, un embudo.")

# Embudo desde lo pedido: la primera barra está partida en las tres listas que pidió el DAG.
pedidas = pd.DataFrame(M["pedidas"]["listas"])
pasos_emb = [{"paso": "Pedidas a TMDb", "filas": n_pedidas, "sale": 0,
              "motivo": "Tres listas para no quedarnos solo con éxitos (todas con al menos 100 votos)."},
             {"paso": "Llegaron", "filas": n_bajadas, "sale": n_pedidas - n_bajadas,
              "motivo": "Las que faltan no se pudieron bajar por cortes de conexión con la API."}]
nombres_paso = {"Con budget y revenue > 0": "Con presupuesto y recaudación",
                "Con budget >= 10.000 USD": "Presupuesto de 10 mil USD o más",
                "Con revenue >= 10.000 USD": "Recaudación de 10 mil USD o más",
                "Estrenadas antes de 2026": "Estrenadas antes de 2026"}
for i in range(1, len(embudo)):
    pasos_emb.append({"paso": nombres_paso.get(embudo.loc[i, "paso"], embudo.loc[i, "paso"]),
                      "filas": int(embudo.loc[i, "filas"]),
                      "sale": int(embudo.loc[i - 1, "filas"] - embudo.loc[i, "filas"]),
                      "motivo": embudo.loc[i, "motivo"]})
emb = pd.DataFrame(pasos_emb)
emb["x"], emb["x2"] = -emb["filas"] / 2, emb["filas"] / 2
emb["quedan_txt"] = [f"{S.num(f)}" + (f"  (−{S.num(s)})" if s else "") for f, s in zip(emb["filas"], emb["sale"])]
emb["pct_txt"] = [S.pct(f / n_pedidas) for f in emb["filas"]]
segs, x0 = [], -n_pedidas / 2
for _, fila in pedidas.iterrows():
    segs.append({"paso": "Pedidas a TMDb", "lista": fila["lista"], "x": x0, "x2": x0 + fila["peliculas"],
                 "t": f"{fila['lista']} {S.num(fila['peliculas'])}"})
    x0 += fila["peliculas"]
segs = pd.DataFrame(segs)
orden_pasos = list(emb["paso"])
mayor = emb.iloc[2:]["sale"].idxmax()

titulo_grafico("El embudo: de lo que pedimos a lo que usamos",
               "Cada barra es un paso y su ancho, cuántas películas quedan. Entre paréntesis, cuántas saca ese paso.")
# Gráfico: embudo centrado.
#  datos: metricas.json["pedidas"] y ["embudo"] (filas que quedan después de cada paso)
#  transformación: cada barra va de −filas/2 a +filas/2 (centrada), así el ancho se lee como un embudo
#  marcas: barra (rect) por paso; la primera, partida en las tres listas del muestreo; texto con lo que queda
#  mapeo: y = paso (ordinal, en el orden del pipeline), x/x2 = extremos de la barra, color = lista (solo en la 1ª)
y_paso = alt.Y("paso:N", sort=orden_pasos, title=None, axis=alt.Axis(labelFontSize=13, labelLimit=280, ticks=False,
                                                                     domain=False))
x_emb = alt.Scale(domain=[-n_pedidas / 2 - 200, n_pedidas / 2 + 5200], nice=False)
barras_emb = alt.Chart(emb.iloc[1:]).mark_bar(color=S.GRIS_3, height=30, cornerRadius=3).encode(
    x=alt.X("x:Q", scale=x_emb, axis=None), x2="x2:Q", y=y_paso,
    tooltip=[alt.Tooltip("paso:N", title="Paso"), alt.Tooltip("filas:Q", title="Quedan", format=","),
             alt.Tooltip("sale:Q", title="Saca este paso", format=","), alt.Tooltip("motivo:N", title="Por qué")])
barras_pedidas = alt.Chart(segs).mark_bar(height=30, stroke=S.SUPERFICIE, strokeWidth=2).encode(
    x=alt.X("x:Q", scale=x_emb), x2="x2:Q", y=y_paso,
    color=alt.Color("lista:N", scale=alt.Scale(domain=list(pedidas["lista"]), range=[S.GRIS_3, S.GRIS_2, S.GRIS_1]),
                    legend=None),
    tooltip=[alt.Tooltip("t:N", title="Lista")])
texto_listas = alt.Chart(segs).mark_text(color=S.SUPERFICIE, fontSize=10.5, fontWeight="bold").encode(
    x=alt.X("medio:Q", scale=x_emb), y=y_paso, text="t:N").transform_calculate(medio="(datum.x + datum.x2) / 2")
texto_emb = alt.Chart(emb).mark_text(align="left", dx=10, fontSize=13, color=S.TINTA).encode(
    x=alt.X("x2:Q", scale=x_emb), y=y_paso, text="quedan_txt:N")
nota_emb = alt.Chart(emb.loc[[mayor]]).mark_text(align="left", dx=10, dy=15, fontSize=11.5, color=S.TINTA_2).encode(
    x=alt.X("x2:Q", scale=x_emb), y=y_paso, text=alt.value("sin presupuesto o sin recaudación"))
mostrar(alt.layer(barras_emb, barras_pedidas, texto_listas, texto_emb, nota_emb), alto=330)
st.markdown(
    f"La pérdida grande es una sola: **{S.num(emb.loc[mayor, 'sale'])} películas** no tienen presupuesto o recaudación "
    f"en TMDb, y sin esos dos números no hay nada que medir. No se inventan: completar un presupuesto sería inventar "
    f"la respuesta. Los otros filtros sacan poco: montos imposibles (presupuestos de 1 o 7 dólares) y las de 2026, "
    f"que siguen en cines. Quedan **{S.num(n_alcance)}** ({S.pct(n_alcance / n_pedidas)} de lo pedido).")

anios = P[P["release_year"] >= 1940].groupby(["release_year", "conjunto"]).size().reset_index(name="peliculas")
anios["conjunto_es"] = anios["conjunto"].map({"entrenamiento": "Para aprender (hasta 2017)",
                                              "test": f"Examen final ({anios_test[0]}–{anios_test[1]})"})
titulo_grafico("De qué época son",
               "La mayoría es de 1990 en adelante. Las más nuevas se guardan aparte desde el principio: son el examen "
               "final del modelo (sección 3).")
# Gráfico: películas por año de estreno.
#  datos: predicciones.parquet agrupado por año y conjunto (entrenamiento / test) con pandas
#  marca: barra apilada por año; mapeo: x = año (temporal discreto), y = cantidad, color = conjunto
#  escala de color: gris = para aprender, dorado = examen final (el modelo nunca lo ve)
escala_conj = alt.Scale(domain=list(anios["conjunto_es"].unique()), range=[S.GRIS_1, S.DORADO])
mostrar(alt.Chart(anios).mark_bar(width=4).encode(
    x=alt.X("release_year:Q", title="Año de estreno", axis=alt.Axis(format="d", tickCount=12),
            scale=alt.Scale(domain=[1940, 2026], nice=False)),
    y=alt.Y("sum(peliculas):Q", title="Películas por año"),
    color=alt.Color("conjunto_es:N", scale=escala_conj, legend=alt.Legend(title=None)),
    tooltip=[alt.Tooltip("release_year:Q", title="Año"), alt.Tooltip("conjunto_es:N", title=" "),
             alt.Tooltip("sum(peliculas):Q", title="Películas")]), alto=230)
st.caption(f"Antes de 1940 hay {S.num(int((P['release_year'] < 1940).sum()))} películas (no se ven en el gráfico).")

with como_lo_sabemos():
    st.markdown(f"""
**De dónde salen.** De la API oficial de [TMDb](https://www.themoviedb.org/) (The Movie Database), con un DAG de
Airflow (`dags/tmdb_ingest.py`, corrido con Astronomer) en modo `full`:

1. **Muestreo estratificado** (`discover_batches`): para no quedarnos solo con éxitos, se piden tres listas
   distintas, todas con al menos 100 votos: 4.000 películas **más populares**, 3.000 **peor calificadas** y
   3.000 de **mayor taquilla**. Los IDs repetidos se cuentan una vez.
2. **Capa bronce** (`land_bronze`): el JSON crudo de cada película, con créditos y palabras clave, en gzip.
3. **Capa plata** (`refine_silver`, `consolidate`, `validate`): una fila tipada por película, sin duplicados y con
   controles de esquema (columnas, nulos, rango de votos).
4. **Sagas** (`enrich_sagas`): pide cada saga una vez y ordena sus películas por fecha (cuál es la primera y cuáles,
   secuelas).
5. **Respaldo**: si la API no responde, el DAG usa la última corrida buena (`include/frozen/ultimo_ok.csv`).

**Fecha de los datos:** corrida del {pd.Timestamp(M['datos']['fecha']).strftime('%d/%m/%Y')}
(`{M['datos']['csv']}`): {S.num(M['datos']['filas'])} películas × {M['datos']['columnas']} columnas. Las columnas de
director, actor, saga, palabras clave y países se agregaron el 10/11 a esa misma corrida
(`include/tmdb/enriquecer.py`), sin cambiar las películas.

**Qué es una fila:** una película (`movie_id`), con su presupuesto y recaudación en USD, fecha de estreno (año, mes y
día de la semana), duración, idioma, dos géneros, dos productoras, la popularidad del director y de los dos primeros
actores, quiénes son el director y el protagonista, su saga y lugar en ella, palabras clave, países, y el promedio y la
cantidad de votos.

**El filtrado, paso a paso:**
""")
    st.dataframe(emb[["paso", "filas", "sale", "motivo"]].rename(
        columns={"paso": "Paso", "filas": "Quedan", "sale": "Saca", "motivo": "Por qué"}),
        hide_index=True, width="stretch")
    st.markdown("""
**Columnas descartadas por fuga** (se conocen recién después del estreno):
- `revenue`: es parte de la respuesta (múltiplo = revenue / budget). Usarla sería darle la respuesta al modelo.
- `vote_average` y `vote_count`: el público vota después de ver la película. No entran al modelo; se usan solo
  para auditar sus predicciones.
- «Pertenece a una saga»: TMDb crea la saga cuando sale una secuela, y eso pasa si a la primera le fue bien
  (sección 4). Sí entra «es secuela», que se sabe antes.
""")

pase("Ya tenemos las películas. <b>Primera pregunta:</b> ¿les va mejor a las que cuestan mucho o a las que cuestan poco?")
st.divider()

# =============================================================================================
# 2 · LO QUE ENCONTRAMOS (Entrega 2)
# =============================================================================================
esc = pd.DataFrame(M["hallazgo_presupuesto"]["por_escala"])
barato, medio, caro = esc.iloc[0], esc.iloc[2], esc.iloc[3]
seccion("hallazgo", "2 · Lo que encontramos",
        "Las caras ganan más plata; las baratas multiplican más lo que cuestan, pero son una lotería",
        f"Depende de cómo lo midas. En dólares, una película típica de 10 a 100 millones deja "
        f"{S.usd_corto(medio['ganancia_mediana'])} de ganancia recuperando solo {S.mult(medio['multiplo_mediano'], 0)}; "
        f"una de menos de 1 millón deja {S.usd_corto(barato['ganancia_mediana'])} aunque multiplique "
        f"{S.mult(barato['multiplo_mediano'], 0)}. En múltiplo pasa al revés.")

esc["nombre"] = esc["escala"] + " (" + esc["peliculas"].map(S.num) + ")"
esc["etiqueta"] = [f"recauda {S.usd_corto(r)} · gana {S.usd_corto(g)}" for r, g in
                   zip(esc["revenue_mediana"], esc["ganancia_mediana"])]
esc["costo_txt"], esc["recauda_txt"] = esc["budget_mediano"].map(S.usd), esc["revenue_mediana"].map(S.usd)
titulo_grafico("En dólares: las caras ganan más",
               "Película típica (mediana) de cada tramo de presupuesto: hueco = lo que cuesta, dorado = lo que recauda.")
# Gráfico: de lo que cuesta a lo que recauda, por tramo de presupuesto.
#  datos: metricas.json["hallazgo_presupuesto"] (build_app agrupa las 6.391 películas en 4 tramos y calcula la
#         mediana del presupuesto, de la recaudación y de la ganancia con pandas)
#  mapeo: y = tramo (ordinal, de barato a caro), x = dólares (escala LOG)
#  marcas: regla de lo que cuesta a lo que recauda; círculo hueco = cuesta, lleno = recauda; texto = ganancia
y_esc = alt.Y("nombre:N", sort=list(esc["nombre"]), title="Presupuesto (películas)",
              axis=alt.Axis(labelFontSize=12.5, labelLimit=200))
x_esc = alt.Scale(type="log", domain=[1e5, 3e10], nice=False)
ax_esc = alt.Axis(title="Dólares (escala log)", values=[1e5, 1e6, 1e7, 1e8, 1e9], labelExpr=S.ETIQUETA_USD)
tip_esc = [alt.Tooltip("escala:N", title="Presupuesto"), alt.Tooltip("costo_txt:N", title="Cuesta (típica)"),
           alt.Tooltip("recauda_txt:N", title="Recauda (típica)"), alt.Tooltip("etiqueta:N", title=" ")]
be = alt.Chart(esc)
mostrar(alt.layer(
    be.mark_rule(color=S.EJE, strokeWidth=4).encode(x=alt.X("budget_mediano:Q", scale=x_esc, axis=ax_esc),
                                                    x2="revenue_mediana:Q", y=y_esc, tooltip=tip_esc),
    be.mark_point(size=230, filled=True, fill=S.SUPERFICIE, stroke=S.TINTA_2, strokeWidth=2.5).encode(
        x=alt.X("budget_mediano:Q", scale=x_esc), y=y_esc, tooltip=tip_esc),
    be.mark_circle(size=230, color=S.DORADO, opacity=1).encode(
        x=alt.X("revenue_mediana:Q", scale=x_esc), y=y_esc, tooltip=tip_esc),
    be.mark_text(align="left", dx=14, fontSize=12, color=S.TINTA).encode(
        x=alt.X("revenue_mediana:Q", scale=x_esc), y=y_esc, text="etiqueta:N"),
), alto=240)

rng = np.random.default_rng(42)
strip = P[["title", "release_year", "escala", "multiplo_real", "multiplo_real_txt", "presupuesto_txt", "resultado"]].copy()
strip["jit"] = rng.uniform(-0.38, 0.38, len(strip))  # dispersión vertical fija (semilla), solo para que no se pisen
strip["nombre"] = strip["escala"].map(dict(zip(esc["escala"], esc["nombre"])))
med = esc.assign(t=esc["multiplo_mediano"].map(lambda v: f"típica {S.mult(v, 1)}"))
fam_s = strip[strip["title"].isin(["Paranormal Activity", "Rocky", "Star Wars", "Titanic", "Gattaca",
                                   "The Irishman"])].drop_duplicates("title")
titulo_grafico("La lotería, película por película",
               "Cada punto es una película, en su tramo de presupuesto; el trazo dorado marca la típica. Cuanto más barata, "
               "más se desparrama: hay milagros y desastres. Las caras se amontonan cerca de ×1 a ×10.")
# Gráfico: dispersión del múltiplo dentro de cada tramo de presupuesto (strip plot con jitter).
#  datos: predicciones.parquet (las 6.391); jitter vertical fijo con semilla (numpy), solo para separar los puntos
#  mapeo: y = tramo (ordinal) + yOffset = jitter, x = múltiplo real (LOG), color = resultado (misma escala que la portada)
#  marcas: círculos chicos; tick dorado = mediana del tramo; texto = la típica; anillos dorados = famosas
y_st = alt.Y("nombre:N", sort=list(esc["nombre"]), title=None, axis=alt.Axis(labelFontSize=12.5, labelLimit=200))
x_st = alt.Scale(type="log", domain=[0.0005, 20000], nice=False, clamp=True)
mostrar(alt.layer(
    alt.Chart(strip).mark_circle(size=11, opacity=0.6).encode(
        x=alt.X("multiplo_real:Q", scale=x_st, axis=S.eje_multiplo("Cuántas veces recuperó lo que costó (log)")),
        y=y_st, yOffset=alt.YOffset("jit:Q", scale=alt.Scale(domain=[-0.5, 0.5])),
        color=alt.Color("resultado:N", scale=ESCALA_RESULTADO, legend=None),
        tooltip=[alt.Tooltip("title:N", title="Película"), alt.Tooltip("release_year:Q", title="Año"),
                 alt.Tooltip("presupuesto_txt:N", title="Costó"), alt.Tooltip("multiplo_real_txt:N", title="Recuperó")]),
    alt.Chart(pd.DataFrame({"x": [1]})).mark_rule(strokeDash=[5, 4], color=S.TINTA_SUAVE).encode(x=alt.X("x:Q", scale=x_st)),
    alt.Chart(med).mark_tick(color=S.DORADO, thickness=4, size=96).encode(x=alt.X("multiplo_mediano:Q", scale=x_st), y=y_st),
    alt.Chart(med).mark_text(align="center", dy=-62, fontSize=13, fontWeight=700, color=S.DORADO).encode(
        x=alt.X("multiplo_mediano:Q", scale=x_st), y=y_st, text="t:N"),
    alt.Chart(fam_s).mark_point(size=110, color=S.DORADO, strokeWidth=1.8).encode(
        x=alt.X("multiplo_real:Q", scale=x_st), y=y_st, yOffset=alt.YOffset("jit:Q", scale=alt.Scale(domain=[-0.5, 0.5]))),
    alt.Chart(fam_s).mark_text(align="left", dx=8, fontSize=11.5, fontWeight=600, color=S.DORADO).encode(
        x=alt.X("multiplo_real:Q", scale=x_st), y=y_st, yOffset=alt.YOffset("jit:Q", scale=alt.Scale(domain=[-0.5, 0.5])),
        text="title:N"),
), alto=600)

dist = (P.groupby(["escala", "tramo_roi"]).size().reset_index(name="n"))
dist["tot"] = dist.groupby("escala")["n"].transform("sum")
dist["pct"] = dist["n"] / dist["tot"]
dist["resultado"] = dist["tramo_roi"].map(TRAMOS_ES)
dist["orden_res"] = dist["tramo_roi"].map({t: i for i, t in enumerate(TRAMOS)})
dist["pct_txt"] = dist["pct"].map(S.pct)
dist["nombre"] = dist["escala"].map(dict(zip(esc["escala"], esc["nombre"])))
tipica = esc.assign(t=esc["multiplo_mediano"].map(lambda v: f"típica {S.mult(v, 1)}"))
titulo_grafico("En múltiplo: las baratas multiplican más",
               "Qué le pasó a cada película del tramo, de perder plata (gris) a multiplicar más de 11 veces (verde brillante).")
# Gráfico: resultado de las películas de cada tramo (partes de un todo).
#  datos: predicciones.parquet; pandas cuenta las películas por tramo de presupuesto y tramo de resultado
#  transformación: porcentaje dentro de cada tramo (suma 100%)
#  marca: barra horizontal apilada al 100%; mapeo: y = tramo, x = % de películas, color = resultado (ordinal)
#  texto: el % dentro de cada parte (si entra) y el múltiplo típico al final
y_d = alt.Y("nombre:N", sort=list(esc["nombre"]), title=None, axis=alt.Axis(labelFontSize=12.5, labelLimit=200))
bd = alt.Chart(dist)
mostrar(alt.layer(
    bd.mark_bar(height=34, stroke=S.SUPERFICIE, strokeWidth=1).encode(
        x=alt.X("pct:Q", stack="normalize", title="Películas del tramo",
                axis=alt.Axis(format="%", values=[0, .25, .5, .75, 1])),
        y=y_d, order=alt.Order("orden_res:Q"),
        color=alt.Color("resultado:N", scale=ESCALA_RESULTADO,
                        legend=alt.Legend(title=None, orient="top", symbolType="square")),
        tooltip=[alt.Tooltip("escala:N", title="Presupuesto"), alt.Tooltip("resultado:N", title="Resultado"),
                 alt.Tooltip("pct_txt:N", title="De las películas"), alt.Tooltip("n:Q", title="Películas")]),
    bd.transform_filter(alt.datum.pct >= 0.09).mark_text(fontSize=11.5, fontWeight="bold").encode(
        x=alt.X("pct:Q", stack="normalize", bandPosition=0.5), y=y_d, order=alt.Order("orden_res:Q"),
        detail="resultado:N", text="pct_txt:N",
        color=alt.condition(alt.datum.orden_res >= 3, alt.value(S.SUPERFICIE), alt.value(S.TINTA))),
).resolve_scale(color="independent"), alto=260)

st.markdown(
    f"Las de menos de 1 millón son pocas ({S.num(barato['peliculas'])}) y extremas: más de la mitad multiplica más de "
    f"11 veces, pero igual una de cada ocho pierde plata. Las de 10 a 100 millones son las más riesgosas: "
    f"**{S.pct(medio['pierde'])} pierde plata**. Las de más de 100 millones casi nunca explotan, pero tampoco se hunden "
    f"seguido ({S.pct(caro['pierde'])}): son apuestas grandes y relativamente seguras.")

fichas = {f["id"]: f for f in M["entrega2"]["fichas"]}
f1, f2 = fichas[1], fichas[2]
with como_lo_sabemos():
    st.markdown(f"""
En la Entrega 2 pusimos a prueba cinco hipótesis, cada una con un umbral fijado antes de mirar. Acá están
recalculadas con el dataset actual, con el mismo método del notebook `presentacion2.ipynb`: las
{S.num(M['entrega2']['universo']['con_montos'])} películas con los dos montos mayores a 0, recortando el 1% de ROI más
alto ({S.num(M['entrega2']['universo']['recortado_p99'])} películas).

**Una que se sostuvo: a más presupuesto, más recaudación.** Esperábamos un Pearson de al menos 0,60 y dio
**{S.num(f1['pearson_logbudget_logrevenue'], 2)}** entre log10(presupuesto) y log10(recaudación). Pero contra el ROI da
**{S.num(f1['spearman_logbudget_roi'], 2)}** (Spearman): las caras recaudan más en dólares y recuperan menos veces lo que
costaron. Son los dos gráficos de arriba.

**Una que no: las películas en inglés recuperan más.** Esperábamos una separación estandarizada mayor a 0,80 a favor del
inglés y dio **{S.num(f2['separacion_estandarizada'], 2)}**: el signo es el contrario. La mediana del ROI es
{S.num(f2['mediana_en'], 2)} en inglés y {S.num(f2['mediana_resto'], 2)} en el resto. Tiene sentido con lo de arriba:
Hollywood filma más caro.
""")
    st.dataframe(pd.DataFrame([{"Hipótesis": f["afirmacion"], "Métrica": f["metrica"], "Esperábamos": f["esperado"],
                                "Dio": S.num(f["valor"], 4), "¿Se sostiene?": "Sí" if f["se_sostiene"] else "No"}
                               for f in fichas.values()]), hide_index=True, width="stretch")
    st.markdown(f"""
Las otras tres también cayeron: el género explica el {S.pct(fichas[3]['eta2'], 1)} de la variación del ROI (η²), la
popularidad del reparto se asocia levemente en contra ({S.num(fichas[4]['pearson'], 2)}) y el mes de estreno explica el
{S.pct(fichas[5]['eta2_mes'], 1)}. Por eso ninguna de esas columnas entra sola: entran junto con el presupuesto, para
que el modelo encuentre las combinaciones.

**Cómo se construyeron los gráficos.** Con las {S.num(n_alcance)} películas de la sección 1. Se corta el presupuesto en
cuatro tramos por potencias de 10 (1, 10 y 100 millones). El de la izquierda usa la **mediana** del presupuesto, de la
recaudación y de la ganancia de cada tramo (no el promedio, porque unos pocos fenómenos lo arrastrarían). El de la
derecha cuenta, en cada tramo, qué porcentaje de películas cayó en cada resultado.

**Por qué escala logarítmica.** El múltiplo va de ×0,001 a más de ×10.000, y los presupuestos, de 10 mil a cientos de
millones. En escala lineal unas pocas aplastan a todas contra el cero; en escala log cada salto de 10 veces ocupa lo
mismo. El modelo predice justamente y = log10(recaudación / presupuesto). Así se reparte:
""")
    # Gráfico: distribución del objetivo y.
    #  datos: predicciones.parquet (columna y); transformación: transform_bin en intervalos de 0,1
    #  mapeo: x = intervalo de y, y = cantidad de películas; marca: barra; regla punteada en y = 0 (recupera justo)
    hist = alt.Chart(P[["y"]]).transform_bin("y_bin", "y", bin=alt.Bin(step=0.1)).mark_bar(
        color=S.REAL, opacity=0.8).encode(
        x=alt.X("y_bin:Q", title="y = log10(recaudación / presupuesto)  ·  0 = recupera justo, 1 = ×10, −1 = pierde el 90%",
                bin="binned"),
        x2="y_bin_end:Q", y=alt.Y("count():Q", title="Películas"),
        tooltip=[alt.Tooltip("y_bin:Q", title="Desde", format=".1f"), alt.Tooltip("count():Q", title="Películas")])
    cero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=S.PISO, strokeDash=[4, 4]).encode(x="x:Q")
    mostrar(alt.layer(hist, cero), alto=220)

pase("Si el resultado es tan variable, <b>¿se puede anticipar?</b> Le pedimos a un modelo que lo intente, con lo que "
     "se sabe antes del estreno.")
st.divider()

# =============================================================================================
# 3 · EL MODELO (Entrega 3)
# =============================================================================================
seccion("modelo", "3 · El modelo",
        f"En películas que nunca vio, el modelo se equivoca un {S.pct(FIN['mejora_test_pct'], 1)} menos que adivinar: "
        "acierta con las normales y falla en las puntas",
        "Aprendió con las películas hasta 2017 y lo medimos una sola vez, al final, con las de 2017 a 2025. Le gana "
        "a adivinar, pero el margen sigue siendo grande.")

c1, c2 = st.columns([1, 1.25], gap="large")
with c1:
    titulo_grafico("¿Le gana a adivinar?",
                   f"Error típico en el examen final ({S.num(M['particion']['n_test'])} películas). Menos es mejor.")
    # Gráfico: error del piso contra el modelo final en el test.
    #  datos: metricas.json["final"] (MAE de test pasado a factor: 10^MAE)
    #  marca: barra horizontal; mapeo: y = quién predice, x = factor de error (lineal desde ×1, que sería acertar
    #  siempre), color = rol (gris = piso, violeta = modelo); texto = el factor
    duelo = pd.DataFrame({"quien": ["Adivinar", "Nuestro modelo"],
                          "factor": [FIN["factor_piso_test"], FIN["factor_test"]], "rol": ["Piso", "Modelo final"]})
    duelo["txt"] = duelo["factor"].map(lambda v: S.mult(v, 2))
    duelo["uno"] = 1.0  # las barras arrancan en ×1 (acertar siempre), no en 0
    bdu = alt.Chart(duelo)
    y_du = alt.Y("quien:N", sort=list(duelo["quien"]), title=None, axis=alt.Axis(labelFontSize=13, labelLimit=240))
    mostrar(alt.layer(
        bdu.mark_bar(height=42, cornerRadiusEnd=5).encode(
            x2="uno:Q", x=alt.X("factor:Q", title="Error típico (×1 = acierta siempre)", scale=alt.Scale(domain=[1, 3.8]),
                    axis=alt.Axis(values=[1, 1.5, 2, 2.5, 3, 3.5], labelExpr=S.ETIQUETA_MULTIPLO)),
            y=y_du, color=alt.Color("rol:N", scale=alt.Scale(domain=["Piso", "Modelo final"], range=[S.GRIS_1, S.MODELO]),
                                    legend=None)),
        bdu.mark_text(align="left", dx=8, fontSize=16, fontWeight="bold", color=S.TINTA).encode(
            x="factor:Q", y=y_du, text="txt:N"),
    ), alto=170)
    st.markdown(f"**{S.pct(FIN['mejora_test_pct'], 1)} menos error.** Si una película recuperó ×2, adivinar dice "
                f"{S.mult(10 ** FIN['mediana_piso_y'], 1)} siempre; el modelo suele quedar entre "
                f"{S.mult(2 / FIN['factor_test'], 1)} y {S.mult(2 * FIN['factor_test'], 1)}.")

orden_modelos = ["Piso (mediana)", "Ridge", "Random Forest", "HistGB (absolute_error)", E.NOMBRE_RF, E.NOMBRE_FINAL]
nombres_cortos = {"Piso (mediana)": "Piso (adivinar)", "Ridge": "Ridge (lineal)",
                  "Random Forest": "Random Forest", "HistGB (absolute_error)": "HistGB (boosting)",
                  E.NOMBRE_RF: "Random Forest ajustado", E.NOMBRE_FINAL: "Promedio de los 3 (final)"}
mm = modelos.loc[orden_modelos].reset_index()
mm["nombre"] = mm["modelo"].map(nombres_cortos)
mm["rol"] = np.select([mm["es_final"], mm["modelo"] == "Piso (mediana)"], ["Modelo final", "Piso"], "Otros modelos")
mm["lo"], mm["hi"] = fac(mm["mae_cv"] - mm["desvio_cv"]), fac(mm["mae_cv"] + mm["desvio_cv"])
mm["f"] = fac(mm["mae_cv"])
mm["f_txt"] = mm["f"].map(lambda v: S.mult(v, 2))
with c2:
    titulo_grafico("¿Por qué este modelo?",
                   "Práctica: 5 pruebas con películas posteriores a las que vio al aprender. Punto = error promedio; "
                   "línea = cuánto varía entre pruebas.")
    # Gráfico: error de cada modelo en la validación cruzada temporal.
    #  datos: metricas.json["modelos"] (MAE medio y desvío en 5 cortes, pasados a factor 10^MAE)
    #  mapeo: y = modelo (orden fijo, del más simple al final), x = factor (no arranca en ×1: se comparan diferencias
    #         chicas), color = rol (violeta = final, gris = piso, gris claro = otros)
    #  marcas: regla = ± un desvío entre cortes; punto = error medio; texto = factor
    escala_rol = alt.Scale(domain=["Modelo final", "Piso", "Otros modelos"], range=[S.MODELO, S.GRIS_1, S.GRIS_3])
    y_mod = alt.Y("nombre:N", sort=[nombres_cortos[m] for m in orden_modelos], title=None,
                  axis=alt.Axis(labelLimit=220, labelFontSize=12.5))
    x_mod = alt.Scale(domain=[2.05, 3.0], nice=False)
    bm = alt.Chart(mm)
    mostrar(alt.layer(
        bm.mark_rule(strokeWidth=2.5).encode(
            x=alt.X("lo:Q", scale=x_mod, title="Factor de error en la práctica",
                    axis=alt.Axis(values=[2.2, 2.4, 2.6, 2.8, 3.0], labelExpr=S.ETIQUETA_MULTIPLO)),
            x2="hi:Q", y=y_mod, color=alt.Color("rol:N", scale=escala_rol, legend=None)),
        bm.mark_circle(size=190, opacity=1).encode(
            x=alt.X("f:Q", scale=x_mod), y=y_mod, color=alt.Color("rol:N", scale=escala_rol, legend=None),
            tooltip=[alt.Tooltip("modelo:N", title="Modelo"), alt.Tooltip("f_txt:N", title="Error promedio")]),
        bm.mark_text(dy=-13, fontSize=11.5, color=S.TINTA_2).encode(x=alt.X("f:Q", scale=x_mod), y=y_mod, text="f_txt:N"),
    ), alto=270)
    st.markdown("Todos le ganan a adivinar, y entre ellos casi empatan. Elegimos el **promedio de los tres** porque "
                "es el más estable y no depende de un solo modelo.")

T = P[P["conjunto"] == "test"].copy()
p_lo, p_hi = T["multiplo_pred"].quantile([0.01, 0.99])
r_lo, r_hi = T["multiplo_real"].quantile([0.01, 0.99])
titulo_grafico("Así achata el modelo",
               f"Las {S.num(len(T))} películas del examen final. Si el modelo acertara siempre, todas caerían sobre la "
               "diagonal; la franja violeta es su error típico. Lo real se reparte de "
               f"{S.mult(r_lo, 2)} a {S.mult(r_hi, 0)}; lo que predice, de {S.mult(p_lo, 1)} a {S.mult(p_hi, 1)}: "
               "los puntos quedan aplastados contra el medio.")
# Gráfico: lo real contra lo predicho, película por película (test).
#  datos: predicciones.parquet filtrado al test (1.278 películas que el modelo nunca vio)
#  mapeo: x = múltiplo real (LOG), y = múltiplo predicho (LOG, mismo dominio que x), color = error del modelo
#         (divergente: azul = predijo de menos, rojo = predijo de más, la misma escala de «Explorá»)
#  marcas: círculos; regla diagonal = acierto perfecto; área violeta = ± el error típico; texto dorado = famosas
dom_rp = [0.002, 2000]
esc_rp = alt.Scale(type="log", domain=dom_rp, nice=False, clamp=True)
xs = np.logspace(np.log10(dom_rp[0]), np.log10(dom_rp[1]), 60)
franja = pd.DataFrame({"x": xs, "lo": xs / FIN["factor_test"], "hi": xs * FIN["factor_test"]})
fam_t = T[T["title"].isin(["Get Out", "The Irishman", "Joker", "Cats", "Avengers: Endgame"])].drop_duplicates("title").copy()
fam_t["etiqueta"] = fam_t["title"]
ax_rp = dict(values=[0.01, 0.1, 1, 10, 100, 1000], labelExpr=S.ETIQUETA_MULTIPLO)
mostrar(alt.layer(
    alt.Chart(franja).mark_area(color=S.MODELO, opacity=0.16).encode(
        x=alt.X("x:Q", scale=esc_rp, axis=alt.Axis(title="Lo que recuperó de verdad (log)", **ax_rp)),
        y=alt.Y("lo:Q", scale=esc_rp, axis=alt.Axis(title="Lo que dijo el modelo (log)", **ax_rp)), y2="hi:Q"),
    alt.Chart(franja).mark_line(color=S.MODELO, strokeDash=[6, 4], strokeWidth=1.5).encode(
        x=alt.X("x:Q", scale=esc_rp), y=alt.Y("x:Q", scale=esc_rp)),
    alt.Chart(T).mark_circle(size=26, opacity=0.8).encode(
        x=alt.X("multiplo_real:Q", scale=esc_rp), y=alt.Y("multiplo_pred:Q", scale=esc_rp),
        color=alt.Color("error_log:Q", scale=S.escala_error(), legend=None),
        tooltip=[alt.Tooltip("title:N", title="Película"), alt.Tooltip("release_year:Q", title="Año"),
                 alt.Tooltip("multiplo_real_txt:N", title="Recuperó"), alt.Tooltip("multiplo_pred_txt:N", title="Dijo el modelo"),
                 alt.Tooltip("error_txt:N", title="Error")]),
    alt.Chart(fam_t).mark_point(size=150, color=S.DORADO, strokeWidth=2).encode(
        x=alt.X("multiplo_real:Q", scale=esc_rp), y=alt.Y("multiplo_pred:Q", scale=esc_rp)),
    alt.Chart(fam_t).mark_text(align="left", dx=10, dy=-8, fontSize=12, fontWeight=600, color=S.DORADO).encode(
        x=alt.X("multiplo_real:Q", scale=esc_rp), y=alt.Y("multiplo_pred:Q", scale=esc_rp), text="etiqueta:N"),
    alt.Chart(pd.DataFrame({"x": [400], "y": [400 * 1.6], "t": ["acierto perfecto"]})).mark_text(
        angle=360 - 37, fontSize=11.5, color=S.MODELO, align="right").encode(
        x=alt.X("x:Q", scale=esc_rp), y=alt.Y("y:Q", scale=esc_rp), text="t:N"),
), alto=520)
st.markdown("A la izquierda, abajo de la diagonal quedarían los fracasos bien anticipados; están arriba: el modelo les "
            "dice que van a ganar (*The Irishman*, *Cats*). A la derecha, los fenómenos quedan abajo: *Get Out* recuperó "
            "×57 y el modelo le daba menos de ×4. Azul = predijo de menos; rojo = predijo de más.")

tramos["real"] = tramos["roi_real_mediano"] + 1
tramos["con"] = tramos["roi_pred_mediano"] + 1
tramos["sin"] = tramos["roi_pred_sin_ctx_mediano"] + 1
tramos["tramo_es"] = tramos["tramo"].map(TRAMOS_ES) + " (" + tramos["peliculas"].map(S.num) + ")"
tl = tramos.melt(id_vars=["tramo_es", "peliculas"], value_vars=["real", "con"], var_name="valor", value_name="multiplo")
tl["valor"] = tl["valor"].map({"real": "Lo que pasó de verdad", "con": "Lo que dijo el modelo"})
tl["multiplo_txt"] = tl["multiplo"].map(lambda v: S.mult(v, 2))
titulo_grafico("¿Dónde acierta y dónde falla?",
               "Películas del examen agrupadas por lo que les pasó de verdad. Blanco = lo real (típico del grupo); "
               "violeta = lo que dijo el modelo. Cuanto más larga la línea, más le erra.")
# Gráfico: lo real contra lo predicho, por tramo de resultado (test).
#  datos: metricas.json["por_tramo"] (mediana del múltiplo real y del predicho en cada tramo), en formato largo
#  mapeo: y = tramo (ordinal, de perder plata a multiplicar más de 11 veces), x = múltiplo (escala LOG),
#         color = real (gris tinta) o predicho (violeta)
#  marcas: regla entre los dos (su largo es el error) + punto para cada uno + texto con el valor
y_tr = alt.Y("tramo_es:N", sort=list(tramos["tramo_es"]), title=None, axis=alt.Axis(labelLimit=240, labelFontSize=12.5))
x_tr = alt.Scale(type="log", domain=[0.25, 30], nice=False)
mostrar(alt.layer(
    alt.Chart(tramos).mark_rule(color=S.EJE, strokeWidth=3).encode(
        y=y_tr, x=alt.X("real:Q", scale=x_tr, axis=S.eje_multiplo("Cuántas veces recuperó lo que costó (escala log)")),
        x2="con:Q"),
    alt.Chart(pd.DataFrame({"x": [1]})).mark_rule(strokeDash=[4, 4], color=S.PISO).encode(x=alt.X("x:Q", scale=x_tr)),
    alt.Chart(tl).mark_circle(size=200, opacity=1).encode(
        y=y_tr, x=alt.X("multiplo:Q", scale=x_tr),
        color=alt.Color("valor:N", scale=alt.Scale(domain=["Lo que pasó de verdad", "Lo que dijo el modelo"],
                                                   range=[S.REAL, S.MODELO]), legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip("tramo_es:N", title="Grupo"), alt.Tooltip("valor:N", title=" "),
                 alt.Tooltip("multiplo_txt:N", title="Múltiplo típico")]),
    alt.Chart(tl).mark_text(dy=-14, fontSize=11.5).encode(
        y=y_tr, x=alt.X("multiplo:Q", scale=x_tr), text="multiplo_txt:N",
        color=alt.Color("valor:N", scale=alt.Scale(domain=["Lo que pasó de verdad", "Lo que dijo el modelo"],
                                                   range=[S.REAL, S.MODELO]), legend=None)),
), alto=300)
perdida, top = tramos.iloc[0], tramos.iloc[-1]
st.markdown(
    f"En el medio acierta bastante. En las puntas, no: a las que **perdieron plata** (típica {S.mult(perdida['real'], 2)}) "
    f"les dice {S.mult(perdida['con'], 2)}, o sea que les anticipa ganancia; a las que **multiplicaron más de 11 veces** "
    f"(típica {S.mult(top['real'], 1)}) les dice {S.mult(top['con'], 1)}. Es lo que el profe llamó «achatar». Con lo que se "
    "sabe antes del estreno, un fenómeno viral se parece mucho a una película chica cualquiera.")

im = imp.head(10).copy()
im["nombre"] = im["variable"].map(VAR_ES)
im["origen"] = im["variable"].map(origen)
im["txt"] = im["aumento_mae"].map(lambda v: S.num(v, 4))
im["lo"], im["hi"] = im["aumento_mae"] - im["desvio"], im["aumento_mae"] + im["desvio"]
titulo_grafico("¿En qué se apoya?",
               "Cuánto empeora el modelo si se le desordena cada columna (las 10 que más pesan). Más largo = más importante.")
# Gráfico: importancia por permutación en el test.
#  datos: metricas.json["importancia"] (build_app: permutation_importance con 20 repeticiones)
#  mapeo: y = columna (ordenada por importancia), x = cuánto sube el MAE, color = de dónde viene la columna
#  marcas: barra + regla de ± un desvío entre repeticiones
y_im = alt.Y("nombre:N", sort=list(im["nombre"]), title=None, axis=alt.Axis(labelLimit=300, labelFontSize=12.5))
mostrar(alt.layer(
    alt.Chart(im).mark_bar(height=18, cornerRadiusEnd=4).encode(
        x=alt.X("aumento_mae:Q", title="Cuánto sube el error al desordenar la columna (MAE)", axis=alt.Axis(tickCount=7)), y=y_im,
        color=alt.Color("origen:N", scale=ESCALA_ORIGEN, legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("nombre:N", title="Columna"), alt.Tooltip("variable:N", title="En el código"),
                 alt.Tooltip("origen:N", title="Origen"), alt.Tooltip("txt:N", title="Aumento del MAE")]),
    alt.Chart(im).mark_rule(color=S.TINTA_2).encode(x="lo:Q", x2="hi:Q", y=y_im),
), alto=330)
st.markdown(f"El **presupuesto** pesa {S.num(peso_budget, 1)} veces más que la segunda columna. Atrás vienen la "
            "combinación de director y protagonista conocidos y **cómo les fue antes a las películas de la saga**, una "
            "de las columnas nuevas (sección 4).")

# Gráfico: el modelo a lo largo del tiempo, antes y después de 2017 (arranque del streaming).
#  datos: predicciones.parquet (las 6.391 del alcance; hasta 2016 son películas con las que el modelo aprendió)
#  transformación: por año de estreno desde 1980 (antes hay menos de 100 películas por año) se toma la mediana
#         del múltiplo real y del predicho, y el MAE (error absoluto medio de log10) de ese año
#  mapeo A: x = año, y = múltiplo mediano (lineal), color = real (gris) o predicción (violeta)
#  mapeo B: x = año, y = factor de error 10^MAE, color = si el modelo ya había visto esas películas o no
#  marcas: línea + punto (A), barras (B), regla vertical dorada en 2017 en los dos
anual = (P[P["release_year"] >= 1980].groupby("release_year")
         .agg(n=("y", "size"), real=("y", "median"), pred=("y_pred", "median"), mae=("error_abs", "mean"))
         .reset_index())
anual["factor"] = 10 ** anual["mae"]
anual["uno"] = 1.0  # las barras arrancan en ×1 (acertar siempre), no en 0
anual["real"] = 10 ** anual["real"].rolling(3, center=True, min_periods=2).mean()
anual["pred"] = 10 ** anual["pred"].rolling(3, center=True, min_periods=2).mean()
anual["etapa"] = np.where(anual["release_year"] < anios_test[0], "Ya las había visto",
                          "Examen (2017–2025)")
anual["factor_txt"] = anual["factor"].map(lambda v: S.mult(v, 2))
anual["n_txt"] = anual["n"].map(lambda v: f"{S.num(v)} películas")
largo_t = anual.melt(id_vars=["release_year", "n_txt"], value_vars=["real", "pred"], var_name="serie", value_name="mult")
largo_t["serie"] = largo_t["serie"].map({"real": "Lo que recuperaron de verdad", "pred": "Lo que predijo el modelo"})
largo_t["mult_txt"] = largo_t["mult"].map(lambda v: S.mult(v, 2))
anio_stream = pd.DataFrame({"release_year": [anios_test[0]], "t": ["2017: despega el streaming"]})
x_t = alt.X("release_year:Q", title="Año de estreno", scale=alt.Scale(domain=[1980, 2025]),
            axis=alt.Axis(format="d", values=list(range(1980, 2026, 5)), labelExpr="datum.label"))
regla_t = alt.Chart(anio_stream).mark_rule(color=S.DORADO, strokeDash=[5, 4], strokeWidth=2).encode(x=x_t)
texto_t = alt.Chart(anio_stream).mark_text(align="right", dx=-6, dy=10, baseline="top", color=S.DORADO, fontSize=12.5,
                                          fontWeight="bold").encode(x=x_t, y=alt.value(0), text="t:N")

titulo_grafico("Antes y después de 2017: el modelo se separa de la realidad",
               "Año por año (líneas: promedio móvil de 3 años), lo que recuperó la película mediana contra lo que predijo el modelo, y cuánto se "
               "equivocó. Hasta 2016 son películas con las que el modelo aprendió; desde 2017 son las del examen.")
ct1, ct2 = st.columns(2, gap="large")
with ct1:
    mostrar(alt.layer(
        alt.Chart(largo_t).mark_line(strokeWidth=2.6, point=alt.OverlayMarkDef(size=36, filled=True)).encode(
            x=x_t, y=alt.Y("mult:Q", title="Cuántas veces recupera lo que costó (mediana)",
                           scale=alt.Scale(domain=[1.4, 3.7], nice=False),
                           axis=alt.Axis(values=[1.5, 2, 2.5, 3, 3.5], labelExpr=S.ETIQUETA_MULTIPLO)),
            color=alt.Color("serie:N", scale=alt.Scale(domain=["Lo que recuperaron de verdad", "Lo que predijo el modelo"],
                                                       range=[S.TINTA_2, S.MODELO]), legend=alt.Legend(title=None)),
            tooltip=[alt.Tooltip("release_year:Q", title="Año", format="d"), alt.Tooltip("serie:N", title=""),
                     alt.Tooltip("mult_txt:N", title="Mediana"), alt.Tooltip("n_txt:N", title="Muestra")]),
        regla_t, texto_t), alto=300)
with ct2:
    mostrar(alt.layer(
        alt.Chart(anual).mark_bar(cornerRadiusEnd=2, size=7).encode(
            x=alt.X("release_year:Q", title="Año de estreno", scale=alt.Scale(domain=[1980, 2025]),
                    axis=alt.Axis(format="d", values=list(range(1980, 2026, 5)), labelExpr="datum.label")),
            y2="uno:Q", y=alt.Y("factor:Q", title="Error típico (×1 = acierta siempre)",
                    scale=alt.Scale(domain=[1, 3.6], nice=False),
                    axis=alt.Axis(values=[1, 1.5, 2, 2.5, 3], labelExpr=S.ETIQUETA_MULTIPLO)),
            color=alt.Color("etapa:N", scale=alt.Scale(domain=["Ya las había visto", "Examen (2017–2025)"],
                                                       range=[S.GRIS_1, S.DORADO]), legend=alt.Legend(title=None)),
            tooltip=[alt.Tooltip("release_year:Q", title="Año", format="d"), alt.Tooltip("factor_txt:N", title="Error típico"),
                     alt.Tooltip("n_txt:N", title="Muestra")]),
        ), alto=300)
st.markdown(
    f"**Hasta 2016 el modelo acompaña la tendencia; desde 2017 predice más de lo que las películas recuperan.** "
    f"Con películas que no había visto al ajustarse (validación, 1988–2017) se equivoca {S.mult(fac(FIN['mae_cv']))}; "
    f"en el examen {anios_test[0]}–{anios_test[1]}, {S.mult(FIN['factor_test'])}. El error sube, y justo en el año "
    "en que las plataformas de streaming empiezan a ser el lugar donde se estrena una película: muchas ya no pasan por el "
    "cine, o pasan unos días, y su recaudación en TMDb es mínima o inexistente. El modelo, entrenado con 100 años de "
    "estrenos en sala, sigue esperando una taquilla que ya no llega.")
st.caption(f"Es una pista, no una prueba: 2020 y 2021 (pandemia) caen en el mismo tramo, no hay forma de separar los "
           f"dos efectos con estos datos, y el piso (adivinar la mediana) también empeora, de {S.mult(fac(FIN['mae_piso_cv']))} "
           f"a {S.mult(FIN['factor_piso_test'])}. Antes de 2017 las barras son optimistas: el modelo ya había visto esas "
           "películas.")

with como_lo_sabemos():
    st.markdown(f"""
**La métrica: MAE sobre log10(recaudación / presupuesto).** Lo caro es errarle al *múltiplo*: prever que una película
triplica su costo y que lo recupere a medias es el mismo error en una de 1 millón que en una de 200. Un MAE de
{S.num(FIN['mae_test'], 2)} quiere decir que la predicción está errada, en promedio, por un factor de
10^{S.num(FIN['mae_test'], 2)} = {S.mult(FIN['factor_test'])}, para arriba o para abajo. **¿Por qué no el ROI crudo?** El
error del piso sobre el ROI crudo da, por corte, {", ".join(S.num(x, 1) for x in M['roi']['mae_crudo_por_fold'])}: el
corte 4 se dispara por *Paranormal Activity* (ROI ≈ 12.900). Con ese desvío no se distingue ningún modelo del piso.

**El piso:** `DummyRegressor(strategy="median")`, que dice siempre la mediana del entrenamiento
({S.mult(10 ** FIN['mediana_piso_y'])}) sin mirar ninguna columna.

**La validación imita el uso:** el modelo se usaría para películas futuras con lo aprendido de las pasadas.
1. Las películas se ordenan por fecha de estreno.
2. El **20% más reciente** ({S.num(M['particion']['n_test'])} películas, {anios_test[0]}–{anios_test[1]}) es el examen
   final (test): se aparta al principio y se mide una sola vez.
3. Sobre el 80% restante ({S.num(M['particion']['n_train'])} películas), `TimeSeriesSplit(5)`: cada corte aprende con el
   pasado y se prueba con el período siguiente.
""")
    st.dataframe(pd.DataFrame(M["particion"]["folds"]).rename(columns={
        "fold": "Corte", "entrena": "Aprende con", "n_entrena": "Películas", "valida": "Se prueba con",
        "n_valida": "Películas "}), hide_index=True, width="stretch")
    rf_simple, rf_aj, fin_m, ridge = (modelos.loc["Random Forest"], modelos.loc[E.NOMBRE_RF], modelos.loc[E.NOMBRE_FINAL],
                                      modelos.loc["Ridge"])
    grilla = pd.DataFrame(M["grilla"]["tabla"])
    st.markdown(f"""
Si los cortes se armaran al azar (`KFold`), el Random Forest daría {S.num(rf_simple['mae_kfold_azar'], 4)} en vez de
{S.num(rf_simple['mae_cv'], 4)}: más optimista, porque aprende con películas del futuro y con secuelas cuya primera parte
quedó del otro lado.

Todo el preprocesamiento (logaritmos, imputación, escalado y one-hot) está dentro de un `Pipeline` que se ajusta en
cada corte solo con su entrenamiento. Las historias (productora, director, actor, saga) usan solo años estrictamente
anteriores al estreno. Semilla {M['entorno']['seed']} en todo.

**La comparación.** Todos con el mismo pipeline y los mismos 5 cortes. La mejora se mide **corte contra corte**: la mejora
media tiene que superar su desvío (✓).
""")
    st.dataframe(pd.DataFrame({
        "Modelo": [nombres_cortos[m] for m in orden_modelos],
        "MAE validación": [S.num(modelos.loc[m, "mae_cv"], 4) for m in orden_modelos],
        "Desvío": [S.num(modelos.loc[m, "desvio_cv"], 4) for m in orden_modelos],
        "Factor": [S.mult(modelos.loc[m, "factor_cv"]) for m in orden_modelos],
        "Mejora vs. piso · cortes": ["—"] + [
            f"{S.num(modelos.loc[m, 'mejora_vs_piso'], 4)} ({S.num(modelos.loc[m, 'desvio_mejora_vs_piso'], 4)}) · "
            f"{modelos.loc[m, 'folds_gana_vs_piso']} {'✓' if modelos.loc[m, 'supera_ruido_vs_piso'] else '✗'}"
            for m in orden_modelos[1:]],
        "Mejora vs. Ridge · cortes": ["—", "—"] + [
            f"{S.num(modelos.loc[m, 'mejora_vs_ridge'], 4)} ({S.num(modelos.loc[m, 'desvio_mejora_vs_ridge'], 4)}) · "
            f"{modelos.loc[m, 'folds_gana_vs_ridge']} {'✓' if modelos.loc[m, 'supera_ruido_vs_ridge'] else '✗'}"
            for m in orden_modelos[2:]],
        "MAE test (informativo)": [S.num(modelos.loc[m, "mae_test"], 4) for m in orden_modelos],
    }), hide_index=True, width="stretch")
    hv, fv = M["histgb_vs_final"], M["final_vs_rf"]
    st.markdown(f"""
**Por qué un promedio de tres.** Entre el Random Forest ajustado, HistGB y el promedio la diferencia es ruido (el promedio
mejora {S.num(fv['mejora media'], 4)} al Random Forest con un desvío de {S.num(fv['desvío de la mejora'], 4)}; HistGB,
{S.num(hv['mejora media'], 4)} con un desvío de {S.num(hv['desvío de la mejora'], 4)}). Elegimos el promedio por
estabilidad: es el que le gana a Ridge de forma más pareja ({S.num(fin_m['mejora_vs_ridge'], 4)} ±
{S.num(fin_m['desvio_mejora_vs_ridge'], 4)}, {fin_m['folds_gana_vs_ridge']} cortes) y no depende de un solo modelo.

**El ajuste.** Grilla sobre `min_samples_leaf` (1, 5, 10, 20, 40) y `max_features` (0,3, 0,6, 1,0) del Random Forest, con
los mismos cortes. Ganó `min_samples_leaf = {M['grilla']['mejores']['min_samples_leaf']}` y
`max_features = {S.num(M['grilla']['mejores']['max_features'], 1)}`, con 300 árboles. Entre la mejor y la peor de las
{len(grilla)} combinaciones hay {S.num(grilla['mae'].max() - grilla['mae'].min(), 4)} de MAE: el modelo es poco sensible a
estos hiperparámetros.
""")
    st.dataframe(grilla.assign(mae=grilla["mae"].map(lambda v: S.num(v, 4)),
                               desvio=grilla["desvio"].map(lambda v: S.num(v, 4))).rename(
        columns={"min_samples_leaf": "min_samples_leaf", "max_features": "max_features", "mae": "MAE", "desvio": "Desvío"}),
        hide_index=True, width="stretch", height=220)
    st.markdown(f"""
**El examen final** ({S.num(M['particion']['n_test'])} películas, mirado una vez): MAE {S.num(FIN['mae_test'], 4)}
({S.mult(FIN['factor_test'])}), contra {S.num(FIN['mae_piso_test'], 4)} ({S.mult(FIN['factor_piso_test'])}) del piso.
R² = {S.num(FIN['r2_test'], 2)}: explica una parte chica de las diferencias entre películas. El test da peor que la
validación para todos, también para el piso: {anios_test[0]}–{anios_test[1]} es más difícil (pandemia y estrenos en
streaming).

**Los errores más grandes del examen:**
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
    pop = M["popularidades"]
    st.markdown(f"""
Varios «fracasos» son estrenos de streaming con una recaudación simbólica en TMDb (*Red Notice*, *The Gray Man*: Netflix).
Son un problema de datos, no del modelo; no los sacamos porque sería tocar el test después de mirarlo.

**Las popularidades, bajo sospecha.** TMDb da la popularidad de la persona **hoy**, no la de antes del estreno. Se
controló: sin las tres popularidades el MAE de test da {S.num(pop['mae_test_sin'], 4)}, contra
{S.num(pop['mae_test_con'], 4)} con ellas. El modelo no depende de ellas.

**Por qué Streamlit y no una página HTML.** El modelo final promedia un bosque de 300 árboles, HistGB y Ridge: no hay
una fórmula corta que se pueda escribir dentro de una página. Streamlit corre Python, así que carga el mismo
`modelo.joblib` que genera `build_app.py`. La alternativa era llevar Ridge, que queda a
{S.num(ridge['mae_cv'] - FIN['mae_cv'], 4)} de MAE (menos de un desvío), pero pierde en los 5 cortes por más que el
desvío de la diferencia: preferimos mostrar el modelo que defendemos.
""")

pase("El modelo achata las puntas. <b>¿Qué hicimos después de la Entrega 3 para mejorarlo?</b>")
st.divider()

# =============================================================================================
# 4 · CÓMO LO MEJORAMOS (correcciones de la Entrega 3)
# =============================================================================================
seccion("mejora", "4 · Cómo lo mejoramos",
        f"Forzarlo no sirvió; darle más información, sí: el error pasó de {S.mult(E3['factor_test'])} a "
        f"{S.mult(FIN['factor_test'])}",
        "En la Entrega 3 el profe marcó que el modelo era demasiado conservador. Probamos empujarlo a alejarse de la "
        "mediana y empeoraba. Lo que funcionó fue sumar datos que se conocen antes del estreno: quién la dirige, quién "
        "actúa, si es secuela y cómo le fue a la saga, de qué trata y de dónde es.")

recorrido = pd.DataFrame([
    {"etapa": "Adivinar (piso)", "factor": FIN["factor_piso_test"], "rol": "Piso"},
    {"etapa": "Entrega 3 (lo que presentamos)", "factor": E3["factor_test"], "rol": "Antes"},
    {"etapa": "Modelo final, sin los datos nuevos", "factor": fac(CTX["sin"]["mae_test"]), "rol": "Antes"},
    {"etapa": "Con los datos nuevos (final)", "factor": fac(CTX["con"]["mae_test"]), "rol": "Modelo final"},
])
recorrido["uno"] = 1.0
recorrido["mejora"] = 1 - np.log10(recorrido["factor"]) / np.log10(FIN["factor_piso_test"])
recorrido["txt"] = [S.mult(f, 2) + ("" if i == 0 else f"   ({S.pct(m, 1)} menos que adivinar)")
                    for i, (f, m) in enumerate(zip(recorrido["factor"], recorrido["mejora"]))]
titulo_grafico("El recorrido del error",
               "Error típico en el mismo examen final (películas de 2017 a 2025). Menos es mejor.")
# Gráfico: error de test en cada etapa del proyecto.
#  datos: piso y final de metricas.json["final"], la Entrega 3 presentada (dato histórico) y el modelo final
#         con y sin las columnas nuevas (metricas.json["contexto"], E.con_y_sin_contexto)
#  marca: barra horizontal; mapeo: y = etapa (en orden), x = factor de error (desde ×1), color = rol
y_rec = alt.Y("etapa:N", sort=list(recorrido["etapa"]), title=None, axis=alt.Axis(labelLimit=280, labelFontSize=12.5))
br = alt.Chart(recorrido)
mostrar(alt.layer(
    br.mark_bar(height=30, cornerRadiusEnd=5).encode(
        x2="uno:Q", x=alt.X("factor:Q", scale=alt.Scale(domain=[1, 4.4]), title="Factor de error típico",
                axis=alt.Axis(values=[1, 1.5, 2, 2.5, 3, 3.5], labelExpr=S.ETIQUETA_MULTIPLO)),
        y=y_rec, color=alt.Color("rol:N", scale=alt.Scale(domain=["Piso", "Antes", "Modelo final"],
                                                          range=[S.GRIS_1, S.GRIS_2, S.MODELO]), legend=None)),
    br.mark_text(align="left", dx=8, fontSize=12.5, fontWeight="bold", color=S.TINTA).encode(
        x="factor:Q", y=y_rec, text="txt:N"),
), alto=230)
grupos = pd.DataFrame(CTX["grupos"]).sort_values("aporte", ascending=False)
grupos["lo"], grupos["hi"] = grupos["aporte"] - grupos["desvio"], grupos["aporte"] + grupos["desvio"]
grupos["txt"] = grupos["aporte"].map(lambda v: S.num(v, 4))
titulo_grafico("¿Qué dato nuevo aportó más?",
               "Cuánto empeora el modelo en la práctica si se le saca cada grupo de columnas nuevas.")
# Gráfico: aporte de cada grupo de columnas nuevas (ablación en la validación cruzada).
#  datos: metricas.json["contexto"]["grupos"] (MAE del Random Forest ajustado sin el grupo − con todas)
#  marca: barra + regla de ± un desvío entre cortes; mapeo: y = grupo, x = aumento del MAE; color violeta
y_g = alt.Y("grupo:N", sort=list(grupos["grupo"]), title=None, axis=alt.Axis(labelLimit=260, labelFontSize=12.5))
mostrar(alt.layer(
    alt.Chart(grupos).mark_bar(height=22, color=S.MODELO, cornerRadiusEnd=4).encode(
        x=alt.X("aporte:Q", title="Cuánto sube el error sin ese grupo (MAE)", axis=alt.Axis(tickCount=6)), y=y_g,
        tooltip=[alt.Tooltip("grupo:N", title="Grupo"), alt.Tooltip("txt:N", title="Aporte"),
                 alt.Tooltip("cortes:N", title="Cortes en que ayuda")]),
    alt.Chart(grupos).mark_rule(color=S.TINTA_2).encode(x="lo:Q", x2="hi:Q", y=y_g),
    alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=S.TINTA_SUAVE).encode(x="x:Q"),
), alto=230)

ext = tramos.melt(id_vars=["tramo_es"], value_vars=["real", "sin", "con"], var_name="valor", value_name="multiplo")
ext["valor"] = ext["valor"].map({"real": "Lo que pasó de verdad", "sin": "Sin los datos nuevos",
                                 "con": "Con los datos nuevos (final)"})
ext["multiplo_txt"] = ext["multiplo"].map(lambda v: S.mult(v, 2))
titulo_grafico("¿Se animó más en las puntas?",
               "El mismo gráfico de la sección 3, con el modelo sin los datos nuevos (gris) y con ellos (violeta).")
# Gráfico: lo real contra lo predicho con y sin las columnas nuevas, por tramo de resultado (test).
#  datos: metricas.json["por_tramo"] (mediana real, predicha con y sin columnas nuevas), en formato largo
#  mapeo: y = tramo, x = múltiplo (escala LOG), color = real / sin / con
#  marcas: regla de lo real a lo predicho con los datos nuevos (el error que queda) + puntos
escala_ext = alt.Scale(domain=["Lo que pasó de verdad", "Sin los datos nuevos", "Con los datos nuevos (final)"],
                       range=[S.REAL, S.GRIS_1, S.MODELO])
mostrar(alt.layer(
    alt.Chart(tramos).mark_rule(color=S.GRILLA, strokeWidth=3).encode(
        y=y_tr, x=alt.X("real:Q", scale=x_tr, axis=S.eje_multiplo("Cuántas veces recuperó lo que costó (escala log)")),
        x2="con:Q"),
    alt.Chart(pd.DataFrame({"x": [1]})).mark_rule(strokeDash=[4, 4], color=S.PISO).encode(x=alt.X("x:Q", scale=x_tr)),
    alt.Chart(ext).mark_circle(size=170, opacity=1).encode(
        y=y_tr, x=alt.X("multiplo:Q", scale=x_tr), color=alt.Color("valor:N", scale=escala_ext,
                                                                    legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip("tramo_es:N", title="Grupo"), alt.Tooltip("valor:N", title=" "),
                 alt.Tooltip("multiplo_txt:N", title="Múltiplo típico")]),
), alto=300)
st.markdown(
    f"Con los datos nuevos el modelo se acerca un poco más a lo real en las dos puntas: a las que perdieron plata pasa "
    f"de decirles {S.mult(perdida['sin'], 2)} a {S.mult(perdida['con'], 2)}, y a las que explotaron, de "
    f"{S.mult(top['sin'], 1)} a {S.mult(top['con'], 1)}. Mejora, pero los extremos siguen siendo lo más difícil.")

pc = pd.DataFrame(M["pruebas_correccion"]).iloc[1:].copy()
pc["nombre"] = pc["prueba"].map({
    "Promedio de 3 modelos (el final)": "Promediar 3 modelos (el final)",
    "Pérdida Huber (GradientBoosting)": "Pérdida Huber",
    "Estirar las predicciones ×1,5": "Estirar las predicciones ×1,5",
    "Más peso a los casos extremos": "Más peso a los extremos al aprender",
    "+ «pertenece a una saga» (trampa)": "«Pertenece a una saga» (trampa)"})
pc["resultado"] = np.select([pc["prueba"].str.contains("trampa"), pc["supera_ruido"], pc["mejora"] < 0],
                            ["Mejora, pero es trampa", "Mejora", "Empeora"], "Empata (ruido)")
pc.loc[(pc["mejora"] > 0) & ~pc["supera_ruido"] & ~pc["prueba"].str.contains("trampa"), "resultado"] = "Empata (ruido)"
pc["lo"], pc["hi"] = pc["mejora"] - pc["desvio_mejora"], pc["mejora"] + pc["desvio_mejora"]
pc["txt"] = pc["mejora"].map(lambda v: ("+" if v > 0 else "") + S.num(v, 4))
titulo_grafico("Lo que pidió probar el profe para que no achate",
               "Cada estrategia contra el Random Forest ajustado, con las mismas 5 pruebas. A la derecha del cero se "
               "equivoca menos; a la izquierda, más.")
# Gráfico: mejora de cada estrategia contra la referencia (validación cruzada temporal).
#  datos: metricas.json["pruebas_correccion"] (E.pruebas_correccion: MAE de la referencia − MAE de la prueba, por corte)
#  marcas: barra (mejora media) + regla (± un desvío entre cortes) + regla vertical en 0
#  mapeo: y = estrategia, x = mejora del MAE, color = resultado (mejora / empata / empeora / trampa)
escala_res = alt.Scale(domain=["Mejora", "Empata (ruido)", "Empeora", "Mejora, pero es trampa"],
                       range=[S.MODELO, S.GRIS_2, S.ERROR_MAS, S.GRIS_1])
y_pc = alt.Y("nombre:N", sort=list(pc["nombre"]), title=None, axis=alt.Axis(labelLimit=300, labelFontSize=12.5))
mostrar(alt.layer(
    alt.Chart(pc).mark_bar(height=22, cornerRadius=3, stroke=S.TINTA_SUAVE, strokeWidth=0.6).encode(
        x=alt.X("mejora:Q", title="Mejora del error contra la referencia (MAE). Positivo = se equivoca menos"), y=y_pc,
        color=alt.Color("resultado:N", scale=escala_res, legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("prueba:N", title="Prueba"), alt.Tooltip("que_es:N", title="Qué hace"),
                 alt.Tooltip("txt:N", title="Mejora"), alt.Tooltip("cortes_gana:N", title="Cortes en que gana")]),
    alt.Chart(pc).mark_rule(color=S.TINTA_2).encode(x="lo:Q", x2="hi:Q", y=y_pc),
    alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=S.TINTA).encode(x="x:Q"),
), alto=250)
st.markdown(
    "**Estirar** las predicciones y darles **más peso a los extremos** empeoran en las 5 pruebas: aciertan un poco más en "
    "las puntas, pero se equivocan mucho más en el medio, donde está la mayoría. **Huber** empata. **«Pertenece a una "
    "saga»** es la que más mejora, pero es trampa: TMDb arma la saga cuando sale una secuela, y la secuela sale si a la "
    "primera le fue bien. Por eso solo usamos **«es secuela»**, que sí se sabe antes del estreno.")

with como_lo_sabemos():
    st.markdown(f"""
**Las columnas nuevas** (segunda ronda, 10/11). El DAG ahora pide a TMDb, para cada película, quién la dirige, quién la
protagoniza, a qué saga pertenece y en qué lugar, sus palabras clave y los países que la producen. Con eso se arman
columnas que el modelo puede usar (todas números o sí/no; los nombres nunca entran al modelo):

| Columna | Qué es | Ejemplo |
|---|---|---|
| Historia del director y del protagonista | Mediana del múltiplo de sus películas de **años anteriores**, y cuántas tiene | Para una película de 2014, solo cuentan las suyas hasta 2013 |
| Historia de la saga | Mediana del múltiplo de las películas anteriores de la saga | Para la tercera, cómo les fue a la primera y la segunda |
| Es secuela | No es la primera de su saga | *The Dark Knight*: sí. *Batman Begins*: no |
| Palabras clave | 9 casillas sí/no: secuela, remake, libro, cómic, historia real, videojuego, navidad, musical, found footage | *Paranormal Activity*: found footage |
| Países | Cuántos la producen y si uno es EE. UU. | Una coproducción de EE. UU. y Reino Unido: 2 y sí |

Sin historia (un director debutante, la primera de una saga) se completa con la mediana y una columna marca el faltante.

**Con y sin, en números** (mismo modelo final, mismo pipeline, misma semilla; `E.con_y_sin_contexto`):

| | Validación (MAE · R²) | Examen final (MAE · factor · R²) |
|---|---|---|
| Sin las columnas nuevas | {S.num(CTX['sin']['mae_cv'], 4)} · {S.num(CTX['sin']['r2_cv'], 3)} | {S.num(CTX['sin']['mae_test'], 4)} · {S.mult(fac(CTX['sin']['mae_test']))} · {S.num(CTX['sin']['r2_test'], 3)} |
| Con las columnas nuevas | {S.num(CTX['con']['mae_cv'], 4)} · {S.num(CTX['con']['r2_cv'], 3)} | {S.num(CTX['con']['mae_test'], 4)} · {S.mult(fac(CTX['con']['mae_test']))} · {S.num(CTX['con']['r2_test'], 3)} |

En la validación, la mejora es {S.num(CTX['mejora_cv']['media'], 4)} con un desvío de {S.num(CTX['mejora_cv']['desvio'], 4)}
({CTX['mejora_cv']['cortes_gana']} cortes). En el examen final, el error baja un {S.pct(mejora_ctx_pct, 1)}.

**Las correcciones de la primera ronda** (5 y 8/11), que ya estaban antes de las columnas nuevas:
- Columnas que combinan datos que ya teníamos: director × protagonista (`power_pack`, la segunda que más pesa), el
  presupuesto contra la mediana de su género, terror barato, la historia de la productora y tres más que comparan
  popularidad con presupuesto.
- MAE, RMSE y R² en todas las comparaciones; HistGB con `loss="absolute_error"`; un solo modelo final en todo el notebook.
- `prod_hist_roi` usa solo años **estrictamente anteriores**. `budget_vs_genre_median` divide por la mediana del género
  de todo el alcance (no usa el objetivo; queda declarado).
- El test se miró más de una vez (en la Entrega 3 y al probar las columnas nuevas) y está declarado en el notebook. Las
  decisiones se tomaron con la validación cruzada, que también mejoró.

**La Entrega 3 presentada** ({S.mult(E3['factor_test'])}) es el resultado de la rama `presentacion3`, antes de cualquier
corrección. Se muestra como dato histórico.
""")
    st.dataframe(pd.DataFrame({
        "Prueba": pd.DataFrame(M["pruebas_correccion"])["prueba"],
        "Qué hace": pd.DataFrame(M["pruebas_correccion"])["que_es"],
        "MAE": pd.DataFrame(M["pruebas_correccion"])["mae_cv"].map(lambda v: S.num(v, 4)),
        "R²": pd.DataFrame(M["pruebas_correccion"])["r2_cv"].map(lambda v: S.num(v, 3)),
        "Mejora (desvío) · cortes": ["—"] + [f"{S.num(a, 4)} ({S.num(b, 4)}) · {c} {'✓' if d else '✗'}"
                                             for a, b, c, d in zip(pc["mejora"], pc["desvio_mejora"], pc["cortes_gana"],
                                                                   pc["supera_ruido"])]}),
        hide_index=True, width="stretch")

pase("Ya sabés qué mirar: el presupuesto manda, el modelo acierta en el medio y le cuestan las puntas. <b>Ahora, "
     "probalo vos.</b>")
st.divider()

# =============================================================================================
# COPA: EXPLORÁ Y PROBÁ
# =============================================================================================
seccion("copa", "Explorá y probá", "Ahora te toca a vos",
        f"Recorré las {S.num(n_alcance)} películas con lo que dijo el modelo de cada una, o inventá una y mirá qué "
        "predice.")
tab_explora, tab_proba = st.tabs(["Explorá las películas", "Probá el modelo"])

with tab_explora:
    anio_min, anio_max = int(P["release_year"].min()), int(P["release_year"].max())
    generos_orden = P["genero_es"].value_counts().index.tolist()
    idiomas_orden = P["idioma_es"].value_counts().index.tolist()

    k1, k2, k3, k4 = st.columns([2, 2, 2, 1.4])
    rango_anios = k1.slider("Años de estreno", anio_min, anio_max, (anio_min, anio_max))
    generos_sel = k2.multiselect("Géneros", generos_orden, placeholder="Todos los géneros")
    idiomas_sel = k3.multiselect("Idioma original", idiomas_orden, placeholder="Todos los idiomas")
    conjunto = k4.radio("Películas", ["Todas", "Solo test", "Solo entrenamiento"],
                        help=f"Test: {anios_test[0]}–{anios_test[1]}, el modelo nunca las vio. Entrenamiento: el modelo "
                             "las vio al aprender, así que su predicción es optimista.")
    k5, k6 = st.columns([2, 3])
    eje_y = k5.radio("Eje vertical", ["Múltiplo real", "Múltiplo predicho por el modelo"], horizontal=True)
    color_por = k6.radio("Color", ["Error del modelo", "Género"], horizontal=True)

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

    if len(vista):
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Películas en la vista", S.num(len(vista)))
        v2.metric("Recuperan lo invertido", S.pct((vista["multiplo_real"] >= 1).mean()))
        v3.metric("Múltiplo típico (real)", S.mult(vista["multiplo_real"].median(), 2))
        v4.metric("Error típico del modelo", S.mult(10 ** vista["error_abs"].mean(), 2),
                  help="10^(error medio en log10) en las películas de la vista. Si incluye películas de entrenamiento, es optimista: el modelo ya las vio.")

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
    #  transformaciones: transform_filter con los controles (años, géneros, idiomas, train/test)
    #  mapeo: x = presupuesto (LOG), y = múltiplo real o predicho (LOG), color = error del modelo (divergente) o
    #         género, tooltip = ficha corta
    #  marca: círculo chico; selección por clic sobre movie_id (apaga el resto y abre la ficha de abajo)
    sel = alt.selection_point(name="peli", fields=["movie_id"], on="click", clear="dblclick")
    puntos = alt.Chart(P).mark_circle(size=30, stroke=S.SUPERFICIE, strokeWidth=0.4).encode(
        x=alt.X("budget:Q", scale=S.escala_budget_log(), axis=S.eje_budget()),
        y=alt.Y(f"{campo_y}:Q", scale=S.escala_multiplo_log(),
                axis=S.eje_multiplo(f"{eje_y} (recaudación / presupuesto, log)")),
        color=color,
        opacity=alt.condition(sel, alt.value(0.75), alt.value(0.15)),
        tooltip=[alt.Tooltip("title:N", title="Película"), alt.Tooltip("release_year:Q", title="Año"),
                 alt.Tooltip("genero_es:N", title="Género"), alt.Tooltip("presupuesto_txt:N", title="Presupuesto"),
                 alt.Tooltip("recaudacion_txt:N", title="Recaudación"),
                 alt.Tooltip("multiplo_real_txt:N", title="Recuperó (real)"),
                 alt.Tooltip("multiplo_pred_txt:N", title="Predijo el modelo"),
                 alt.Tooltip("error_txt:N", title="Error"), alt.Tooltip("conjunto:N", title="Conjunto")],
    ).add_params(sel)
    for pr in predicados:
        puntos = puntos.transform_filter(pr)
    linea = alt.Chart(pd.DataFrame({"y": [1]})).mark_rule(strokeDash=[4, 4], color=S.PISO).encode(
        y=alt.Y("y:Q", scale=S.escala_multiplo_log()))
    texto_linea = alt.Chart(pd.DataFrame({"x": [1.2e4], "y": [1], "t": ["×1: recupera justo lo invertido"]})).mark_text(
        align="left", dy=-7, color=S.TINTA_SUAVE, fontSize=11).encode(
        y=alt.Y("y:Q", scale=S.escala_multiplo_log()), x=alt.X("x:Q", scale=S.escala_budget_log()), text="t:N")
    if len(vista) == 0:
        st.info("No hay películas con esos filtros. Las de **test** son todas de "
                f"{anios_test[0]} a {anios_test[1]}: si elegiste «Solo test», incluí esos años.")
    evento = mostrar(alt.layer(linea, texto_linea, puntos), alto=520, on_select="rerun", selection_mode="peli",
                     key="panorama")
    st.caption("Arriba de la línea punteada recuperaron más de lo que costaron; abajo perdieron plata. Pasá el mouse por "
               "un punto para ver su ficha corta; hacé clic para abrirla abajo (doble clic para soltarla).")

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
                    f"**Error:** {f['error_txt']} (tramo real: {TRAMOS_ES.get(f['tramo_roi'], f['tramo_roi'])}). "
                    f"{nota_conj}  \n"
                    f"Después del estreno (no lo usa el modelo): {S.num(f['vote_average'], 1)} de 10 con "
                    f"{S.num(f['vote_count'])} votos.")

num_r, cat_r = R["numericas"], R["categoricas"]
OTRA = "Otra (el modelo no la conoce)"
TEMAS_ES = {"kw_libro": "Basada en un libro", "kw_comic": "Cómic o superhéroes", "kw_historia_real": "Historia real",
            "kw_remake": "Remake", "kw_videojuego": "Basada en un videojuego", "kw_navidad": "Navidad",
            "kw_musical": "Musical", "kw_found_footage": "Found footage"}
CARGADAS = ["budget", "runtime", "release_year", "director_popularity", "lead_actor_popularity", "co_star_popularity"]
# Límites para recortar: todo el alcance (entrenamiento + test), o sea, lo que el modelo ya predijo en el pipeline.
# Así una película real cargada como punto de partida da lo mismo que en «Explorá».
COLS_RECORTE = E.COLS_LOG + E.COLS_NUM + ["director_hist", "actor_hist", "saga_hist", "director_n_prev", "actor_n_prev",
                                           "paises_n"]
LIMITES = {c: (float(P[c].min()), float(P[c].max())) for c in COLS_RECORTE}
ANIO_MAX_TEST = int(anios_test[1])


def opciones_cat(col):
    return cat_r[col]["categorias"]


NINGUNO = 0  # opción «ninguno» de los menús de director, protagonista y saga (los ids de TMDb empiezan en 1)


def lista_ids(clave):
    c = cat_r[clave]
    return [NINGUNO] + c["ids"], dict(zip(c["ids"], c["nombres"])), dict(zip(c["ids"], c["conteo"]))


DIR_IDS, DIR_NOM, DIR_CNT = lista_ids("director_id")
ACT_IDS, ACT_NOM, ACT_CNT = lista_ids("lead_actor_id")
SAGA_IDS, SAGA_NOM, SAGA_CNT = lista_ids("collection_id")
saga_mayor = SAGA_IDS[1] if len(SAGA_IDS) > 1 else NINGUNO   # la saga con más películas (para el ejemplo)


def en(valor, opciones, otra=OTRA):
    return valor if valor in opciones else otra


def valores_tipicos() -> dict:
    return {"pb_budget": float(num_r["budget"]["mediana"]), "pb_runtime": int(num_r["runtime"]["mediana"]),
            "pb_anio": 2025, "pb_mes": 6, "pb_dia": 4, "pb_idioma": "en",
            "pb_gen1": en("Drama", opciones_cat("primary_genre")), "pb_gen2": opciones_cat("secondary_genre")[0],
            "pb_prod1": opciones_cat("primary_production_company")[0],
            "pb_prod2": en("Unknown", opciones_cat("co_production_company")),
            "pb_pop_dir": float(num_r["director_popularity"]["mediana"]),
            "pb_pop_lead": float(num_r["lead_actor_popularity"]["mediana"]),
            "pb_pop_co": float(num_r["co_star_popularity"]["mediana"]),
            "pb_director": NINGUNO, "pb_actor": NINGUNO, "pb_secuela": False, "pb_saga": NINGUNO, "pb_temas": [],
            "pb_eeuu": True, "pb_paises": 1, "pb_real": None, "pb_kw_raw": None, "pb_paises_raw": None}


EJEMPLOS = {
    "Valores típicos": {},
    "Ejemplo: terror barato": {"pb_budget": 1_000_000.0, "pb_runtime": 88, "pb_gen1": en("Horror", opciones_cat("primary_genre")),
                               "pb_gen2": en("Thriller", opciones_cat("secondary_genre"), opciones_cat("secondary_genre")[0]),
                               "pb_mes": 10, "pb_temas": ["kw_found_footage"]},
    "Ejemplo: superproducción de saga": {"pb_budget": 200_000_000.0, "pb_runtime": 140,
                                         "pb_gen1": en("Action", opciones_cat("primary_genre")),
                                         "pb_gen2": en("Adventure", opciones_cat("secondary_genre"), opciones_cat("secondary_genre")[0]),
                                         "pb_secuela": True, "pb_saga": saga_mayor, "pb_mes": 5,
                                         "pb_temas": ["kw_comic"], "pb_pop_lead": float(num_r["lead_actor_popularity"]["p99"])},
    "Ejemplo: drama de autor": {"pb_budget": 8_000_000.0, "pb_runtime": 125, "pb_gen1": en("Drama", opciones_cat("primary_genre")),
                                "pb_mes": 11, "pb_temas": ["kw_historia_real"]},
}
MUESTRA = P.sort_values("vote_count", ascending=False).head(150)
ORIGENES = list(EJEMPLOS) + [f"{t} ({a})" for t, a in zip(MUESTRA["title"], MUESTRA["release_year"])]
ID_ORIGEN = dict(zip(ORIGENES[len(EJEMPLOS):], MUESTRA["movie_id"]))


def id_en(valor, nombres: dict):
    """El id si está en la lista del formulario (como int de Python); si no, «ninguno»."""
    if pd.isna(valor):
        return NINGUNO
    return int(valor) if int(valor) in nombres else NINGUNO


def desde_pelicula(movie_id: int) -> dict:
    f = P.loc[P["movie_id"] == movie_id].iloc[0]
    temas = [k for k in TEMAS_ES if bool(f[k])]
    return {"pb_budget": float(f["budget"]), "pb_runtime": int(f["runtime"]), "pb_anio": int(f["release_year"]),
            "pb_mes": int(f["release_month"]), "pb_dia": int(f["release_day_of_week"]),
            "pb_idioma": en(f["original_language"], opciones_cat("original_language")),
            "pb_gen1": en(f["primary_genre"], opciones_cat("primary_genre")),
            "pb_gen2": en(f["secondary_genre"], opciones_cat("secondary_genre")),
            "pb_prod1": en(f["primary_production_company"], opciones_cat("primary_production_company")),
            "pb_prod2": en(f["co_production_company"], opciones_cat("co_production_company")),
            "pb_pop_dir": float(f["director_popularity"]), "pb_pop_lead": float(f["lead_actor_popularity"]),
            "pb_pop_co": float(f["co_star_popularity"]),
            "pb_director": id_en(f["director_id"], DIR_NOM), "pb_actor": id_en(f["lead_actor_id"], ACT_NOM),
            "pb_secuela": bool(f["es_secuela"]), "pb_saga": id_en(f["collection_id"], SAGA_NOM),
            "pb_temas": temas, "pb_eeuu": bool(f["es_eeuu"]), "pb_paises": max(int(f["paises_n"]), 1),
            "pb_real": int(movie_id),
            # palabras clave y países tal cual vienen de TMDb, mientras no se toquen esos campos
            "pb_kw_raw": (f["keywords"], temas, bool(f["es_secuela"])),
            "pb_paises_raw": (f["production_countries"], bool(f["es_eeuu"]), max(int(f["paises_n"]), 1))}


def aplicar_origen():
    elegido = st.session_state["pb_origen"]
    st.session_state.update(valores_tipicos())
    if elegido in EJEMPLOS:
        st.session_state.update(EJEMPLOS[elegido])
    else:
        st.session_state.update(desde_pelicula(int(ID_ORIGEN[elegido])))


if "pb_budget" not in st.session_state:
    st.session_state.update(valores_tipicos())


def armar_casos(base: dict, budgets) -> pd.DataFrame:
    """Una fila por presupuesto, con las columnas derivadas y el recorte a lo que vio el modelo."""
    return armar_filas([{**base, "budget": float(b)} for b in budgets])


def armar_filas(casos: list[dict]):
    """Una fila por caso, con las columnas derivadas (features.py) y el recorte a lo que vio el modelo."""
    filas = F.agregar_features(pd.DataFrame(casos), ref)
    recortadas = set()
    for c in COLS_RECORTE:
        if c == "release_year" and (filas[c] <= ANIO_MAX_TEST).all():
            continue  # años del test: el modelo ya los predijo así en el pipeline
        lo, hi = LIMITES[c]
        fuera = filas[c].notna() & ((filas[c] < lo) | (filas[c] > hi))
        if fuera.any():
            recortadas.add(c)
            filas.loc[fuera, c] = filas.loc[fuera, c].clip(lo, hi)
    return filas, recortadas


@st.fragment
def proba_el_modelo():
    # Solo este bloque se vuelve a correr al tocar un control: la respuesta es instantánea.
    o1, o2 = st.columns([1.6, 3])
    o1.selectbox("Arrancá desde…", ORIGENES, key="pb_origen", on_change=aplicar_origen,
                 help="Elegí un ejemplo o una película real y se completan todos los datos. Después cambiá lo que quieras.")
    real = st.session_state.get("pb_real")
    o2.markdown("<p class='grafico-bajada' style='margin-top:1.9rem'>Elegí un punto de partida y cambiá solo lo que quieras "
                "probar: el resultado se actualiza al instante, a la derecha. "
                + ("Partiste de una película real: el punto blanco es lo que pasó de verdad." if real else "")
                + "</p>", unsafe_allow_html=True)

    izq, der = st.columns([1, 1.35], gap="large")
    with izq:
        st.markdown("**Lo que más pesa**")
        st.number_input("Presupuesto (USD)", min_value=10_000.0, max_value=5_000_000_000.0, step=1_000_000.0,
                        format="%.0f", key="pb_budget",
                        help=f"El modelo vio de {S.usd(num_r['budget']['min'])} a {S.usd(num_r['budget']['max'])}.")
        a1, a2 = st.columns(2)
        a1.selectbox("Género principal", opciones_cat("primary_genre") + [OTRA], key="pb_gen1",
                     format_func=lambda g: g if g == OTRA else S.genero_es(g))
        a2.selectbox("Mes de estreno", list(range(1, 13)), key="pb_mes", format_func=lambda m: S.MESES[m - 1])
        s1, s2 = st.columns([1, 2])
        s1.checkbox("Es una secuela", key="pb_secuela")
        s2.selectbox("Saga", SAGA_IDS, key="pb_saga", format_func=lambda i: "Ninguna u otra" if i == NINGUNO
                     else f"{SAGA_NOM[i]} ({SAGA_CNT[i]})")
        d1, d2 = st.columns(2)
        d1.selectbox("Director", DIR_IDS, key="pb_director", format_func=lambda i: "Otro o debutante" if i == NINGUNO
                     else f"{DIR_NOM[i]} ({DIR_CNT[i]})")
        d2.selectbox("Protagonista", ACT_IDS, key="pb_actor", format_func=lambda i: "Otro o sin historia" if i == NINGUNO
                     else f"{ACT_NOM[i]} ({ACT_CNT[i]})")
        st.multiselect("De qué se trata", list(TEMAS_ES), key="pb_temas", format_func=TEMAS_ES.get,
                       placeholder="Ninguno de estos")
        with st.expander("Más datos (opcional)"):
            b1, b2, b3 = st.columns(3)
            b1.number_input("Duración (min)", min_value=1, max_value=600, step=1, key="pb_runtime")
            b2.number_input("Año", min_value=1900, max_value=2040, step=1, key="pb_anio")
            b3.selectbox("Día", list(range(7)), key="pb_dia", format_func=lambda d: S.DIAS[d])
            c1, c2 = st.columns(2)
            c1.selectbox("Idioma original", opciones_cat("original_language") + [OTRA], key="pb_idioma",
                         format_func=lambda c: c if c == OTRA else S.idioma_es(c))
            c2.selectbox("Género secundario", opciones_cat("secondary_genre") + [OTRA], key="pb_gen2",
                         format_func=lambda g: g if g == OTRA else S.genero_es(g))
            c1.selectbox("Productora principal", opciones_cat("primary_production_company") + [OTRA], key="pb_prod1")
            c2.selectbox("Coproductora", opciones_cat("co_production_company") + [OTRA], key="pb_prod2",
                         format_func=lambda c: "Sin coproductora" if c == "Unknown" else c)
            q1, q2, q3 = st.columns(3)
            q1.number_input("Pop. director", min_value=0.0, max_value=1000.0, step=0.1, format="%.2f", key="pb_pop_dir")
            q2.number_input("Pop. protag.", min_value=0.0, max_value=1000.0, step=0.1, format="%.2f", key="pb_pop_lead")
            q3.number_input("Pop. coprotag.", min_value=0.0, max_value=1000.0, step=0.1, format="%.2f", key="pb_pop_co")
            e1, e2 = st.columns(2)
            e1.checkbox("Producida en EE. UU.", key="pb_eeuu")
            e2.number_input("Países que la producen", min_value=1, max_value=10, step=1, key="pb_paises")
            st.caption("Popularidad de TMDb (la mediana del entrenamiento es "
                       f"{S.num(num_r['director_popularity']['mediana'], 1)} para el director y "
                       f"{S.num(num_r['lead_actor_popularity']['mediana'], 1)} para el protagonista).")

    ss = st.session_state
    nada = lambda v: None if v == NINGUNO else v  # noqa: E731
    secuela, saga = ss["pb_secuela"], nada(ss["pb_saga"])
    otra = lambda v: "__otra__" if v == OTRA else v  # noqa: E731
    kw_raw, pa_raw = ss.get("pb_kw_raw"), ss.get("pb_paises_raw")
    if kw_raw and list(ss["pb_temas"]) == list(kw_raw[1]) and secuela == kw_raw[2]:
        palabras = kw_raw[0]
    else:
        palabras = "|".join([F.GRUPOS_KEYWORDS[g][0] for g in ss["pb_temas"]] + (["sequel"] if secuela else []))
    if pa_raw and ss["pb_eeuu"] == pa_raw[1] and int(ss["pb_paises"]) == pa_raw[2]:
        paises = pa_raw[0]
    else:
        paises = "|".join((["US"] if ss["pb_eeuu"] else []) +
                          [f"P{i}" for i in range(int(ss["pb_paises"]) - (1 if ss["pb_eeuu"] else 0))])
    base = {"movie_id": -1, "title": "Tu película", "runtime": ss["pb_runtime"], "release_year": int(ss["pb_anio"]),
            "release_month": int(ss["pb_mes"]), "release_day_of_week": int(ss["pb_dia"]),
            "original_language": otra(ss["pb_idioma"]), "primary_genre": otra(ss["pb_gen1"]),
            "secondary_genre": otra(ss["pb_gen2"]), "primary_production_company": otra(ss["pb_prod1"]),
            "co_production_company": otra(ss["pb_prod2"]), "director_popularity": ss["pb_pop_dir"],
            "lead_actor_popularity": ss["pb_pop_lead"], "co_star_popularity": ss["pb_pop_co"],
            "director_id": nada(ss["pb_director"]), "lead_actor_id": nada(ss["pb_actor"]),
            # la saga solo cuenta si es secuela: la primera de una saga no tiene películas anteriores
            "collection_id": saga if secuela else None, "saga_position": 1 if secuela else 0,
            "keywords": palabras, "production_countries": paises}
    budget = float(ss["pb_budget"])
    caso, recortadas = armar_casos(base, [budget])
    multiplo = float(10 ** modelo.predict(caso[E.FEATURES])[0])
    factor = FIN["factor_test"]
    bajo, alto = multiplo / factor, multiplo * factor

    # Avisos: lo que el modelo nunca vio
    avisos, notas = [], []
    for c in CARGADAS:
        v, r = float(base.get(c, budget) if c != "budget" else budget), num_r[c]
        if c == "release_year":
            if v > ANIO_MAX_TEST:
                avisos.append(f"El año {int(v)} es posterior a todo lo que vio el modelo: responde como si fuera "
                              f"{ANIO_MAX_TEST}.")
            elif v < r["min"]:
                avisos.append(f"El año {int(v)} es anterior a todo lo que vio el modelo.")
            continue
        if v < r["min"] or v > r["max"]:
            nombre = {"budget": "El presupuesto", "runtime": "La duración", "director_popularity": "La popularidad del director",
                      "lead_actor_popularity": "La popularidad del protagonista",
                      "co_star_popularity": "La popularidad del coprotagonista"}[c]
            f_ = S.usd if c == "budget" else (lambda x: S.num(x, 1))
            avisos.append(f"{nombre} ({f_(v)}) está fuera de lo que vio el modelo (de {f_(r['min'])} a {f_(r['max'])}): "
                          "usa el extremo más cercano.")
    extra = sorted(VAR_ES.get(c, c).lower() for c in recortadas if c not in CARGADAS)
    if extra:
        avisos.append("Hay datos más extremos que cualquier película que vio (" + ", ".join(extra) +
                      "): usa los extremos que sí vio.")
    for clave, nombre in [("pb_idioma", "Idioma"), ("pb_gen1", "Género principal"), ("pb_gen2", "Género secundario"),
                          ("pb_prod1", "Productora"), ("pb_prod2", "Coproductora")]:
        if ss[clave] == OTRA:
            notas.append(f"{nombre}: el modelo no conoce ese valor y lo agrupa con «otras».")
    if saga is not None and not secuela:
        notas.append("Elegiste una saga sin marcar «Es una secuela»: si es la primera, no hay películas anteriores y el "
                     "modelo no usa la saga.")

    with der:
        st.markdown("<div class='pb-pegajoso'></div>", unsafe_allow_html=True)
        verbo = "Recuperaría" if multiplo >= 1 else "Recuperaría solo"
        st.markdown(f"<p class='kicker' style='margin-top:.2rem'>Lo que predice el modelo</p>"
                    f"<p class='titulo-seccion' style='margin-bottom:.2rem'>{verbo} <span style='color:{S.DORADO}'>"
                    f"{S.mult(multiplo)}</span> lo invertido</p>"
                    f"<p class='grafico-bajada'>Recaudación de unos <b>{S.usd(budget * multiplo)}</b> · rango típico "
                    f"{S.mult(bajo)} a {S.mult(alto)} · "
                    + ("ni lo más bajo pierde plata" if bajo >= 1 else "ni lo más alto recupera lo invertido" if alto < 1
                       else "puede ganar o perder plata") + "</p>", unsafe_allow_html=True)
        fila_real = P.loc[P["movie_id"] == real].iloc[0] if real else None
        if fila_real is not None:
            st.markdown(f"<p class='grafico-bajada'>De verdad, <i>{fila_real['title']}</i> recuperó "
                        f"<b>{S.mult(fila_real['multiplo_real'])}</b>"
                        + (" (el modelo la vio al aprender)." if fila_real["conjunto"] == "entrenamiento"
                           else " (el modelo nunca la vio).") + "</p>", unsafe_allow_html=True)

        # Gráfico: tu película sobre el panorama.
        #  datos: predicciones.parquet (gris = lo que recuperaron de verdad) + tu caso + (si partiste de una real) lo real
        #  mapeo: x = presupuesto (LOG), y = múltiplo (LOG), las mismas escalas que «Explorá»
        #  marcas: círculos grises; regla dorada = rango típico; rombo dorado = predicción; círculo blanco = lo real
        fondo = alt.Chart(P[["budget", "multiplo_real"]]).mark_circle(size=12, color=S.GRIS_1, opacity=0.4).encode(
            x=alt.X("budget:Q", scale=S.escala_budget_log(), axis=S.eje_budget()),
            y=alt.Y("multiplo_real:Q", scale=S.escala_multiplo_log(), axis=S.eje_multiplo("Recaudación / presupuesto (log)")))
        tu = pd.DataFrame({"budget": [budget], "multiplo": [multiplo], "bajo": [bajo], "alto": [alto],
                           "t": [f"{S.mult(multiplo)}"], "rango": [f"{S.mult(bajo)} a {S.mult(alto)}"]})
        capas = [fondo,
                 alt.Chart(pd.DataFrame({"y": [1]})).mark_rule(strokeDash=[4, 4], color=S.TINTA_SUAVE).encode(
                     y=alt.Y("y:Q", scale=S.escala_multiplo_log())),
                 alt.Chart(tu).mark_rule(color=S.DORADO, strokeWidth=3).encode(
                     x="budget:Q", y=alt.Y("bajo:Q", scale=S.escala_multiplo_log()), y2="alto:Q"),
                 alt.Chart(tu).mark_point(shape="diamond", filled=True, size=420, color=S.DORADO, stroke=S.SUPERFICIE,
                                          strokeWidth=1.5).encode(
                     x="budget:Q", y=alt.Y("multiplo:Q", scale=S.escala_multiplo_log()),
                     tooltip=[alt.Tooltip("t:N", title="Predicción"), alt.Tooltip("rango:N", title="Rango típico")]),
                 alt.Chart(tu).mark_text(align="left", dx=14, fontSize=15, fontWeight="bold", color=S.DORADO).encode(
                     x="budget:Q", y=alt.Y("multiplo:Q", scale=S.escala_multiplo_log()), text="t:N")]
        if fila_real is not None:
            rr = pd.DataFrame({"budget": [fila_real["budget"]], "m": [fila_real["multiplo_real"]],
                               "t": [f"real {S.mult(fila_real['multiplo_real'])}"]})
            capas += [alt.Chart(rr).mark_circle(size=200, color=S.REAL, opacity=1).encode(
                          x="budget:Q", y=alt.Y("m:Q", scale=S.escala_multiplo_log())),
                      alt.Chart(rr).mark_text(align="right", dx=-12, fontSize=13, fontWeight="bold", color=S.REAL).encode(
                          x="budget:Q", y=alt.Y("m:Q", scale=S.escala_multiplo_log()), text="t:N")]
        mostrar(alt.layer(*capas), alto=300)

        # Gráfico: ¿y si cambia el presupuesto? (la misma película con 30 presupuestos distintos)
        #  datos: 30 predicciones del modelo, cambiando solo el presupuesto (las columnas derivadas se recalculan)
        #  mapeo: x = presupuesto (LOG), y = múltiplo predicho (LOG); marcas: línea violeta + rombo dorado en el actual
        budgets = np.logspace(np.log10(max(num_r["budget"]["min"], 1e5)), np.log10(num_r["budget"]["max"]), 30)
        curva_x, _ = armar_casos(base, budgets)
        curva = pd.DataFrame({"budget": budgets, "multiplo": 10 ** modelo.predict(curva_x[E.FEATURES])})
        curva["t"] = curva["multiplo"].map(lambda v: S.mult(v, 2))
        curva["b"] = curva["budget"].map(S.usd_corto)
        st.markdown("<p class='grafico-titulo' style='margin-top:.4rem'>¿Y si cambia el presupuesto?</p>"
                    "<p class='grafico-bajada'>La misma película con otros presupuestos, según el modelo.</p>",
                    unsafe_allow_html=True)
        mostrar(alt.layer(
            alt.Chart(curva).mark_line(color=S.MODELO, strokeWidth=3).encode(
                x=alt.X("budget:Q", scale=alt.Scale(type="log"), axis=S.eje_budget()),
                y=alt.Y("multiplo:Q", scale=alt.Scale(type="log", zero=False),
                        axis=alt.Axis(title="Múltiplo predicho (log)", labelExpr=S.ETIQUETA_MULTIPLO,
                                      values=[0.1, 0.3, 1, 3, 10, 30, 100, 300])),
                tooltip=[alt.Tooltip("b:N", title="Presupuesto"), alt.Tooltip("t:N", title="Predicción")]),
            alt.Chart(tu).mark_point(shape="diamond", filled=True, size=260, color=S.DORADO).encode(
                x=alt.X("budget:Q", scale=alt.Scale(type="log")), y=alt.Y("multiplo:Q", scale=alt.Scale(type="log"))),
        ), alto=150)
        for a in avisos:
            st.warning(a)
        for n_ in notas:
            st.caption("ℹ️ " + n_)

    p_izq, p_der = st.columns(2, gap="large")
    with p_izq:
        # Películas parecidas: mismo género principal y misma condición de secuela (si hay suficientes), las más
        # cercanas en presupuesto (log10) y año. Es una referencia para leer la incertidumbre, no es el modelo.
        cand = P[P["movie_id"] != (real or -1)]
        gen = base["primary_genre"]
        filtro = (cand["primary_genre"] == gen) & (cand["es_secuela"] == int(secuela))
        if filtro.sum() >= 5:
            cand = cand[filtro]
        elif (cand["primary_genre"] == gen).sum() >= 5:
            cand = cand[cand["primary_genre"] == gen]
        dist_p = (np.log10(cand["budget"]) - np.log10(budget)).abs() + 0.02 * (cand["release_year"] - base["release_year"]).abs()
        par = cand.loc[dist_p.nsmallest(5).index]
        med_par = float(par["multiplo_real"].median())
        st.markdown("<p class='grafico-titulo'>Películas parecidas a la tuya</p>"
                    "<p class='grafico-bajada'>Del mismo género" + (" y también secuelas" if secuela else "")
                    + ", con presupuesto y año parecidos. Lo que les pasó de verdad muestra por qué el margen es grande.</p>",
                    unsafe_allow_html=True)
        filas_html = "".join(
            f"<div style='display:flex;justify-content:space-between;gap:.8rem;padding:.45rem .2rem;"
            f"border-bottom:1px solid {S.GRILLA}'><span style='color:{S.TINTA}'>{r.title} "
            f"<span style='color:{S.TINTA_SUAVE}'>({r.release_year} · {S.usd_corto(r.budget)})</span></span>"
            f"<b style='color:{S.REAL if r.multiplo_real >= 1 else S.ERROR_MAS}'>{S.mult(r.multiplo_real)}</b></div>"
            for r in par.itertuples())
        st.markdown(f"<div style='background:{S.PANEL};border:1px solid {S.GRILLA};border-radius:12px;padding:.4rem .9rem'>"
                    f"{filas_html}<div style='padding:.55rem .2rem 0 .2rem;color:{S.TINTA_2};font-size:.92rem'>"
                    f"La del medio recuperó <b style='color:{S.TINTA}'>{S.mult(med_par)}</b>; el modelo dice "
                    f"<b style='color:{S.DORADO}'>{S.mult(multiplo)}</b> para la tuya.</div></div>", unsafe_allow_html=True)

    with p_der:
        # Qué mueve la predicción: la misma película cambiando un solo dato por vez.
        variantes = [("Con el doble de presupuesto", {}, budget * 2), ("Con la mitad de presupuesto", {}, budget / 2)]
        if secuela:
            variantes.append(("Si no fuera secuela", {"collection_id": None, "saga_position": 0,
                                                       "keywords": base["keywords"].replace("sequel", "")}, budget))
        else:
            variantes.append(("Si fuera secuela", {"saga_position": 1, "keywords": (base["keywords"] + "|sequel").strip("|")},
                              budget))
        for g, nombre_g in [("Horror", "de terror"), ("Action", "de acción"), ("Comedy", "comedia")]:
            if base["primary_genre"] != g and g in opciones_cat("primary_genre"):
                variantes.append((f"Si fuera {nombre_g}", {"primary_genre": g}, budget))
        if base["release_month"] != 12:
            variantes.append(("Si se estrenara en diciembre", {"release_month": 12}, budget))
        if base["release_month"] != 7:
            variantes.append(("Si se estrenara en julio", {"release_month": 7}, budget))
        variantes.append(("Con un protagonista muy conocido",
                          {"lead_actor_popularity": float(num_r["lead_actor_popularity"]["p99"])}, budget))
        filas_v, _ = armar_filas([{**base, **cambio, "budget": float(b)} for _, cambio, b in variantes])
        pred_v = 10 ** modelo.predict(filas_v[E.FEATURES])
        mov = pd.DataFrame({"cambio": [v[0] for v in variantes], "multiplo": pred_v})
        mov["razon"] = mov["multiplo"] / multiplo
        mov["log"] = np.log10(mov["razon"])
        mov["sentido"] = np.where(mov["razon"] >= 1, "Sube", "Baja")
        mov["txt"] = [f"{S.mult(m)} ({'+' if r >= 1 else '−'}{S.pct(abs(r - 1))})" for m, r in zip(mov["multiplo"], mov["razon"])]
        mov = mov.sort_values("log", ascending=False)
        st.markdown("<p class='grafico-titulo'>¿Qué mueve la predicción?</p>"
                    f"<p class='grafico-bajada'>La misma película cambiando un solo dato. Hoy el modelo dice "
                    f"{S.mult(multiplo)}.</p>", unsafe_allow_html=True)
        # Gráfico: sensibilidad de la predicción a un cambio por vez.
        #  datos: predicciones del modelo para variantes de tu película (una columna cambiada por fila)
        #  mapeo: y = cambio (ordenado de lo que más sube a lo que más baja), x = log10(predicción variante / actual)
        #         (escala simétrica alrededor de 0 = no cambia), color = sube (verde) o baja (gris)
        #  marcas: barra horizontal + texto con el nuevo múltiplo y el cambio en %
        lim = max(0.15, float(mov["log"].abs().max()) * 2.6)
        y_mv = alt.Y("cambio:N", sort=list(mov["cambio"]), title=None, axis=alt.Axis(labelLimit=260, labelFontSize=12))
        mostrar(alt.layer(
            alt.Chart(mov).mark_bar(height=18, cornerRadius=3).encode(
                x=alt.X("log:Q", scale=alt.Scale(domain=[-lim, lim]),
                        axis=alt.Axis(title="◀ baja · sube ▶", grid=False, ticks=False, labels=False)),
                y=y_mv, color=alt.Color("sentido:N", scale=alt.Scale(domain=["Sube", "Baja"], range=["#47b47d", S.GRIS_1]),
                                        legend=None),
                tooltip=[alt.Tooltip("cambio:N", title="Cambio"), alt.Tooltip("txt:N", title="El modelo diría")]),
            alt.Chart(mov).mark_text(align="left", dx=6, fontSize=11.5, color=S.TINTA).encode(
                x=alt.X("log:Q", scale=alt.Scale(domain=[-lim, lim])), y=y_mv, text="txt:N",
                ).transform_filter(alt.datum.log >= 0),
            alt.Chart(mov).mark_text(align="right", dx=-6, fontSize=11.5, color=S.TINTA).encode(
                x=alt.X("log:Q", scale=alt.Scale(domain=[-lim, lim])), y=y_mv, text="txt:N",
                ).transform_filter(alt.datum.log < 0),
            alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=S.TINTA_SUAVE).encode(x="x:Q"),
        ), alto=40 + 30 * len(mov))


with tab_proba:
    proba_el_modelo()


st.divider()
st.caption("Proyecto integrador de Ciencia de Datos · UTN FRM 2026 · Grupo 5K10-06 · Datos: TMDb (este producto usa la "
           "API de TMDb pero no está avalado ni certificado por TMDb). Los números salen de `python build_app.py`.")
