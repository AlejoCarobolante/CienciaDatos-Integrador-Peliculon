"""Colores, escalas y formatos de toda la app, definidos en un solo lugar.

Regla: un color significa lo mismo en todos los gráficos (tallo y copa).
- Roles del modelo: violeta = el modelo final y lo que predice; tinta gris oscura = el valor real;
  gris medio punteado = el piso (adivinar la mediana); gris claro = los otros modelos.
- Error del modelo (divergente): azul = predijo de menos (éxito no anticipado),
  rojo = predijo de más (fracaso no anticipado), gris = acertó.
- Géneros: solo tres tienen color (los que pasan el control de daltonismo entre todos los pares
  de un scatter); el resto va en gris como «Otros».
Paleta validada con el script de la guía de visualización (modo claro, superficie #fcfcfb).
"""
from __future__ import annotations

import altair as alt

# --- Superficie y tintas ----------------------------------------------------------------------
# Tema oscuro de sala de cine: fondo casi negro, tinta clara, acento dorado para los casos famosos.
SUPERFICIE = "#0f0e17"
PANEL = "#1a1926"       # tarjetas y paneles
TINTA = "#f4f2ec"
TINTA_2 = "#c3bfb3"
TINTA_SUAVE = "#8c8879"
GRILLA = "#262433"
EJE = "#4a4758"
DORADO = "#ffc857"      # anotaciones: películas famosas, la película del usuario
GRIS_1, GRIS_2, GRIS_3 = "#6e6b7b", "#8f8b9c", "#b6b2c2"  # secundarios (de oscuro a claro)
FUENTE = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"

# --- Roles del modelo -------------------------------------------------------------------------
MODELO = "#a594ff"      # modelo final / predicho
REAL = "#e9e5d8"        # valor real
PISO = TINTA_SUAVE      # piso: mediana
OTROS_MODELOS = EJE     # Ridge, RF sin ajustar, HistGB

# --- Error del modelo (divergente azul <-> rojo, gris al medio) ------------------------------
ERROR_MENOS = "#4ea8ff"  # predijo de menos: le fue mejor de lo anticipado
ERROR_MAS = "#ff6b6b"    # predijo de más: le fue peor de lo anticipado
ERROR_CERO = "#5a5768"

# --- Géneros ----------------------------------------------------------------------------------
GENERO_ES = {
    "Action": "Acción", "Adventure": "Aventura", "Animation": "Animación", "Comedy": "Comedia",
    "Crime": "Crimen", "Documentary": "Documental", "Drama": "Drama", "Family": "Familia",
    "Fantasy": "Fantasía", "History": "Historia", "Horror": "Terror", "Music": "Música",
    "Mystery": "Misterio", "Romance": "Romance", "Science Fiction": "Ciencia ficción",
    "TV Movie": "Película de TV", "Thriller": "Suspenso", "War": "Bélica", "Western": "Western",
    "Unknown": "Sin segundo género", "None": "Sin segundo género",
}
# Tres géneros con color propio: tres regímenes de presupuesto distintos (Terror barato,
# Drama medio, Acción caro). Amarillo, verde y magenta pasan el control entre todos los pares.
GENERO_COLOR = {"Terror": "#ffb000", "Drama": "#3ddc84", "Acción": "#ff7eb6"}
GENERO_OTROS = "Otros géneros"
GENERO_OTROS_COLOR = "#55526a"

IDIOMA_ES = {
    "en": "Inglés", "fr": "Francés", "hi": "Hindi", "ja": "Japonés", "zh": "Chino (mandarín)",
    "es": "Español", "ko": "Coreano", "it": "Italiano", "ru": "Ruso", "cn": "Chino (cantonés)",
    "de": "Alemán", "te": "Telugu", "ta": "Tamil", "da": "Danés", "sv": "Sueco", "pt": "Portugués",
    "no": "Noruego", "nl": "Neerlandés", "th": "Tailandés", "fa": "Persa", "pl": "Polaco",
    "id": "Indonesio", "tr": "Turco", "ml": "Malayalam", "kn": "Canarés", "he": "Hebreo",
    "fi": "Finlandés", "ar": "Árabe", "cs": "Checo", "hu": "Húngaro", "el": "Griego",
}
MESES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto",
         "Septiembre", "Octubre", "Noviembre", "Diciembre"]
DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]  # weekday(): lunes = 0


def genero_es(g: str) -> str:
    return GENERO_ES.get(g, g)


def genero_grupo(g: str) -> str:
    """Género para el color: los tres con color propio o «Otros géneros»."""
    es = genero_es(g)
    return es if es in GENERO_COLOR else GENERO_OTROS


def idioma_es(c: str) -> str:
    return f"{IDIOMA_ES.get(c, c)} ({c})"


# --- Escalas compartidas ----------------------------------------------------------------------
# Dominios fijos: el mismo punto cae en el mismo lugar en todos los gráficos.
DOMINIO_BUDGET = [10_000, 1_000_000_000]
DOMINIO_MULTIPLO = [0.0005, 20_000]


def escala_genero() -> alt.Scale:
    return alt.Scale(domain=list(GENERO_COLOR) + [GENERO_OTROS],
                     range=list(GENERO_COLOR.values()) + [GENERO_OTROS_COLOR])


def escala_error() -> alt.Scale:
    # error_log = log10(predicho) - log10(real): -1 = predijo 10 veces menos, +1 = 10 veces más
    return alt.Scale(domain=[-1, 0, 1], range=[ERROR_MENOS, ERROR_CERO, ERROR_MAS], clamp=True)


def escala_budget_log() -> alt.Scale:
    return alt.Scale(type="log", domain=DOMINIO_BUDGET, clamp=True, nice=False)


def escala_multiplo_log() -> alt.Scale:
    return alt.Scale(type="log", domain=DOMINIO_MULTIPLO, clamp=True, nice=False)


# Etiquetas de eje en formato rioplatense (coma decimal), escritas como expresiones de Vega
ETIQUETA_USD = ("datum.value >= 1e9 ? replace(format(datum.value / 1e9, '.0f'), '.', ',') + ' mil M' : "
                "datum.value >= 1e6 ? replace(format(datum.value / 1e6, '.0f'), '.', ',') + ' M' : "
                "replace(format(datum.value / 1e3, '.0f'), '.', ',') + ' mil'")
ETIQUETA_MULTIPLO = "'×' + replace(format(datum.value, '~g'), '.', ',')"


def eje_budget(titulo: str = "Presupuesto (USD, escala log)") -> alt.Axis:
    return alt.Axis(title=titulo, values=[1e4, 1e5, 1e6, 1e7, 1e8, 1e9], labelExpr=ETIQUETA_USD)


def eje_multiplo(titulo: str) -> alt.Axis:
    return alt.Axis(title=titulo, values=[0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
                    labelExpr=ETIQUETA_MULTIPLO)


def configurar(chart: alt.TopLevelMixin, alto: int | None = None) -> alt.TopLevelMixin:
    """Tema común: fuente, ejes y grilla recesivos, sin borde de vista."""
    if alto is not None:
        chart = chart.properties(height=alto)
    return (chart
            .configure(font=FUENTE, background=SUPERFICIE)
            .configure_view(stroke=None)
            .configure_axis(labelColor=TINTA_2, titleColor=TINTA_2, gridColor=GRILLA, domainColor=EJE,
                            tickColor=EJE, labelFontSize=12, titleFontSize=12, titleFontWeight="normal")
            # Ejes numéricos sin labelExpr propio: coma decimal y punto de miles (d3 los da al revés)
            .configure_axisQuantitative(labelExpr="replace(replace(replace(datum.label, regexp(',', 'g'), '#'), "
                                                  "regexp('\\\\.', 'g'), ','), regexp('#', 'g'), '.')")
            .configure_legend(labelColor=TINTA_2, titleColor=TINTA_2, labelFontSize=12, titleFontSize=12,
                              orient="top")
            .configure_title(color=TINTA, fontSize=14, anchor="start")
            .configure_text(font=FUENTE))


# --- Formatos de texto (rioplatense) ----------------------------------------------------------
def num(x: float, dec: int = 0) -> str:
    s = f"{x:,.{dec}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def pct(x: float, dec: int = 0) -> str:
    return f"{num(100 * x, dec)}%"


def mult(x: float, dec: int = 2) -> str:
    return f"×{num(x, dec)}"


def usd_corto(x: float) -> str:
    """Para métricas grandes: USD 137,4 M."""
    if x >= 1e9:
        return f"USD {num(x / 1e9, 2)} mil M"
    if x >= 1e6:
        return f"USD {num(x / 1e6, 1)} M"
    return f"USD {num(x / 1e3, 0)} mil"


def usd(x: float) -> str:
    if x >= 1e9:
        return f"USD {num(x / 1e9, 2)} mil millones"
    if x >= 1e6:
        return f"USD {num(x / 1e6, 1)} millones"
    return f"USD {num(x, 0)}"
