# bloques del esquema, en el orden en que salen al CSV

IDENTIDAD = [
    "movie_id",
    "title",
]

TEMPORAL = [
    "release_year",
    "release_month",
    "release_day_of_week",
]

PRODUCCION = [
    "budget",
    "runtime",
    "original_language",
    "primary_genre",
    "primary_production_company",
]

TALENTO = [
    "director_popularity",
    "lead_actor_popularity",
]

VALORACION = [
    "vote_average",
    "vote_count",
]

COLUMNS = IDENTIDAD + TEMPORAL + PRODUCCION + TALENTO + VALORACION

# tipos, para validar y para castear al final

ENTEROS = [
    "movie_id",
    "release_year",
    "release_month",
    "release_day_of_week",
    "runtime",
    "vote_count",
]

FLOTANTES = [
    "budget",
    "director_popularity",
    "lead_actor_popularity",
    "vote_average",
]

TEXTO = [
    "title",
    "original_language",
    "primary_genre",
    "primary_production_company",
]

# Columnas sin las cuales la fila no sirve para el modelado.
OBLIGATORIAS = [
    "movie_id",
    "title",
    "budget",
    "runtime",
    "vote_average",
    "vote_count",
]