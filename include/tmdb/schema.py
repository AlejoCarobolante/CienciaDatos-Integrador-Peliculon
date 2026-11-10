# Bloques del esquema, en el orden en que salen al CSV

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
    "revenue",
    "runtime",
    "original_language",
    "primary_genre",
    "secondary_genre",
    "primary_production_company",
    "co_production_company",
]

TALENTO = [
    "director_popularity",
    "lead_actor_popularity",
    "co_star_popularity",
]

VALORACION = [
    "vote_average",
    "vote_count",
]

# Contexto conocido antes del estreno (agregado en la Entrega 4): quién la hace,
# si es parte de una saga, de qué trata y de dónde es.
CONTEXTO = [
    "director_id",
    "director_name",
    "lead_actor_id",
    "lead_actor_name",
    "collection_id",
    "collection_name",
    "saga_position",
    "production_countries",
    "keywords",
]

COLUMNS = IDENTIDAD + TEMPORAL + PRODUCCION + TALENTO + VALORACION + CONTEXTO

# Tipos para validar y castear al final

ENTEROS = [
    "movie_id",
    "release_year",
    "release_month",
    "release_day_of_week",
    "runtime",
    "vote_count",
    "director_id",
    "lead_actor_id",
    "collection_id",
    "saga_position",
]

FLOTANTES = [
    "budget",
    "revenue",
    "director_popularity",
    "lead_actor_popularity",
    "co_star_popularity",
    "vote_average",
]

TEXTO = [
    "title",
    "original_language",
    "primary_genre",
    "secondary_genre",
    "primary_production_company",
    "co_production_company",
]

# Texto que puede venir vacío (se completa con "" en vez de "Unknown")
TEXTO_LIBRE = [
    "director_name",
    "lead_actor_name",
    "collection_name",
    "production_countries",
    "keywords",
]

# Columnas sin las cuales la fila no sirve para el modelado
OBLIGATORIAS = [
    "movie_id",
    "title",
    "budget",
    "revenue",
    "runtime",
    "vote_average",
    "vote_count",
]