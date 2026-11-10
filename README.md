# Peliculón · ¿Cuántas veces recupera una película lo que costó?

Proyecto integrador de Ciencia de Datos · UTN FRM 2026.

**La pregunta:** con lo que se sabe *antes del estreno* (presupuesto, género, idioma, duración, productora, director,
elenco, saga y fecha), ¿podemos anticipar cuántas veces recupera una película lo que costó?

**El hallazgo:** sí, pero poco. En películas de 2017–2025 que nunca vio, el modelo (el promedio de un Random Forest
ajustado, HistGB y Ridge) anticipa el múltiplo `recaudación / presupuesto` con un error típico de **×2,67**, un
**13,5%** menos que adivinar siempre la mediana (×3,11). En la Entrega 3 era ×2,86 (7% menos): la mejora vino de
sumar director, actor, saga, palabras clave y países, que se conocen antes del estreno. Lo que más pesa es el presupuesto: las películas baratas multiplican mucho más lo que
cuestan (mediana ×18 con menos de 1 millón de dólares, ×2 entre 10 y 100 millones).

**App:** _pendiente de publicar en Streamlit Community Cloud_ (ver [Cómo se publica](#cómo-se-publica)).

---

## Estructura del repo

```
dags/tmdb_ingest.py          DAG de Airflow: TMDb -> bronce -> plata -> include/output/movies_<fecha>.csv
include/tmdb/                cliente de la API, esquema y transformación JSON -> fila
  enriquecer.py              agrega las columnas de CONTEXTO a un CSV ya generado, sin cambiar las demás
include/frozen/ultimo_ok.csv dataset versionado (corrida full del 24/10/2026: 9.940 películas x 27 columnas;
                             las 9 de CONTEXTO se agregaron el 10/11 con include/tmdb/enriquecer.py)
include/output/              corridas propias del DAG (en .gitignore: no se sube)
include/modelo/              lógica compartida por el notebook, build_app.py y la app
  features.py                carga, filtros de alcance, objetivo y features derivadas
  entrenar.py                columnas, pipeline, partición temporal, candidatos, grilla y comparaciones
  analisis.py                números de las fichas de hipótesis de la Entrega 2
presentacion2.ipynb          Entrega 2: análisis (fichas de hipótesis y chequeo de fuga)
presentacion3.ipynb          Entrega 3: modelo (corregido y guardado con salidas)
build_app.py                 de los datos a la app: entrena el modelo final y genera app/artefactos/
app/streamlit_app.py         la app (Entrega 4)
app/estilo.py                colores, escalas y formatos de todos los gráficos
app/artefactos/              modelo.joblib, referencias.joblib, metricas.json, predicciones.parquet, rangos.json
app/requirements.txt         dependencias fijadas de la app (las usa Streamlit Cloud)
.streamlit/config.toml       tema de la app
requirements.txt             dependencias del contenedor de Airflow (no tocar para la app)
requirements-notebooks.txt   dependencias para correr los notebooks y build_app.py
```

## 1. Correr el pipeline (Entrega 1)

Requisitos: Docker y la [CLI de Astronomer](https://www.astronomer.io/docs/astro/cli/install-cli).

1. Crear un archivo `.env` en la raíz (está en `.gitignore`; **no se sube**):
   ```
   TMDB_API_KEY=<tu clave de https://www.themoviedb.org/settings/api>
   BASE_URL=https://api.themoviedb.org/3
   ```
2. Levantar Airflow:
   ```bash
   astro dev start
   ```
3. En http://localhost:8080, disparar el DAG **`tmdb_ingest`** con el parámetro `mode = full`
   (`subset` baja ~120 películas para probar).
4. El DAG hace muestreo estratificado (4.000 más populares, 3.000 peor calificadas y 3.000 de mayor taquilla, todas
   con al menos 100 votos), guarda el JSON crudo en la capa bronce, arma la capa plata (27 columnas, sin duplicados,
   validada), completa la posición de cada película en su saga (`enrich_sagas`) y publica
   `include/output/movies_<fecha>.csv`. También actualiza `include/frozen/ultimo_ok.csv`, que
   usa como respaldo si la API no responde.

**Columnas de CONTEXTO (Entrega 4):** director, actor principal y saga (id y nombre), posición dentro de la saga,
países de producción y palabras clave. Se conocen antes del estreno y dan la historia del director, del actor y de
la saga. Para agregarlas a la corrida del 24/10 sin volver a muestrear (una corrida nueva trae otras películas, porque
la popularidad de TMDb cambia todos los días):

```bash
cd include
TMDB_API_KEY=<tu clave> BASE_URL=https://api.themoviedb.org/3 python -m tmdb.enriquecer frozen/ultimo_ok.csv
```

## 2. Correr los notebooks (Entregas 2 y 3)

Con Python 3.12:

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows (en Linux/Mac: source .venv/bin/activate)
pip install -r requirements-notebooks.txt
```

Los notebooks leen el CSV más reciente de `include/output/movies_*.csv`. Si no corriste el DAG, copiá el congelado:

```bash
mkdir -p include/output
cp include/frozen/ultimo_ok.csv include/output/movies_2026-10-24.csv
```

Después, desde la raíz del repo:

```bash
jupyter nbconvert --to notebook --execute --inplace presentacion3.ipynb
```

(o abrirlos con `jupyter lab` y correr todo). `presentacion3.ipynb` importa la lógica de `include/modelo/`, así que
se ejecuta desde la raíz.

## 3. Regenerar la app

```bash
python build_app.py
streamlit run app/streamlit_app.py
```

`build_app.py` toma el CSV más reciente (o `include/frozen/ultimo_ok.csv` si no hay ninguno), aplica los mismos
filtros, features, partición temporal, pipeline y semilla (42) que el notebook, entrena el modelo final (promedio de
Random Forest ajustado, HistGB y Ridge) con el 80% de entrenamiento y guarda los artefactos en `app/artefactos/`. Al final controla que el MAE de test coincida con
el que imprimió el notebook:

```
MAE test notebook = 0.4267 | MAE test build_app = 0.4267
```

Si no coinciden, el script falla. **Si cambian los datos:** correr el DAG, ejecutar `presentacion3.ipynb` (paso 2) y
después `python build_app.py`. Commitear `presentacion3.ipynb` y `app/artefactos/`: Streamlit Cloud no corre ni el DAG
ni el entrenamiento, solo lee los artefactos.

## Cómo se publica

La app está en Streamlit Community Cloud, conectada a este repositorio:

| Campo | Valor |
|---|---|
| Repository | `AlejoCarobolante/CienciaDatos-Integrador-Peliculon` |
| Branch | `mejoras-tmdb` |
| Main file path | `app/streamlit_app.py` |
| Python version (Advanced settings) | 3.12 |
| Dependencias | `app/requirements.txt` (Cloud lo busca primero en la carpeta del archivo principal) |

Cada push a la rama vuelve a desplegar la app. No hace falta ningún secreto: la app no llama a la API de TMDb.

---

Datos: [TMDb](https://www.themoviedb.org/). Este producto usa la API de TMDb pero no está avalado ni certificado por
TMDb.
