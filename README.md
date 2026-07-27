# NLP Flow v4

> 🇪🇸 Español | [🇬🇧 English](README.en.md) | [📖 Documentación interactiva](https://TUUSUARIO.github.io/NOMBRE-REPO/)

Entorno visual de machine learning basado en pipelines. Conectas bloques en un canvas, configuras cada uno con doble clic y ejecutas el flujo completo con un botón. **No hace falta escribir código.**

---

## Requisitos y arranque

Python 3.10 o superior. Instala dependencias:

```bash
pip install -r requirements.txt
```

**Modo escritorio** (ventana nativa 1440×900):

```bash
python main.py
```

**Solo navegador** (sin pywebview):

```bash
python server.py
# Abre http://localhost:5053
```

---

## Estructura del proyecto

```
nlp-flow4/
├── main.py              # Lanzador: Flask + ventana nativa
├── server.py            # Backend Flask — 55+ rutas de ML
├── requirements.txt     # Dependencias Python
└── static/
    ├── index.html       # Frontend SPA — canvas, bloques, modales
    └── style.css        # Estilos
```

---

## Bloques disponibles

Cada bloque tiene puertos de **entrada** (●—) y/o **salida** (—●). Se conectan arrastrando de puerto a puerto.

| Bloque | Puertos | Descripción |
|--------|---------|-------------|
| 📂 **Datos** | solo salida | Carga datasets built-in o CSV propio. Configura columnas y target. |
| 🔬 **Análisis** | solo entrada | Info del dataset: filas, columnas, tipos, valores nulos y estadísticas. |
| 📊 **Plots** | entrada + salida | Histograma, boxplot, scatter, correlación, barras, tarta, líneas. Historial de gráficas. |
| 🔧 **Preprocesado** | entrada + salida | Imputación, normalización, codificación, split train/test, selección de features. |
| 📉 **Regresión Lineal** | entrada + salida | OLS, Ridge (L2) y LASSO (L1). Detecta automáticamente clasificación o regresión según el target. Grid search CV de lambda/C. |
| 📋 **Evaluación** | entrada + salida | Métricas, matriz de confusión, curva ROC, frontera de decisión, supuestos de regresión, tab CV. |
| 💾 **Guardar** | solo entrada | Exporta modelo (.pkl), métricas (.json/.csv) y gráficas (.png). |
| 📂 **Cargar Modelo** | solo salida | Carga un modelo .pkl guardado para evaluarlo o clasificar nuevas muestras. |
| 🧠 **Entrenamiento NLP** | entrada + salida | Pipeline de texto: TF-IDF + clasificador. |
| 🗂️ **Topic Model** | solo entrada | LDA para descubrir temas latentes en corpus de texto. |
| 🎯 **Clasificar NLP** | solo entrada | Clasifica nuevas muestras de texto con un modelo NLP entrenado. |

---

## Pipelines típicas

**Clasificación / regresión tabular:**
```
📂 Datos → 🔧 Preprocesado → 📉 Regresión Lineal → 📋 Evaluación → 💾 Guardar
```

**Exploración de datos:**
```
📂 Datos → 🔬 Análisis
📂 Datos → 📊 Plots
```

**NLP — texto:**
```
📂 Datos → 🔧 Preprocesado → 🧠 Entrenamiento → 🎯 Clasificar
```

---

## Datasets incluidos

### Texto

| Dataset | Tarea | Descripción |
|---------|-------|-------------|
| 🎬 Movie Reviews | Clasificación | Críticas de cine positivas y negativas |
| 📧 Spam vs Ham | Clasificación | Mensajes spam y legítimos |
| 😊 Twitter Sentiment | Clasificación | Tweets positivos, negativos y neutros |
| 📰 News | Topic model | Artículos de noticias por temática |

### Tabulares

| Dataset | Tarea | Descripción |
|---------|-------|-------------|
| 🏥 Patient Health Risk | Clasificación | Variables de salud → riesgo alto/bajo |
| 🏠 California Housing | Regresión | Características de viviendas → precio medio |

También puedes subir tu propio CSV desde el bloque **Datos**.

---

## Atajos de teclado

| Atajo | Acción |
|-------|--------|
| Doble clic sobre bloque | Abre el popup del bloque |
| `⌘/Ctrl` + clic | Añade o quita bloque de la selección |
| `⌘/Ctrl` + `A` | Seleccionar todos (modo selección) |
| `⌘/Ctrl` + `C` | Copiar selección (con conexiones internas) |
| `⌘/Ctrl` + `V` | Pegar (restaura conexiones internas) |
| `⌘/Ctrl` + `Z` | Deshacer |
| `⌘/Ctrl` + `Shift` + `Z` | Rehacer |
| `Supr` / `⌫` | Eliminar bloques seleccionados |
| Rueda del ratón | Zoom in/out |

---

## Guardar y cargar el canvas

- **💾 Guardar sesión (.zip)** — exporta canvas + configuración completa
- **📂 Cargar sesión (.zip)** — restaura un canvas guardado
- **🗺 Exportar canvas (.json)** — solo estructura de bloques y conexiones

> Los datos cargados en el servidor no se persisten entre sesiones. Al restaurar un canvas, vuelve a cargar el dataset en el bloque Datos.

---

## Notas técnicas

- El servidor escucha en `localhost:5053`. No expone puertos externos.
- Los modelos entrenados viven en memoria. Reiniciar el servidor los borra.
- Las gráficas se generan con matplotlib y se devuelven como PNG en base64.
- Cada nodo Datos tiene su propio slot `_NODE_DATA[node_id]`: puedes tener varias pipelines en paralelo con datasets distintos.
- El frontend es una SPA sin frameworks. Todo el estado del canvas vive en memoria JS.
