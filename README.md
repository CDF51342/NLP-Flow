# NLP Flow

> 🇪🇸 Español | [🇬🇧 English](README.en.md) | [📖 Documentación interactiva](https://cdf51342.github.io/NLP-flow/doc.html)

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
├── server.py            # Backend Flask — 80+ rutas de ML
├── requirements.txt     # Dependencias Python
├── .env                 # Variables de entorno (GROQ_API_KEY, HF_TOKEN, etc.)
├── static/
│   ├── index.html       # Frontend SPA — canvas, bloques, modales
│   ├── favicon.png      # Icono de la app
│   └── style.css        # Estilos
└── docs/
    ├── index.html       # Página de inicio de la documentación
    ├── doc.html         # Documentación interactiva de bloques
    ├── pipelines.html   # Guía de pipelines típicas
    └── favicon.svg      # Icono para la web de documentación
```

---

## Bloques disponibles

Cada bloque tiene puertos de **entrada** (●—) y/o **salida** (—●). Se conectan arrastrando de puerto a puerto.

### Bloques de datos

| Bloque | Puertos | Descripción |
|--------|---------|-------------|
| 📂 **Datos** | solo salida | Sube un CSV propio (texto o tabular). Configura columnas, target y tipo de tarea. Detección automática NLP vs tabular. |
| 🔬 **Análisis** | solo entrada | Info del dataset: filas, columnas, tipos, valores nulos, estadísticas y distribuciones. |
| 📊 **Plots** | entrada + salida | **Tabular:** histograma, boxplot, scatter, correlación, barras por clase, tarta, líneas. **NLP / Topic model:** longitud de documentos, top N palabras frecuentes, distribución de clases, longitud media por clase, wordcloud por clase. Historial de gráficas con comparación lado a lado. |

### Bloques compartidos (tabular, NLP e imagen)

| Bloque | Puertos | Descripción |
|--------|---------|-------------|
| 🔧 **Preprocesado** | entrada + salida | Se adapta al tipo de datos upstream. **Tabular:** imputación (mediana/moda), escalado (StandardScaler) y codificación de variables categóricas. **NLP:** lowercase, eliminar puntuación y números, stopwords EN/ES, stemming o lemmatización, preview en vivo. **Imagen:** resize (28–128 px), normalización ([0,1] o media/std), batch size, test split (10/15/20/25%), data augmentation solo en train (flip horizontal, rotación ±15°, recorte aleatorio) con factor de multiplicación x1–x5, vista previa en tiempo real. |
| 📋 **Evaluación** | entrada + salida | Se adapta al modelo upstream. **Tabular/NLP:** accuracy, F1, RMSE, R², matriz de confusión, curva ROC, frontera de decisión 2D, validación cruzada. **Imagen (CNN):** cuatro pestañas — Comparar (tabla de runs con ★ mejor), Informe (métricas por clase + curvas + CM por run), Predecir (imagen propia → probabilidades de todas las clases para todos los runs, historial de 12 imágenes), Errores (imágenes mal clasificadas). |
| 💾 **Guardar** | solo entrada | Se adapta a lo que viene de upstream. **Tabular/NLP:** exporta modelo (.pkl), métricas (.json/.csv) y gráficas (.png/.zip). **RAG:** informe HTML con consulta, fragmentos recuperados, prompt y respuestas. **Imagen (CNN):** genera informe HTML completo en ES o EN — comparativa de runs, métricas por clase, curvas, matriz de confusión y predicciones de imágenes del usuario. En todos los casos abre el diálogo nativo del SO. |
| 📂 **Cargar Modelo** | solo salida | Carga un modelo .pkl guardado para evaluarlo o clasificar nuevas muestras. |

### Bloques tabulares

| Bloque | Puertos | Descripción |
|--------|---------|-------------|
| 📉 **Regresión Lineal** | entrada + salida | OLS, Ridge (L2) y LASSO (L1). Detecta automáticamente clasificación o regresión según el target. Grid search CV de lambda/C con visualización de curvas. |

### Bloques NLP

| Bloque | Puertos | Descripción |
|--------|---------|-------------|
| 🧠 **Entrenamiento NLP** | entrada + salida | Pipeline TF-IDF + clasificador (Naive Bayes, Regresión Logística, KNN, Random Forest, SVM). Validación cruzada y grid search de hiperparámetros. |
| 🗂️ **Topic Model** | entrada + salida | LDA, NMF y LSA para descubrir temas latentes en corpus de texto sin etiquetas. Configurable: nº de tópicos, vocabulario, iteraciones, min_df, max_df, alpha y beta (LDA). Coherencia C_V y C_NPMI por tópico, topic diversity. Etiquetado automático de tópicos con LLM (Groq). |
| 🎯 **Clasificar NLP** | solo entrada | Clasifica nuevas muestras de texto con el modelo NLP entrenado. Muestra clase predicha y distribución de probabilidades. |

### Pipeline RAG

| Bloque | Puertos | Descripción |
|--------|---------|-------------|
| ✂️ **Chunking** | entrada + salida | Divide documentos en fragmentos solapados. Configurable: tamaño del chunk y solapamiento. |
| 🧬 **Embeddings** | entrada + salida | Vectoriza chunks con `all-MiniLM-L6-v2`. Progreso en tiempo real. |
| 🔎 **Retriever** | entrada + salida | Búsqueda semántica por similitud coseno. Muestra los chunks más relevantes con score. |
| 🤖 **LLM (RAG)** | solo entrada | Genera respuesta sin RAG y con RAG en paralelo usando Groq. El prompt se adapta al idioma configurado (ES/EN). |

### Bloques de visión (imagen)

| Bloque | Puertos | Descripción |
|--------|---------|-------------|
| 📷 **Datos Imagen** | solo salida | Descarga automática de MNIST (70 000 imágenes, dígitos 0–9), Chihuahua vs Muffin (299 imgs, color) o Cats vs Dogs (~23 000 imgs, ~720 MB, muestra de 2 000). Muestra distribución de clases y ejemplos por clase. |
| 🧠 **CNN Clasificación** | entrada + salida | CNN entrenada desde cero con PyTorch en CPU. Configurable: épocas (1–20), learning rate, capas conv (1–3), val split. Progreso en tiempo real con curvas de loss/accuracy. Sistema de runs: múltiples configuraciones comparables y renombrables. El botón Train se bloquea durante el entrenamiento. |
| 🔄 **Autoencoder** | entrada + salida | Autoencoder convolucional entrenado desde cero con PyTorch. Aprende a comprimir y reconstruir imágenes sin etiquetas. Configurable: bottleneck (nº de neuronas en el cuello de botella), épocas (1–30), learning rate, arquitectura (auto/wide/deep), función de pérdida (MSE o BCE). Exploración interactiva del espacio latente con slider de ruido. Comparativa de runs por MSE de test. Informe HTML exportable. |

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
📂 Datos → 🔧 Preprocesado → 📊 Plots
```

**NLP — clasificación de texto:**
```
📂 Datos → 🔧 Preprocesado → 🧠 Entrenamiento NLP → 🎯 Clasificar NLP
```

**Topic Model:**
```
📂 Datos → 🔧 Preprocesado → 🗂️ Topic Model
```

**RAG (Retrieval-Augmented Generation):**
```
📂 Datos → ✂️ Chunking → 🧬 Embeddings → 🔎 Retriever → 🤖 LLM (RAG)
📂 Datos → 🔧 Preprocesado → ✂️ Chunking → 🧬 Embeddings → 🔎 Retriever → 🤖 LLM (RAG)
```

**Clasificación de imagen (CNN):**
```
📷 Datos Imagen → 🔧 Preprocesado → 🧠 CNN Clasificación → 📋 Evaluación → 💾 Guardar
```

**Autoencoder (compresión y reconstrucción):**
```
📷 Datos Imagen → 🔧 Preprocesado → 🔄 Autoencoder → 💾 Guardar
```

---

## Carga de datos

El bloque **📂 Datos** solo acepta **CSV subido por el usuario**. No hay datasets predeterminados.

1. Arrastra el CSV a la zona de carga o usa el selector de archivo.
2. El sistema detecta automáticamente si el dataset es NLP (pocas columnas, texto largo) o tabular (columnas numéricas múltiples).
3. Configura la columna de texto, la columna objetivo y el tipo de tarea.

La configuración de columnas y target se propaga a todos los bloques downstream conectados.

---

## Preprocesado NLP — pasos disponibles

| Paso | Descripción |
|------|-------------|
| Lowercase | Convierte todo el texto a minúsculas |
| Remove punctuation | Elimina signos de puntuación (preserva tildes) |
| Remove numbers | Elimina caracteres numéricos |
| Stop words (EN) | Elimina palabras vacías en inglés |
| Stop words (ES) | Elimina palabras vacías en español |
| Stemming | Reduce palabras a su raíz (EN/ES). Exclusivo con Lemmatización |
| Lemmatización | Reduce palabras a su lema morfológico. Exclusivo con Stemming |
| Normalize whitespace | Colapsa espacios múltiples |

El selector de documentos en el panel de preview se actualiza automáticamente al cambiar de documento. Si el dataset tiene columna de categorías, aparece un **filtro de clase** para explorar solo los documentos de esa clase, incluso en modo topic model.

---

## Plots NLP

Al conectar el bloque Plots a un upstream con datos de texto (directo desde Datos, o desde Preprocesado), el panel detecta automáticamente el tipo y muestra gráficas de análisis exploratorio de texto:

| Gráfica | Descripción | Requiere clases |
|---------|-------------|-----------------|
| 📏 Longitud de documentos | Histograma del nº de palabras por doc con mediana | No |
| 🔤 Palabras frecuentes | Barras horizontales con top N palabras (N configurable) | No |
| 📊 Distribución de clases | Barras con conteo y % por categoría | Sí |
| 📐 Longitud por clase | Media ± std de palabras por documento y clase | Sí |
| ☁️ Wordcloud por clase | Grid de nubes de palabras, una por categoría | Sí |

Se puede elegir entre corpus original o preprocesado (si hay un Preprocesado aplicado upstream).

---

## Idioma

La app soporta **español e inglés**. El idioma se cambia desde el selector de la barra superior. Todos los bloques, labels, ejes de gráficas y mensajes de error respetan el idioma activo, incluidos los prompts del bloque LLM RAG.

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

- **💾 Guardar sesión (.zip)** — exporta canvas + configuración completa + estado de los bloques
- **📂 Cargar sesión (.zip)** — restaura un canvas guardado (los datos del servidor deben recargarse manualmente)
- **🗺 Exportar canvas (.json)** — solo estructura de bloques y conexiones

> Los datos cargados en el servidor no se persisten entre sesiones. Al restaurar un canvas, vuelve a cargar el CSV en el bloque Datos.

---

## Configuración de entorno (RAG / LLM)

El bloque LLM (RAG) admite dos proveedores. Crea un fichero `.env` en la raíz del proyecto:

```
# Proveedor Groq (default)
GROQ_API_KEY=tu_clave_groq
GROQ_MODEL=llama-3.1-8b-instant   # opcional, este es el default

# Proveedor Hugging Face (alternativo)
HF_TOKEN=tu_token_hf
HF_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct   # optional, este es el default HF
```

Los embeddings (bloque 🧬 Embeddings) funcionan sin clave — usan `all-MiniLM-L6-v2` de Sentence Transformers.

---

## Notas técnicas

- El servidor escucha en `localhost:5053`. No expone puertos externos.
- Los modelos entrenados viven en memoria. Reiniciar el servidor los borra.
- Las gráficas se generan con matplotlib y se devuelven como PNG en base64.
- Cada nodo Datos tiene su propio slot `_NODE_DATA[node_id]`: puedes tener varias pipelines en paralelo con datasets distintos.
- El frontend es una SPA sin frameworks. Todo el estado del canvas vive en memoria JS.
- El estado de los bloques se persiste en `node.data` y se serializa al guardar sesión, incluyendo imágenes del historial de Plots.
