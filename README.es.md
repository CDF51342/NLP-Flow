<div align="center">

<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark-bg.png">
    <img src="static/icon-nlpflow.png" alt="NLP Flow" width="60" valign="middle" hspace="8">
  </picture>
  &nbsp;NLP Flow
</h1>

🇪🇸 Español | [🇬🇧 English](README.en.md) | [📖 Documentación interactiva](https://cdf51342.github.io/NLP-Flow/doc.html)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows%20%7C%20Linux-64748b?style=for-the-badge)]()

Entorno visual de machine learning basado en pipelines — sin escribir código.

![Demo de NLP Flow](docs/assets/demo.gif)

</div>

---

## ¿Qué es?

NLP Flow es una herramienta educativa y de investigación para explorar modelos de machine learning de forma visual. Diseñada para estudiantes, docentes e investigadores que quieren experimentar con pipelines de datos, NLP, visión artificial y RAG sin necesidad de programar.

**Casos de uso:**
- Enseñanza de ML en aula sin entorno de programación
- Prototipado rápido de pipelines de NLP o visión por computador
- Exploración exploratoria de datasets tabulares y de texto
- Demostración de conceptos (Topic Model, Autoencoder, RAG)

---

## Estructura del proyecto

```
NLP-flow/
├── main.py              # Lanzador — Flask + ventana nativa (pywebview)
├── server.py            # Backend Flask — rutas de ML, entrenamiento, exportación
├── requirements.txt     # Dependencias Python
├── .env                 # Variables de entorno (no incluido en el repo)
├── static/
│   ├── index.html       # Frontend SPA — canvas, bloques, modales
│   ├── icon-nlpflow.png # Icono de la app
│   └── style.css        # Estilos
└── docs/
    ├── index.html       # Página de inicio de la documentación
    ├── doc.html         # Documentación interactiva de bloques
    ├── pipelines.html   # Guía de pipelines
    └── assets/          # Imágenes, GIFs y vídeos para documentación
```

---

## Instalación

**Requisitos:** Python 3.10 o superior.

```bash
git clone https://github.com/CDF51342/NLP-flow.git
cd NLP-flow
pip install -r requirements.txt
```

Crea un fichero `.env` en la raíz si quieres usar el bloque LLM (RAG):

```
GROQ_API_KEY=tu_clave_groq        # https://console.groq.com
HF_TOKEN=tu_token_hf              # alternativa: Hugging Face
```

Los embeddings funcionan sin clave — usan `all-MiniLM-L6-v2` de Sentence Transformers.

---

## Uso

**Modo escritorio** (ventana nativa):

```bash
python main.py
```

**Solo navegador** (sin dependencia de pywebview):

```bash
python server.py
# Abre http://localhost:5053
```

---

## Configuración de entorno

El bloque **🤖 LLM (RAG)** admite dos proveedores. Crea un fichero `.env` en la raíz:

```
# Proveedor Groq (por defecto)
GROQ_API_KEY=tu_clave_groq
GROQ_MODEL=llama-3.1-8b-instant   # opcional

# Proveedor Hugging Face (alternativo)
HF_TOKEN=tu_token_hf
HF_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
```

---

## Guardar y cargar el canvas

- **💾 Guardar sesión (.zip)** — exporta canvas + configuración + estado de los bloques
- **📂 Cargar sesión (.zip)** — restaura un canvas guardado
- **🗺 Exportar canvas (.json)** — solo estructura de bloques y conexiones

> Los datos cargados en el servidor no se persisten entre sesiones. Al restaurar un canvas, vuelve a cargar el CSV en el bloque Datos.

---

## Licencia

Distribuido bajo la licencia [MIT](LICENSE) © 2026 Carlos Díez-Fenoy.

La licencia MIT permite usar, copiar, modificar y distribuir este software libremente. La única condición es mantener el aviso de copyright original en cualquier copia o redistribución.

Para citar este software en trabajos académicos, usa el botón **"Cite this repository"** de GitHub (archivo `CITATION.cff`) o el siguiente formato BibTeX:

```bibtex
@software{diezfenoy2026nlpflow,
  author  = {Díez-Fenoy, Carlos},
  title   = {NLP Flow},
  year    = {2026},
  url     = {https://github.com/CDF51342/NLP-flow},
  license = {MIT}
}
```

---

## Créditos

<p>
<strong>Carlos Díez-Fenoy</strong> &nbsp;
<a href="mailto:carlosdi@pa.uc3m.es"><img src="https://img.shields.io/badge/carlosdi%40pa.uc3m.es-EA4335?style=for-the-badge&logo=gmail&logoColor=white" alt="Email"></a>
<a href="https://github.com/CDF51342"><img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a>
<a href="https://www.linkedin.com/in/carlos-diez-fenoy"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"></a>
<a href="https://orcid.org/0009-0008-5225-3575"><img src="https://img.shields.io/badge/ORCID-A6CE39?style=for-the-badge&logo=orcid&logoColor=white" alt="ORCID"></a>
</p>
