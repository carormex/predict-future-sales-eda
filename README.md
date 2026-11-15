# Predict Future Sales — EDA orientado a forecasting

Proyecto 1 · Exploratory Data Analysis del dataset **Predict Future Sales** (Kaggle, 1C Company).
Autora: Ingrid Carolina Ortiz Mejicanos — Universidad Galileo.

**Notebook principal:** `notebooks/01_exploratory_data_analysis.ipynb`
**Figuras:** `reports/figures/`
**Traducciones ruso → español:** `predict_future_sales_eda/traducciones.py`

## Cómo ejecutarlo
1. Descargar los datos de [Kaggle](https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/data) y colocar los CSV en `data/raw/` (no se versionan, ~100 MB).
2. `pip install -r requirements.txt`
3. Abrir el notebook y ejecutar todas las celdas.

**Uso de IA:** desarrollado con apoyo de Claude Code en VSCode, como permite el enunciado.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         predict_future_sales_eda and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── predict_future_sales_eda   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes predict_future_sales_eda a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

