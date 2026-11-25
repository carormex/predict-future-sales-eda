"""
Proyecto 2 - Pipeline de Machine Learning (Predict Future Sales)

Funciones de preparación de datos y transformadores de sklearn usados en los
notebooks 02 a 05. Los transformadores viven en este módulo (y no dentro del
notebook) para que los archivos .pkl se puedan cargar desde cualquier notebook.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

# Se desactiva el motor numexpr de pandas: en tablas de más de 1 millón de filas
# pandas lo usa automáticamente y dio resultados inconsistentes. Las operaciones
# numéricas de los transformadores se hacen directamente con numpy.
pd.set_option("compute.use_numexpr", False)

# Tiendas duplicadas detectadas en el EDA -> ID que usa test.csv
MAPA_TIENDAS = {0: 57, 1: 58, 11: 10}
COLS_ID = ["shop_id", "item_id", "date_block_num"]
N_MESES = 35  # 0..33 historia + 34 (noviembre 2015, test)


# ---------------------------------------------------------------------------
# 1. Preparación de datos (repite la limpieza del notebook 01)
# ---------------------------------------------------------------------------
def limpiar_ventas(sales):
    """Aplica la misma limpieza del EDA (sección 4d)."""
    df = sales.drop_duplicates().copy()
    df["shop_id"] = df["shop_id"].replace(MAPA_TIENDAS)

    # Precio <= 0 -> mediana del mismo producto, tienda y mes (o del producto)
    mask = df["item_price"] <= 0
    if mask.any():
        ref = df[~mask].groupby(["item_id", "shop_id", "date_block_num"])["item_price"].median()
        ref_item = df[~mask].groupby("item_id")["item_price"].median()
        nuevos = []
        for it, sh, m in df.loc[mask, ["item_id", "shop_id", "date_block_num"]].values:
            nuevos.append(ref.get((it, sh, m), ref_item.get(it, np.nan)))
        df.loc[mask, "item_price"] = nuevos

    # Outliers extremos (ventas corporativas de la tienda en línea)
    df = df[(df["item_price"] <= 100_000) & (df["item_cnt_day"] < 1_000)]
    return df.reset_index(drop=True)


def ventas_mensuales(df):
    """Agrega las ventas diarias a nivel mes x tienda x producto."""
    agrupado = df.groupby(COLS_ID)
    mensual = agrupado.agg(item_cnt_month=("item_cnt_day", "sum"), precio_medio=("item_price", "mean"))
    mensual = mensual.reset_index()
    mensual["shop_id"] = mensual["shop_id"].astype("int16")
    mensual["item_id"] = mensual["item_id"].astype("int32")
    mensual["date_block_num"] = mensual["date_block_num"].astype("int8")
    mensual["item_cnt_month"] = mensual["item_cnt_month"].astype("float32")
    mensual["precio_medio"] = mensual["precio_medio"].astype("float32")
    return mensual


def construir_grid(df, meses=None):
    """
    Grid mensual: para cada mes, todas las combinaciones (tiendas activas x
    productos vendidos ese mes). Los pares sin venta quedan con 0.
    La venta mensual se acota a [0, 20] (así evalúa Kaggle).
    """
    meses = sorted(df["date_block_num"].unique()) if meses is None else meses
    bloques = []
    for m in meses:
        dm = df[df["date_block_num"] == m]
        tiendas = dm["shop_id"].unique()
        productos = dm["item_id"].unique()
        idx = pd.MultiIndex.from_product([tiendas, productos], names=["shop_id", "item_id"])
        g = idx.to_frame(index=False)
        g["date_block_num"] = m
        bloques.append(g)
    grid = pd.concat(bloques, ignore_index=True)
    grid = grid.astype({"shop_id": "int16", "item_id": "int32", "date_block_num": "int8"})

    mensual = ventas_mensuales(df)
    grid = grid.merge(mensual[COLS_ID + ["item_cnt_month"]], on=COLS_ID, how="left")
    grid["item_cnt_month"] = grid["item_cnt_month"].fillna(0).clip(0, 20).astype("float32")
    return grid


# ---------------------------------------------------------------------------
# 2. Transformadores de ingeniería de características
# ---------------------------------------------------------------------------
def _llave(*cols):
    """Combina columnas enteras en una sola llave int64 (para búsquedas rápidas)."""
    k = np.asarray(cols[0], dtype="int64")
    for c in cols[1:]:
        k = k * 100_000 + np.asarray(c, dtype="int64")
    return k


class ConstructorRezagos(BaseEstimator, TransformerMixin):
    """
    Recibe filas (shop_id, item_id, date_block_num) y agrega:
      - catálogo: categoría, grupo de categoría, ciudad y tipo de tienda
      - rezagos de la venta del par tienda-producto (lags)
      - rezagos de la venta media del producto, de la tienda y de la categoría
      - último precio conocido del producto y banderas de producto/par nuevo
    Todos los rezagos de un mes m usan SOLO meses anteriores (m-1, m-2, ...),
    por eso no hay fuga de información.
    """

    def __init__(self, historia, items, tiendas, rezagos_par=(1, 2, 3, 12), rezagos_item=(1, 2, 3), rezagos_tienda=(1,), rezagos_categoria=(1,), mapa_tiendas=None):
        self.historia = historia
        self.items = items
        self.tiendas = tiendas
        self.rezagos_par = rezagos_par
        self.rezagos_item = rezagos_item
        self.rezagos_tienda = rezagos_tienda
        self.rezagos_categoria = rezagos_categoria
        self.mapa_tiendas = mapa_tiendas

    def fit(self, X, y=None):
        h = self.historia.copy()
        h["cnt"] = h["item_cnt_month"].clip(0, 20)
        h = h.merge(self.items[["item_id", "item_category_id"]], on="item_id", how="left")
        self.primer_mes_ = int(h["date_block_num"].min())

        # Tiendas activas por mes (para promedios por tienda)
        n_tiendas = h.groupby("date_block_num")["shop_id"].nunique()

        # Tablas de búsqueda: llave -> valor
        self.t_par_ = pd.Series(h["cnt"].values, index=_llave(h["shop_id"], h["item_id"], h["date_block_num"]))
        it = h.groupby(["item_id", "date_block_num"])["cnt"].sum().reset_index()
        it["v"] = it["cnt"] / it["date_block_num"].map(n_tiendas)
        self.t_item_ = pd.Series(it["v"].values, index=_llave(it["item_id"], it["date_block_num"]))
        sh = h.groupby(["shop_id", "date_block_num"])["cnt"].sum().reset_index()
        self.t_tienda_ = pd.Series(sh["cnt"].values, index=_llave(sh["shop_id"], sh["date_block_num"]))
        ca = h.groupby(["item_category_id", "date_block_num"])["cnt"].sum().reset_index()
        ca["v"] = ca["cnt"] / ca["date_block_num"].map(n_tiendas)
        self.t_cat_ = pd.Series(ca["v"].values, index=_llave(ca["item_category_id"], ca["date_block_num"]))

        # Primer mes de venta del producto y del par
        self.primera_item_ = h.groupby("item_id")["date_block_num"].min()
        pp = h.groupby(["shop_id", "item_id"])["date_block_num"].min().reset_index()
        self.primera_par_ = pd.Series(pp["date_block_num"].values, index=_llave(pp["shop_id"], pp["item_id"]))

        # Último precio conocido ANTES de cada mes (matriz producto x mes)
        n_items = int(max(self.items["item_id"].max(), h["item_id"].max())) + 1
        precio = np.full((n_items, N_MESES + 1), np.nan, dtype="float32")
        pm = h.groupby(["item_id", "date_block_num"])["precio_medio"].mean().reset_index()
        precio[pm["item_id"].values, pm["date_block_num"].values + 1] = pm["precio_medio"].values
        precio = pd.DataFrame(precio).ffill(axis=1).values  # arrastra el último precio
        self.precio_previo_ = np.array(precio, dtype="float32", copy=True)  # columna m = último precio visto en meses < m
        return self

    def _buscar(self, tabla, llave):
        return tabla.reindex(llave).to_numpy(dtype="float32")

    def transform(self, X):
        X = X[COLS_ID].copy()
        if self.mapa_tiendas:
            X["shop_id"] = X["shop_id"].replace(self.mapa_tiendas)
        X = X.astype({"shop_id": "int16", "item_id": "int32", "date_block_num": "int8"})
        m = X["date_block_num"].to_numpy().astype("int64")

        # Catálogo (variables exógenas de tienda y producto)
        X = X.merge(self.items[["item_id", "item_category_id", "categoria_grupo"]], on="item_id", how="left")
        X = X.merge(self.tiendas[["shop_id", "ciudad", "tienda_tipo"]], on="shop_id", how="left")
        X["mes"] = (m % 12 + 1).astype("int8")
        cat = X["item_category_id"].fillna(-1).astype("int64").to_numpy()

        def rezago(tabla, cols, k, nombre):
            llave = _llave(*cols, m - k)
            v = np.nan_to_num(self._buscar(tabla, llave), nan=0.0)   # mes sin venta -> 0
            v[(m - k) < self.primer_mes_] = np.nan                    # antes de la historia -> desconocido
            X[nombre] = v

        s, i = X["shop_id"].to_numpy(), X["item_id"].to_numpy()
        for k in self.rezagos_par:
            rezago(self.t_par_, (s, i), k, f"venta_par_lag_{k}")
        for k in self.rezagos_item:
            rezago(self.t_item_, (i,), k, f"venta_item_lag_{k}")
        for k in self.rezagos_tienda:
            rezago(self.t_tienda_, (s,), k, f"venta_tienda_lag_{k}")
        for k in self.rezagos_categoria:
            rezago(self.t_cat_, (cat,), k, f"venta_categoria_lag_{k}")

        # Antigüedad y banderas de producto / par nuevo
        pi = X["item_id"].map(self.primera_item_).to_numpy(dtype="float64")
        pp = self.primera_par_.reindex(_llave(s, i)).to_numpy(dtype="float64")
        item_visto = (pi < m)
        par_visto = (pp < m)
        X["item_nuevo"] = (~item_visto).astype("int8")
        X["par_nuevo"] = (~par_visto).astype("int8")
        X["item_meses_desde_primera_venta"] = np.where(item_visto, m - pi, 0).astype("float32")
        X["par_meses_desde_primera_venta"] = np.where(par_visto, m - pp, 0).astype("float32")

        # Último precio del producto en meses anteriores (NaN si nunca se vendió)
        ids = X["item_id"].to_numpy()
        ok = ids < self.precio_previo_.shape[0]
        precio = np.full(len(X), np.nan, dtype="float32")
        precio[ok] = self.precio_previo_[ids[ok], np.clip(m[ok], 0, N_MESES)]
        X["precio_ultimo_item"] = precio
        return X


class ImputadorCategorico(BaseEstimator, TransformerMixin):
    """Imputa variables categóricas con la moda (aprendida en fit)."""

    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y=None):
        self.moda_ = {v: X[v].mode().iloc[0] for v in self.variables}
        return self

    def transform(self, X):
        X = X.copy()
        for v in self.variables:
            X[v] = X[v].fillna(self.moda_[v])
        return X


class ImputadorMedianaGrupo(BaseEstimator, TransformerMixin):
    """Imputa variables numéricas con la mediana de su grupo (ej. categoría)."""

    def __init__(self, variables, grupo):
        self.variables = variables
        self.grupo = grupo

    def fit(self, X, y=None):
        self.mediana_grupo_ = {v: X.groupby(self.grupo)[v].median() for v in self.variables}
        self.mediana_global_ = {v: X[v].median() for v in self.variables}
        return self

    def transform(self, X):
        X = X.copy()
        for v in self.variables:
            valores = X[v].to_numpy(dtype="float64", copy=True)
            mediana_de_su_grupo = X[self.grupo].map(self.mediana_grupo_[v]).to_numpy(dtype="float64")
            faltan = np.isnan(valores)
            valores[faltan] = mediana_de_su_grupo[faltan]
            valores[np.isnan(valores)] = self.mediana_global_[v]
            X[v] = valores.astype("float32")
        return X


class ImputadorConstante(BaseEstimator, TransformerMixin):
    """Imputa variables numéricas con un valor fijo (ej. 0 = sin venta)."""

    def __init__(self, variables, valor=0):
        self.variables = variables
        self.valor = valor

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        vs = [v for v in X.columns if v in self.variables]
        X[vs] = X[vs].fillna(self.valor)
        return X


class TratamientoOutliers(BaseEstimator, TransformerMixin):
    """
    Acota (winsoriza) valores extremos con límites aprendidos en fit.
      metodo='iqr_log'   -> límites Q1 - 1.5*IQR y Q3 + 1.5*IQR en escala log
      metodo='percentil' -> límite superior en el percentil indicado
    """

    def __init__(self, variables, metodo="iqr_log", percentil=99.5):
        self.variables = variables
        self.metodo = metodo
        self.percentil = percentil

    def fit(self, X, y=None):
        self.limites_ = {}
        for v in self.variables:
            x = X[v].dropna()
            if self.metodo == "iqr_log":
                lx = np.log1p(x.clip(lower=0))
                q1, q3 = lx.quantile([0.25, 0.75])
                iqr = q3 - q1
                self.limites_[v] = (float(np.expm1(q1 - 1.5 * iqr)), float(np.expm1(q3 + 1.5 * iqr)))
            else:
                self.limites_[v] = (float(x.min()), float(np.percentile(x, self.percentil)))
        return self

    def transform(self, X):
        X = X.copy()
        for v, (lo, hi) in self.limites_.items():
            valores = X[v].to_numpy(dtype="float64", copy=True)
            X[v] = np.clip(valores, lo, hi).astype("float32")
        return X


class TransformacionLog(BaseEstimator, TransformerMixin):
    """Aplica log(1 + x) para reducir la asimetría."""

    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for v in self.variables:
            valores = X[v].to_numpy(dtype="float64", copy=True)
            X[v] = np.log1p(np.clip(valores, 0, None)).astype("float32")
        return X


class Mapper(BaseEstimator, TransformerMixin):
    """Reemplaza los valores de una variable según un diccionario (como en clase)."""

    def __init__(self, variables, mappings):
        self.variables = variables
        self.mappings = mappings

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for v in self.variables:
            X[v] = X[v].map(self.mappings).astype("float32")
        return X


class CodificadorMediaObjetivo(BaseEstimator, TransformerMixin):
    """
    Target (mean) encoding con suavizado: cada nivel se reemplaza por la venta
    promedio observada en el entrenamiento. Niveles nuevos -> promedio global.
    """

    def __init__(self, variables, suavizado=20):
        self.variables = variables
        self.suavizado = suavizado

    def fit(self, X, y):
        y = pd.Series(np.asarray(y), index=X.index)
        self.global_ = float(y.mean())
        self.mapas_ = {}
        for v in self.variables:
            g = y.groupby(X[v]).agg(["mean", "count"])
            w = g["count"] / (g["count"] + self.suavizado)
            self.mapas_[v] = w * g["mean"] + (1 - w) * self.global_
        return self

    def transform(self, X):
        X = X.copy()
        for v in self.variables:
            X[v] = X[v].map(self.mapas_[v]).fillna(self.global_).astype("float32")
        return X


class CodificadorOneHot(BaseEstimator, TransformerMixin):
    """One-hot encoding con las categorías vistas en fit (las nuevas quedan en 0)."""

    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y=None):
        self.categorias_ = {v: sorted(X[v].dropna().unique()) for v in self.variables}
        return self

    def transform(self, X):
        X = X.copy()
        for v in self.variables:
            for c in self.categorias_[v]:
                X[f"{v}_{c}"] = (X[v] == c).astype("int8")
        return X.drop(columns=self.variables)


class EliminarVariables(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=[v for v in self.variables if v in X.columns])


class Escalador(BaseEstimator, TransformerMixin):
    """Estandariza (media 0, desviación 1) y devuelve float32 para ahorrar memoria."""

    def fit(self, X, y=None):
        self.columnas_ = list(X.columns)
        valores = X[self.columnas_].to_numpy(dtype="float64")
        self.media_ = valores.mean(axis=0)
        desv = valores.std(axis=0, ddof=1)
        self.desv_ = np.where(desv == 0, 1.0, desv)
        return self

    def transform(self, X):
        valores = X[self.columnas_].to_numpy(dtype="float64", copy=True)
        escalado = ((valores - self.media_) / self.desv_).astype("float32")
        return pd.DataFrame(escalado, columns=self.columnas_)


class RegresorAcotado(BaseEstimator, RegressorMixin):
    """Envuelve un modelo y acota sus predicciones a [minimo, maximo]."""

    def __init__(self, estimador, minimo=0.0, maximo=20.0):
        self.estimador = estimador
        self.minimo = minimo
        self.maximo = maximo

    def fit(self, X, y):
        self.estimador_ = clone(self.estimador).fit(X, y)
        return self

    def predict(self, X):
        return np.clip(self.estimador_.predict(X), self.minimo, self.maximo)