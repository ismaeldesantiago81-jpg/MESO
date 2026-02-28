from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


INPUT_PATH = Path("input") / "P02 IBERIA.xlsx"
SHEET_NAME = "Todos"
OUTPUT_DIR = Path("output")

MES_OBJETIVO = "ene 2026"

COL_MES = "Mes Año"
COL_FR1 = "FR1"
COL_SKU = "Artículo ID"
COL_VENTAS = "Ventas"
COL_DCH = "DCH"
COL_BU = "Unidad de negocio"
COL_SUBPLATAFORMA = "Subplataforma"

REQUIRED_COLUMNS = [
    COL_MES,
    COL_FR1,
    COL_SKU,
    COL_VENTAS,
    COL_DCH,
    COL_BU,
    COL_SUBPLATAFORMA,
]

MES_ABREVIATURAS = {
    1: "ene",
    2: "feb",
    3: "mar",
    4: "abr",
    5: "may",
    6: "jun",
    7: "jul",
    8: "ago",
    9: "sep",
    10: "oct",
    11: "nov",
    12: "dic",
}

MES_ALIAS = {
    "ene": 1,
    "enero": 1,
    "feb": 2,
    "febrero": 2,
    "mar": 3,
    "marzo": 3,
    "abr": 4,
    "abril": 4,
    "may": 5,
    "mayo": 5,
    "jun": 6,
    "junio": 6,
    "jul": 7,
    "julio": 7,
    "ago": 8,
    "agosto": 8,
    "sep": 9,
    "sept": 9,
    "septiembre": 9,
    "oct": 10,
    "octubre": 10,
    "nov": 11,
    "noviembre": 11,
    "dic": 12,
    "diciembre": 12,
}


def _validar_columnas(df: pd.DataFrame) -> None:
    faltantes = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")


def _cargar_datos(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    df = pd.read_excel(path, sheet_name=SHEET_NAME)
    _validar_columnas(df)
    return df


def _filtrar_y_limpiar(df: pd.DataFrame) -> pd.DataFrame:
    objetivo_normalizado = _normalizar_mes_anio_valor(MES_OBJETIVO)
    if objetivo_normalizado is None:
        raise ValueError(f"No se pudo interpretar MES_OBJETIVO: {MES_OBJETIVO}")

    meses_normalizados = df[COL_MES].map(_normalizar_mes_anio_valor)
    filtrado = df[meses_normalizados == objetivo_normalizado].copy()

    filtrado[COL_VENTAS] = pd.to_numeric(filtrado[COL_VENTAS], errors="coerce").fillna(0)
    filtrado[COL_DCH] = pd.to_numeric(filtrado[COL_DCH], errors="coerce").fillna(0)

    filtrado = filtrado[~((filtrado[COL_VENTAS] == 0) & (filtrado[COL_DCH] == 0))].copy()
    return filtrado


def _normalizar_mes_anio_valor(valor: object) -> str | None:
    if pd.isna(valor):
        return None

    ts = pd.to_datetime(valor, errors="coerce")
    if not pd.isna(ts):
        return f"{MES_ABREVIATURAS[int(ts.month)]} {int(ts.year)}"

    texto = str(valor).strip().lower()
    if not texto:
        return None

    tokens = texto.replace("-", " ").replace("/", " ").split()
    if len(tokens) >= 2:
        mes = MES_ALIAS.get(tokens[0])
        if mes is not None:
            anio = _extraer_anio(tokens[1])
            if anio is not None:
                return f"{MES_ABREVIATURAS[mes]} {anio}"

    return None


def _extraer_anio(texto: str) -> int | None:
    digitos = "".join(ch for ch in texto if ch.isdigit())
    if len(digitos) == 4:
        return int(digitos)
    if len(digitos) == 2:
        return 2000 + int(digitos)
    return None


def _consolidar_sku(df: pd.DataFrame) -> pd.DataFrame:
    sku = (
        df.groupby([COL_FR1, COL_SKU, COL_BU, COL_SUBPLATAFORMA], dropna=False, as_index=False)
        .agg(
            Ventas_SKU=(COL_VENTAS, "sum"),
            DCH_SKU=(COL_DCH, "sum"),
        )
        .copy()
    )

    sku["AbsError_SKU"] = (sku["DCH_SKU"] - sku["Ventas_SKU"]).abs()
    sku["Over_SKU"] = (sku["DCH_SKU"] - sku["Ventas_SKU"]).clip(lower=0)
    sku["Gap_SKU"] = (sku["Ventas_SKU"] - sku["DCH_SKU"]).clip(lower=0)
    sku["NoDemand_SKU"] = sku["DCH_SKU"].where(sku["Ventas_SKU"] == 0, 0)
    return sku


def _ranking_por_abs_error_demand(sku: pd.DataFrame) -> pd.DataFrame:
    ranking = sku[sku["Ventas_SKU"] > 0].copy()
    ranking["AbsError_demand"] = ranking["AbsError_SKU"]

    ranking = ranking.sort_values(
        by=[COL_BU, COL_SUBPLATAFORMA, "AbsError_demand", COL_SKU],
        ascending=[True, True, False, True],
    ).copy()

    ranking["Rank_AbsError_demand"] = (
        ranking.groupby([COL_BU, COL_SUBPLATAFORMA])["AbsError_demand"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return ranking


def _top10_por_bu_subplataforma(ranking: pd.DataFrame) -> pd.DataFrame:
    top10 = (
        ranking.sort_values(
            by=[COL_BU, COL_SUBPLATAFORMA, "AbsError_demand", COL_SKU],
            ascending=[True, True, False, True],
        )
        .groupby([COL_BU, COL_SUBPLATAFORMA], as_index=False, group_keys=False)
        .head(10)
        .copy()
    )
    return top10


def _validar_skus(ranking: pd.DataFrame, df_excel: pd.DataFrame) -> pd.DataFrame:
    skus_excel = set(df_excel[COL_SKU].dropna().astype(str))
    ranking = ranking.copy()
    ranking[COL_SKU] = ranking[COL_SKU].astype(str)
    ranking["Existe_en_excel"] = ranking[COL_SKU].isin(skus_excel)

    validacion = (
        ranking[[COL_BU, COL_SUBPLATAFORMA, COL_FR1, COL_SKU, "Existe_en_excel"]]
        .drop_duplicates()
        .sort_values(by=[COL_BU, COL_SUBPLATAFORMA, COL_FR1, COL_SKU])
    )
    return validacion


def _redondear(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols_numericas = out.select_dtypes(include=["number"]).columns
    out[cols_numericas] = out[cols_numericas].round(2)
    return out


def main() -> int:
    try:
        df_excel = _cargar_datos(INPUT_PATH)
        df_filtrado = _filtrar_y_limpiar(df_excel)
        sku = _consolidar_sku(df_filtrado)
        ranking = _ranking_por_abs_error_demand(sku)
        top10 = _top10_por_bu_subplataforma(ranking)
        validacion = _validar_skus(ranking, df_excel)

        if not validacion["Existe_en_excel"].all():
            print("REPORTE KO")
            return 1

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        _redondear(ranking).to_csv(OUTPUT_DIR / "ranking_bu_subplataforma.csv", index=False)
        _redondear(top10).to_csv(OUTPUT_DIR / "top10_skus_por_bu_subplataforma.csv", index=False)
        validacion.to_csv(OUTPUT_DIR / "validacion_skus.csv", index=False)

        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
