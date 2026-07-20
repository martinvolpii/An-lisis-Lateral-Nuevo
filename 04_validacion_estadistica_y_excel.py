#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_validacion_estadistica_y_excel.py

Cuarta parte del pipeline para análisis cinemático lateral con DeepLabCut.
Versión P30 para datos reales — corrección de auditoría Day 1.

Principios de esta versión:
- NO excluye animales por ID.
- NO excluye datasets completos automáticamente.
- NO deduplica ni elige un "mejor dataset" automáticamente.
- Respeta los criterios técnicos aplicados por los scripts 01, 02 y 03.
- Conserva las banderas de control de calidad temporal (accepted_temporal,
  reject_reason) para auditoría.
- Para estadística, las variables que dependen de toe-off (stance/swing/toe clearance)
  solo se consideran válidas cuando accepted_temporal == 1.
- stride_duration_s se conserva para TODOS los ciclos validados porque depende solo de
  start_frame/end_frame y no requiere toe-off.
- Los valores originales se conservan además en columnas *_raw.
- La unidad estadística principal es el animal, no el ciclo.

Entradas esperadas, buscadas recursivamente dentro de --input-dir:
    *_cycle_angle_ranges.csv
    *_gait_temporal_by_cycle.csv

Uso:
    python 04_validacion_estadistica_y_excel.py \
        --input-dir carpeta_resultados_pipeline \
        --out validacion_estadistica_dlc.xlsx
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import shapiro
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# =============================================================================
# VARIABLES DEL PIPELINE
# =============================================================================

ANGLE_VARIABLES = [
    "hip_range_deg",
    "knee_range_deg",
    "ankle_range_deg",
    "foot_range_deg",
]

TEMPORAL_VARIABLES = [
    "stride_duration_s",
    "stance_duration_s",
    "swing_duration_s",
    "stance_percent",
    "swing_percent",
    "toe_clearance_px",
]

DEFAULT_VARIABLES = ANGLE_VARIABLES + TEMPORAL_VARIABLES

ID_COLUMNS = [
    "animal_id",
    "dataset_id",
    "source_stem",
    "cycle_id",
    "start_frame",
    "end_frame",
]

TEMPORAL_QC_COLUMNS = [
    "accepted_temporal",
    "reject_reason",
    "toe_off_frame",
    "stance_duration_frames",
    "swing_duration_frames",
    "contact_level_px",
    "cycle_vertical_range_px",
    "toe_off_threshold_px",
    "finite_fraction_signal",
]


# =============================================================================
# UTILIDADES
# =============================================================================

def parse_animal_id(text: str) -> str:
    """Extrae ID de animal de formatos R1/R2 o 856_P30/857_P30..."""
    text = str(text)

    m = re.search(r"(^|[^A-Za-z0-9])(R\d+)([^A-Za-z0-9]|$)", text, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"R\d+", text, flags=re.IGNORECASE)
    if m:
        token = m.group(2) if len(m.groups()) >= 2 and m.group(2) else m.group(0)
        return token.upper()

    m = re.search(r"(?:^|[^0-9])(\d+)_P\d+", text, flags=re.IGNORECASE)
    if m:
        return m.group(1)

    m = re.match(r"^(\d{2,})(?=[_-])", text)
    if m:
        return m.group(1)

    return "UNKNOWN"


def clean_stem_from_suffix(path: Path, suffix: str) -> str:
    name = path.name
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return path.stem


def sem(values: Sequence[float]) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) <= 1:
        return np.nan
    return float(np.std(arr, ddof=1) / math.sqrt(len(arr)))


def safe_cv_percent(values: Sequence[float]) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) <= 1:
        return np.nan
    mean = float(np.mean(arr))
    if mean == 0 or not np.isfinite(mean):
        return np.nan
    return float(np.std(arr, ddof=1) / abs(mean) * 100.0)


def iqr(values: Sequence[float]) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return np.nan
    return float(np.nanpercentile(arr, 75) - np.nanpercentile(arr, 25))


def descriptive_stats(values: Sequence[float]) -> Dict[str, float]:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    n = int(len(arr))
    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "sd": np.nan,
            "sem": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
            "range": np.nan,
            "iqr": np.nan,
            "cv_percent": np.nan,
        }
    sd = float(np.std(arr, ddof=1)) if n > 1 else np.nan
    return {
        "n": n,
        "mean": float(np.mean(arr)),
        "sd": sd,
        "sem": float(sd / math.sqrt(n)) if n > 1 else np.nan,
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "range": float(np.max(arr) - np.min(arr)),
        "iqr": iqr(arr),
        "cv_percent": safe_cv_percent(arr),
    }


def normality_shapiro(values: Sequence[float]) -> Dict[str, object]:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    n = int(len(arr))
    if not SCIPY_AVAILABLE:
        return {
            "n": n,
            "test": "Shapiro-Wilk",
            "statistic": np.nan,
            "p_value": np.nan,
            "normal_p_ge_0_05": np.nan,
            "note": "scipy_no_disponible",
        }
    if n < 3:
        return {
            "n": n,
            "test": "Shapiro-Wilk",
            "statistic": np.nan,
            "p_value": np.nan,
            "normal_p_ge_0_05": np.nan,
            "note": "n_menor_3",
        }
    if np.nanstd(arr) == 0:
        return {
            "n": n,
            "test": "Shapiro-Wilk",
            "statistic": np.nan,
            "p_value": np.nan,
            "normal_p_ge_0_05": np.nan,
            "note": "sin_variabilidad",
        }
    stat, p = shapiro(arr)
    return {
        "n": n,
        "test": "Shapiro-Wilk",
        "statistic": float(stat),
        "p_value": float(p),
        "normal_p_ge_0_05": bool(p >= 0.05),
        "note": "",
    }


# =============================================================================
# LOCALIZACION Y LECTURA DE ARCHIVOS
# =============================================================================

def find_pipeline_outputs(input_dir: Path) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    """
    Busca outputs 02/03 por stem.

    Si encuentra dos archivos con el mismo stem, se detiene en lugar de sobrescribir
    silenciosamente uno de ellos.
    """
    input_dir = Path(input_dir)
    angle_files: Dict[str, Path] = {}
    temporal_files: Dict[str, Path] = {}

    for p in sorted(input_dir.rglob("*_cycle_angle_ranges.csv")):
        stem = clean_stem_from_suffix(p, "_cycle_angle_ranges.csv")
        if stem in angle_files:
            raise ValueError(
                "Hay dos archivos angulares con el mismo dataset_id/stem y no se "
                f"sobrescribirán automáticamente:\n- {angle_files[stem]}\n- {p}"
            )
        angle_files[stem] = p

    for p in sorted(input_dir.rglob("*_gait_temporal_by_cycle.csv")):
        stem = clean_stem_from_suffix(p, "_gait_temporal_by_cycle.csv")
        if stem in temporal_files:
            raise ValueError(
                "Hay dos archivos temporales con el mismo dataset_id/stem y no se "
                f"sobrescribirán automáticamente:\n- {temporal_files[stem]}\n- {p}"
            )
        temporal_files[stem] = p

    return angle_files, temporal_files


def _validate_unique_cycle_keys(df: pd.DataFrame, label: str) -> None:
    keys = [c for c in ["cycle_id", "start_frame", "end_frame"] if c in df.columns]
    if not keys:
        return
    if df.duplicated(subset=keys, keep=False).any():
        dup = df.loc[df.duplicated(subset=keys, keep=False), keys].head(10)
        raise ValueError(
            f"{label} contiene ciclos duplicados para las llaves {keys}. "
            "Se detuvo para evitar multiplicar filas durante la fusión.\n"
            + dup.to_string(index=False)
        )


def read_one_dataset(stem: str, angle_path: Path, temporal_path: Optional[Path]) -> pd.DataFrame:
    """
    Fusiona un dataset por ciclo.

    Los ciclos presentes en la salida del script 02 ya representan los ciclos que
    superaron los criterios técnicos previos del pipeline. El script 04 no vuelve a
    excluir animales ni datasets.

    Para las variables temporales:
      - se conserva el valor original en <variable>_raw;
      - stride_duration_s se conserva para todos los ciclos validados;
      - stance/swing/toe clearance solo quedan disponibles para estadística si
        accepted_temporal == 1.
    """
    angles = pd.read_csv(angle_path)
    animal_id = parse_animal_id(stem)
    if animal_id == "UNKNOWN":
        raise ValueError(
            f"No pude identificar animal_id desde el dataset '{stem}'. "
            "Use nombres tipo 856_P30..., 857_P30... o R1/R2."
        )

    _validate_unique_cycle_keys(angles, f"Archivo angular {angle_path}")

    angle_keep = [
        c for c in ["cycle_id", "start_frame", "end_frame", "duration_frames", "duration_s"]
        if c in angles.columns
    ]
    angle_keep += [c for c in ANGLE_VARIABLES if c in angles.columns]
    merged = angles[angle_keep].copy()

    if temporal_path is not None and temporal_path.exists():
        temporal = pd.read_csv(temporal_path)
        _validate_unique_cycle_keys(temporal, f"Archivo temporal {temporal_path}")

        # Normalizar bandera de QC; si no existe, no inventar rechazo.
        if "accepted_temporal" in temporal.columns:
            temporal["accepted_temporal"] = pd.to_numeric(
                temporal["accepted_temporal"], errors="coerce"
            )
            accepted_mask = temporal["accepted_temporal"].eq(1)
        else:
            temporal["accepted_temporal"] = np.nan
            accepted_mask = pd.Series(True, index=temporal.index)

        # Conservar valores originales y crear columnas de análisis que respetan QC.
        # IMPORTANTE: stride_duration_s depende solo de los límites del ciclo ya
        # validados por el script 01; NO depende de toe-off. Por ello se conserva
        # para todos los ciclos, incluso si accepted_temporal == 0.
        phase_dependent = {
            "stance_duration_s", "swing_duration_s",
            "stance_percent", "swing_percent", "toe_clearance_px",
        }
        for var in TEMPORAL_VARIABLES:
            if var in temporal.columns:
                raw = pd.to_numeric(temporal[var], errors="coerce")
                temporal[f"{var}_raw"] = raw
                temporal[var] = raw.where(accepted_mask, np.nan) if var in phase_dependent else raw
            else:
                temporal[var] = np.nan
                temporal[f"{var}_raw"] = np.nan

        temporal_keep = [
            c for c in ["cycle_id", "start_frame", "end_frame"]
            if c in temporal.columns
        ]
        temporal_keep += [c for c in TEMPORAL_QC_COLUMNS if c in temporal.columns]
        temporal_keep += [c for c in TEMPORAL_VARIABLES if c in temporal.columns]
        temporal_keep += [f"{c}_raw" for c in TEMPORAL_VARIABLES if f"{c}_raw" in temporal.columns]
        temporal_use = temporal[temporal_keep].copy()

        merge_cols = [
            c for c in ["cycle_id", "start_frame", "end_frame"]
            if c in merged.columns and c in temporal_use.columns
        ]
        if not merge_cols and "cycle_id" in merged.columns and "cycle_id" in temporal_use.columns:
            merge_cols = ["cycle_id"]

        if not merge_cols:
            raise ValueError(
                f"No existe llave segura para fusionar ángulos y temporal del dataset {stem}."
            )

        merged = pd.merge(
            merged,
            temporal_use,
            on=merge_cols,
            how="left",
            validate="one_to_one",
            suffixes=("", "_temporal"),
        )
    else:
        for var in TEMPORAL_VARIABLES:
            merged[var] = np.nan
            merged[f"{var}_raw"] = np.nan
        merged["accepted_temporal"] = np.nan
        merged["reject_reason"] = "sin_archivo_temporal"

    merged.insert(0, "animal_id", animal_id)
    merged.insert(1, "dataset_id", stem)
    merged.insert(2, "source_stem", stem)
    merged["angle_file"] = str(angle_path)
    merged["temporal_file"] = str(temporal_path) if temporal_path else ""

    return merged


def load_all_datasets(input_dir: Path) -> pd.DataFrame:
    """Carga TODOS los datasets encontrados, sin selección ni deduplicación."""
    angle_files, temporal_files = find_pipeline_outputs(input_dir)
    if not angle_files:
        raise FileNotFoundError(
            f"No encontré archivos *_cycle_angle_ranges.csv en {input_dir}"
        )

    rows: List[pd.DataFrame] = []
    for stem, angle_path in angle_files.items():
        rows.append(read_one_dataset(stem, angle_path, temporal_files.get(stem)))

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# =============================================================================
# AUDITORIA Y ESTADISTICA
# =============================================================================

def dataset_audit_table(cycles: pd.DataFrame) -> pd.DataFrame:
    """Una fila por dataset; ninguno se selecciona ni elimina automáticamente."""
    rows = []
    for dataset_id, g in cycles.groupby("dataset_id", dropna=False, sort=True):
        animal = str(g["animal_id"].iloc[0]) if len(g) else "UNKNOWN"
        if "accepted_temporal" in g.columns:
            qc = pd.to_numeric(g["accepted_temporal"], errors="coerce")
            n_acc = int(qc.eq(1).sum())
            n_rej = int(qc.eq(0).sum())
            n_unknown = int(qc.isna().sum())
        else:
            n_acc = n_rej = 0
            n_unknown = int(len(g))

        rows.append({
            "animal_id": animal,
            "dataset_id": dataset_id,
            "n_cycles_from_script02": int(len(g)),
            "n_temporal_accepted": n_acc,
            "n_temporal_rejected": n_rej,
            "n_temporal_qc_unknown": n_unknown,
            "status_dataset": "INCLUIDO",
            "selection_rule": "sin_exclusion_automatica_sin_deduplicacion",
        })
    return pd.DataFrame(rows)


def data_retention_table(cycles: pd.DataFrame) -> pd.DataFrame:
    """Resumen por animal para demostrar qué información entra al Excel."""
    rows = []
    for animal, g in cycles.groupby("animal_id", sort=True):
        qc = pd.to_numeric(g.get("accepted_temporal", pd.Series(index=g.index, dtype=float)), errors="coerce")
        rows.append({
            "animal_id": animal,
            "n_datasets": int(g["dataset_id"].nunique()) if "dataset_id" in g.columns else 0,
            "n_cycles_en_excel": int(len(g)),
            "n_temporal_accepted": int(qc.eq(1).sum()),
            "n_temporal_rejected": int(qc.eq(0).sum()),
            "status": "INCLUIDO_TODO_EL_ANIMAL",
            "nota": "QC tecnico se aplica por medicion/ciclo; no se excluye el animal automaticamente",
        })
    return pd.DataFrame(rows)


def animal_variable_long(cycles: pd.DataFrame, variables: List[str]) -> pd.DataFrame:
    """Formato largo solo con mediciones válidas/finitas para análisis descriptivo."""
    id_cols = [c for c in ID_COLUMNS if c in cycles.columns]
    keep_vars = [v for v in variables if v in cycles.columns]
    if not keep_vars:
        return pd.DataFrame()
    long = cycles[id_cols + keep_vars].melt(
        id_vars=id_cols,
        value_vars=keep_vars,
        var_name="variable",
        value_name="value",
    )
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return long.dropna(subset=["value"]).reset_index(drop=True)


def stats_by_animal(cycles: pd.DataFrame, variables: List[str]) -> pd.DataFrame:
    rows = []
    for animal, g in cycles.groupby("animal_id", sort=True):
        for var in variables:
            if var not in g.columns:
                continue
            stats = descriptive_stats(g[var])
            stats.update({"animal_id": animal, "variable": var})
            rows.append(stats)
    cols = [
        "animal_id", "variable", "n", "mean", "sd", "sem", "median",
        "min", "max", "range", "iqr", "cv_percent",
    ]
    return pd.DataFrame(rows)[cols] if rows else pd.DataFrame(columns=cols)


def animal_means_table(cycles: pd.DataFrame, variables: List[str]) -> pd.DataFrame:
    """Una fila por animal; cada variable es la media de sus mediciones válidas."""
    rows = []
    for animal, g in cycles.groupby("animal_id", sort=True):
        row: Dict[str, object] = {"animal_id": animal}
        row["dataset_id"] = ";".join(sorted(map(str, g["dataset_id"].dropna().unique())))
        for var in variables:
            vals = pd.to_numeric(g[var], errors="coerce").dropna() if var in g.columns else pd.Series(dtype=float)
            row[f"{var}_n_cycles"] = int(len(vals))
            row[f"{var}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{var}_sd_intra_animal"] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
            row[f"{var}_sem_intra_animal"] = sem(vals)
        rows.append(row)
    return pd.DataFrame(rows)


def general_stats_from_animal_means(animal_means: pd.DataFrame, variables: List[str]) -> pd.DataFrame:
    """Resumen grupal con n = animal."""
    rows = []
    for var in variables:
        col = f"{var}_mean"
        if col not in animal_means.columns:
            continue
        stats = descriptive_stats(animal_means[col])
        norm = normality_shapiro(animal_means[col])
        row: Dict[str, object] = {"unit": "animal", "variable": var}
        row.update(stats)
        row.update({
            "shapiro_n": norm["n"],
            "shapiro_statistic": norm["statistic"],
            "shapiro_p_value": norm["p_value"],
            "normal_p_ge_0_05": norm["normal_p_ge_0_05"],
            "normality_note": norm["note"],
        })
        rows.append(row)
    return pd.DataFrame(rows)


def normality_by_cycles(cycles: pd.DataFrame, variables: List[str]) -> pd.DataFrame:
    """Normalidad por ciclos como descriptivo; no reemplaza la unidad animal."""
    rows = []
    for var in variables:
        if var not in cycles.columns:
            continue
        norm = normality_shapiro(cycles[var])
        row = {"unit": "cycle_descriptive", "variable": var}
        row.update(norm)
        rows.append(row)
    return pd.DataFrame(rows)


def temporal_qc_table(cycles: pd.DataFrame) -> pd.DataFrame:
    preferred = [c for c in ID_COLUMNS if c in cycles.columns]
    preferred += [c for c in TEMPORAL_QC_COLUMNS if c in cycles.columns]
    preferred += [c for c in TEMPORAL_VARIABLES if c in cycles.columns]
    preferred += [f"{c}_raw" for c in TEMPORAL_VARIABLES if f"{c}_raw" in cycles.columns]
    return cycles[preferred].copy() if preferred else pd.DataFrame()


# =============================================================================
# EXCEL
# =============================================================================

def autosize_excel_columns(
    writer: pd.ExcelWriter,
    sheet_name: str,
    df: pd.DataFrame,
    max_width: int = 42,
) -> None:
    worksheet = writer.sheets[sheet_name]
    for idx, col in enumerate(df.columns):
        sample = df[col].astype(str).replace("nan", "").head(100).tolist()
        width = max([len(str(col))] + [len(x) for x in sample]) + 2
        worksheet.set_column(idx, idx, min(max(width, 10), max_width))


def write_excel(out_path: Path, sheets: Dict[str, pd.DataFrame], metadata: Dict[str, object]) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        workbook = writer.book
        title_fmt = workbook.add_format({"bold": True, "font_size": 14, "bg_color": "#D9EAF7", "border": 1})
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#EAF3F8", "border": 1})
        note_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
        num_fmt = workbook.add_format({"num_format": "0.000"})

        meta_df = pd.DataFrame([{"campo": k, "valor": v} for k, v in metadata.items()])
        meta_df.to_excel(writer, sheet_name="README", index=False, startrow=3)
        ws = writer.sheets["README"]
        ws.write("A1", "Validación estadística del pipeline DLC - P30", title_fmt)
        ws.write(
            "A2",
            "Todos los animales/datasets encontrados se conservan. El QC técnico se aplica por medición; la unidad inferencial principal es el animal.",
            note_fmt,
        )
        ws.set_column("A:A", 36)
        ws.set_column("B:B", 100)
        ws.freeze_panes(4, 0)

        for sheet_name, df in sheets.items():
            safe_name = sheet_name[:31]
            if df is None or df.empty:
                empty_df = pd.DataFrame({"mensaje": ["Sin datos para esta hoja"]})
                empty_df.to_excel(writer, sheet_name=safe_name, index=False)
                autosize_excel_columns(writer, safe_name, empty_df)
                continue

            df.to_excel(writer, sheet_name=safe_name, index=False)
            ws = writer.sheets[safe_name]
            for col_idx, col in enumerate(df.columns):
                ws.write(0, col_idx, col, header_fmt)
            autosize_excel_columns(writer, safe_name, df)
            ws.freeze_panes(1, 0)
            ws.autofilter(0, 0, len(df), max(0, len(df.columns) - 1))
            for col_idx, col in enumerate(df.columns):
                if pd.api.types.is_numeric_dtype(df[col]):
                    ws.set_column(col_idx, col_idx, None, num_fmt)


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================

def build_validation_workbook(
    input_dir: Path,
    out_path: Path,
    treadmill_speed_cm_s: float = 20.0,
    sex: str = "male",
    litter_id: str = "P30_single_shared_litter",
) -> Dict[str, pd.DataFrame]:
    all_cycles = load_all_datasets(input_dir)
    variables = [v for v in DEFAULT_VARIABLES if v in all_cycles.columns]

    dataset_audit = dataset_audit_table(all_cycles)
    retention = data_retention_table(all_cycles)
    cycle_long = animal_variable_long(all_cycles, variables)
    stats_animal = stats_by_animal(all_cycles, variables)
    animal_means = animal_means_table(all_cycles, variables)
    general_stats = general_stats_from_animal_means(animal_means, variables)
    normality_cycles = normality_by_cycles(all_cycles, variables)
    temporal_qc = temporal_qc_table(all_cycles)

    preferred_cols = [c for c in ID_COLUMNS if c in all_cycles.columns] + variables
    qc_and_raw = [
        c for c in TEMPORAL_QC_COLUMNS + [f"{v}_raw" for v in TEMPORAL_VARIABLES]
        if c in all_cycles.columns and c not in preferred_cols
    ]
    other_cols = [c for c in all_cycles.columns if c not in preferred_cols + qc_and_raw]
    cycles_sheet = all_cycles[preferred_cols + qc_and_raw + other_cols].copy()

    sheets = {
        "dataset_audit_all": dataset_audit,
        "data_retention_all": retention,
        "cycles_individual_all": cycles_sheet,
        "temporal_qc_all": temporal_qc,
        "valid_measurements_long": cycle_long,
        "stats_by_animal": stats_animal,
        "animal_means": animal_means,
        "general_stats_n_animal": general_stats,
        "normality_cycles_desc": normality_cycles,
    }

    n_animals = int(all_cycles["animal_id"].nunique()) if "animal_id" in all_cycles.columns else 0
    n_datasets = int(all_cycles["dataset_id"].nunique()) if "dataset_id" in all_cycles.columns else 0

    metadata = {
        "input_dir": str(Path(input_dir).resolve()),
        "output_file": str(Path(out_path).resolve()),
        "n_animals_found": n_animals,
        "n_datasets_found": n_datasets,
        "animal_exclusion": "NINGUNA_AUTOMATICA",
        "dataset_exclusion": "NINGUNA_AUTOMATICA",
        "dataset_deduplication": "DESACTIVADA_NO_EXISTE_EN_ESTA_VERSION",
        "technical_qc": (
            "stride_duration_s usa todos los ciclos validados; stance/swing/toe_clearance "
            "solo usan ciclos con accepted_temporal == 1"
        ),
        "treadmill_speed_cm_s": treadmill_speed_cm_s,
        "sex": sex,
        "litter_id": litter_id,
        "litter_model_note": (
            "Todos los animales pertenecen a una sola camada. No es posible estimar "
            "un efecto aleatorio de camada con un solo nivel; la inferencia se interpreta "
            "condicionada a esta camada y requiere replicacion en camadas independientes "
            "para generalizacion poblacional."
        ),
        "raw_temporal_preserved": "SI_en_columnas_sufijo_raw",
        "variables": ", ".join(variables),
        "statistical_unit_main": "animal",
        "note": (
            "cycles_individual_all conserva todos los ciclos entregados por script 02. "
            "temporal_qc_all conserva flags y valores raw. Las estadísticas usan valores finitos/validos por variable; "
            "no eliminan animales completos automáticamente."
        ),
    }

    write_excel(out_path=out_path, sheets=sheets, metadata=metadata)
    return sheets


def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Genera Excel de validación estadística desde outputs 02/03. "
            "No excluye animales ni datasets automáticamente."
        )
    )
    p.add_argument(
        "--input-dir",
        required=True,
        help="Carpeta raíz donde están las salidas de los scripts 02 y 03.",
    )
    p.add_argument(
        "--out",
        default="validacion_estadistica_dlc.xlsx",
        help="Ruta del Excel de salida.",
    )
    p.add_argument("--treadmill-speed-cm-s", type=float, default=20.0,
                   help="Metadato experimental. P30 Day 1: 20 cm/s.")
    p.add_argument("--sex", type=str, default="male",
                   help="Metadato experimental. P30 Day 1: todos machos.")
    p.add_argument("--litter-id", type=str, default="P30_single_shared_litter",
                   help="Metadato experimental. P30 Day 1: una sola camada.")
    return p


def main() -> None:
    args = make_argparser().parse_args()
    sheets = build_validation_workbook(
        input_dir=Path(args.input_dir),
        out_path=Path(args.out),
        treadmill_speed_cm_s=args.treadmill_speed_cm_s,
        sex=args.sex,
        litter_id=args.litter_id,
    )
    print("Excel generado correctamente:", args.out)
    print("Hojas generadas:", ", ".join(sheets.keys()))
    print("Política: sin exclusión automática de animales/datasets; QC técnico por medición.")


if __name__ == "__main__":
    main()
