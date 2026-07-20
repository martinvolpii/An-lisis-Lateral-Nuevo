#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_graficos_goniometria_top10.py

Quinta parte del pipeline P30: visualización goniométrica individual y grupal.

Trabaja directamente con las salidas del script 02:
    *_cycle_angle_profiles.csv
    *_cycle_angle_ranges.csv

El script 02 ya utiliza los ciclos validados por el script 01. El script 03
no se usa para excluir perfiles goniométricos, porque su QC depende de toe-off
y corresponde a variables temporales, no a la validez angular.

Criterio Top 10 (inspirado SOLO en la idea del código lateral antiguo):
    - Para CADA articulación por separado se seleccionan los N ciclos con el
      ángulo máximo más alto (N=10 por defecto).
    - Cadera, rodilla, tobillo y pie pueden seleccionar ciclos distintos.

Principios metodológicos:
    - Conserva los valores angulares reales; no genera datos sintéticos.
    - No aplica t-test, ANOVA ni ninguna prueba inferencial.
    - No recorta artificialmente ángulos >120°.
    - Los gráficos grupales se construyen en dos niveles:
          ciclos Top N -> promedio por animal -> promedio del grupo.
      Por tanto, cada animal aporta una sola curva al promedio grupal.
    - La banda grupal es ±1 D.E. ENTRE ANIMALES y es solo descriptiva.

Archivo de grupos:
    CSV con dos columnas obligatorias:
        animal_id,group

    Ejemplo:
        animal_id,group
        856,WT
        857,SOD1

    Se puede pasar con --groups-csv. Si se omite, el script busca groups.csv
    dentro de --input-dir. Si no existe, genera solo gráficos individuales y
    el resumen general de todos los animales.

Salidas:
    Individual/
        <animal>/goniometria_top10.png|pdf
    Grupal comparativo/
        <grupo>/goniometria_top10_<grupo>.png|pdf
        WT vs SOD1/comparacion_goniometria_top10.png|pdf (si existen ambos)
    Tablas/
        ciclos_top10_seleccionados.csv
        perfiles_promedio_por_animal.csv
        perfiles_promedio_por_grupo.csv
    resumen_goniometria_top10_todos_animales.png|pdf

Uso:
    python 05_graficos_goniometria_top10.py \
        --input-dir resultados_pipeline \
        --groups-csv groups.csv \
        --outdir resultados_05

Opciones:
    --top-n 10
    --plot-points 101
    --no-sd
    --show
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROFILE_SUFFIX = "_cycle_angle_profiles.csv"
RANGE_SUFFIX = "_cycle_angle_ranges.csv"

JOINTS: Dict[str, Dict[str, str]] = {
    "hip": {
        "label": "Cadera",
        "profile_col": "hip_angle_deg",
        "max_col": "hip_max_deg",
        "color": "tab:blue",
    },
    "knee": {
        "label": "Rodilla",
        "profile_col": "knee_angle_deg",
        "max_col": "knee_max_deg",
        "color": "tab:orange",
    },
    "ankle": {
        "label": "Tobillo",
        "profile_col": "ankle_angle_deg",
        "max_col": "ankle_max_deg",
        "color": "tab:green",
    },
    "foot": {
        "label": "Pie",
        "profile_col": "foot_angle_deg",
        "max_col": "foot_max_deg",
        "color": "tab:red",
    },
}

# Colores de grupos para la figura WT vs SOD1.
# Se usan solo para distinguir grupos; no alteran los datos.
GROUP_COLORS = {
    "WT": "#0072B2",
    "SOD1": "#D55E00",
}


@dataclass(frozen=True)
class DatasetFiles:
    dataset_id: str
    animal_id: str
    profiles_path: Path
    ranges_path: Path


def clean_stem_from_suffix(path: Path, suffix: str) -> str:
    name = path.name
    return name[: -len(suffix)] if name.endswith(suffix) else path.stem


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")
    return text or "dataset"


def parse_animal_id(text: str) -> str:
    """Extrae IDs tipo 856_P30..., 857_... o R1/R2."""
    text = str(text)
    m = re.search(r"(?:^|[^0-9])(\d+)_P\d+", text, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"(?:^|[^A-Za-z0-9])(R\d+)(?:[^A-Za-z0-9]|$)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"(^|[^0-9])(\d{2,})(?=[_-])", text)
    if m:
        return m.group(2)
    return safe_name(text)


def discover_datasets(input_dir: Path) -> List[DatasetFiles]:
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"No existe --input-dir: {input_dir}")

    profile_files = sorted(input_dir.rglob(f"*{PROFILE_SUFFIX}"))
    if not profile_files:
        raise FileNotFoundError(
            f"No se encontraron archivos *{PROFILE_SUFFIX} dentro de {input_dir}"
        )

    range_files = sorted(input_dir.rglob(f"*{RANGE_SUFFIX}"))
    ranges_by_stem: Dict[str, List[Path]] = {}
    for p in range_files:
        stem = clean_stem_from_suffix(p, RANGE_SUFFIX)
        ranges_by_stem.setdefault(stem, []).append(p)

    datasets: List[DatasetFiles] = []
    seen = set()
    for profile_path in profile_files:
        stem = clean_stem_from_suffix(profile_path, PROFILE_SUFFIX)
        sibling = profile_path.with_name(stem + RANGE_SUFFIX)
        if sibling.exists():
            ranges_path = sibling
        else:
            candidates = ranges_by_stem.get(stem, [])
            if len(candidates) == 1:
                ranges_path = candidates[0]
            elif not candidates:
                print(f"ADVERTENCIA: sin archivo de rangos para {profile_path}; se omite.")
                continue
            else:
                raise ValueError(
                    f"Hay varios archivos de rangos para '{stem}'. Mantenga cada par "
                    "profiles/ranges en la misma carpeta."
                )

        key = (profile_path.resolve(), ranges_path.resolve())
        if key in seen:
            continue
        seen.add(key)

        dataset_id = safe_name(stem)
        animal_id = parse_animal_id(stem)
        datasets.append(DatasetFiles(dataset_id, animal_id, profile_path, ranges_path))

    if not datasets:
        raise ValueError("No se encontraron pares válidos profiles/ranges.")
    return datasets


def read_profiles(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"cycle_id", "percent_gait_cycle"}
    required.update(cfg["profile_col"] for cfg in JOINTS.values())
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} no contiene: {', '.join(missing)}")

    df["cycle_id"] = pd.to_numeric(df["cycle_id"], errors="coerce")
    df["percent_gait_cycle"] = pd.to_numeric(df["percent_gait_cycle"], errors="coerce")
    df = df.dropna(subset=["cycle_id", "percent_gait_cycle"]).copy()
    df["cycle_id"] = df["cycle_id"].astype(int)
    for cfg in JOINTS.values():
        df[cfg["profile_col"]] = pd.to_numeric(df[cfg["profile_col"]], errors="coerce")
    return df.sort_values(["cycle_id", "percent_gait_cycle"]).reset_index(drop=True)


def read_ranges(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"cycle_id"}
    required.update(cfg["max_col"] for cfg in JOINTS.values())
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} no contiene: {', '.join(missing)}")

    df["cycle_id"] = pd.to_numeric(df["cycle_id"], errors="coerce")
    df = df.dropna(subset=["cycle_id"]).copy()
    df["cycle_id"] = df["cycle_id"].astype(int)
    for cfg in JOINTS.values():
        df[cfg["max_col"]] = pd.to_numeric(df[cfg["max_col"]], errors="coerce")
    if df["cycle_id"].duplicated().any():
        raise ValueError(f"{path} contiene cycle_id duplicados.")
    return df.reset_index(drop=True)


def read_groups(groups_csv: Optional[Path], input_dir: Path) -> Dict[str, str]:
    """Lee animal_id -> grupo. Si no hay archivo, devuelve un diccionario vacío."""
    path = Path(groups_csv) if groups_csv is not None else Path(input_dir) / "groups.csv"
    if not path.exists():
        return {}

    df = pd.read_csv(path, dtype=str)
    required = {"animal_id", "group"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{path} debe contener las columnas animal_id,group. Faltan: {', '.join(sorted(missing))}"
        )
    df = df[["animal_id", "group"]].dropna().copy()
    df["animal_id"] = df["animal_id"].astype(str).str.strip()
    df["group"] = df["group"].astype(str).str.strip()
    df = df[(df["animal_id"] != "") & (df["group"] != "")]
    if df["animal_id"].duplicated().any():
        dup = df.loc[df["animal_id"].duplicated(keep=False), "animal_id"].tolist()
        raise ValueError(f"IDs repetidos en groups CSV: {dup}")
    return dict(zip(df["animal_id"], df["group"]))


def select_top_cycles(ranges: pd.DataFrame, joint: str, top_n: int) -> pd.DataFrame:
    if joint not in JOINTS:
        raise KeyError(joint)
    if top_n < 1:
        raise ValueError("top_n debe ser >= 1")
    max_col = JOINTS[joint]["max_col"]
    keep = [
        c for c in ["cycle_id", "start_frame", "end_frame", "duration_frames", "duration_s", max_col]
        if c in ranges.columns
    ]
    temp = ranges[keep].copy()
    temp = temp[np.isfinite(pd.to_numeric(temp[max_col], errors="coerce"))].copy()
    temp = temp.sort_values([max_col, "cycle_id"], ascending=[False, True], kind="mergesort")
    temp = temp.head(int(top_n)).reset_index(drop=True)
    temp["selection_rank"] = np.arange(1, len(temp) + 1, dtype=int)
    return temp


def interpolate_curve(x: Sequence[float], y: Sequence[float], target_x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return np.full_like(target_x, np.nan, dtype=float)
    temp = pd.DataFrame({"x": x[finite], "y": y[finite]}).groupby("x", as_index=False)["y"].mean()
    if len(temp) < 2:
        return np.full_like(target_x, np.nan, dtype=float)
    return np.interp(target_x, temp["x"].to_numpy(), temp["y"].to_numpy())


def mean_top_profile(
    profiles: pd.DataFrame,
    selected: pd.DataFrame,
    angle_col: str,
    target_x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve media, SD entre ciclos y matriz de ciclos interpolados."""
    curves = []
    for cycle_id in selected["cycle_id"].astype(int):
        cyc = profiles[profiles["cycle_id"] == cycle_id].sort_values("percent_gait_cycle")
        interp = interpolate_curve(
            cyc["percent_gait_cycle"].to_numpy(),
            cyc[angle_col].to_numpy(),
            target_x,
        )
        if np.isfinite(interp).sum() >= 2:
            curves.append(interp)
    if not curves:
        nan = np.full_like(target_x, np.nan, dtype=float)
        return nan, nan, np.empty((0, len(target_x)))
    arr = np.vstack(curves)
    mean = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.full_like(mean, np.nan)
    return mean, sd, arr


def plot_individual(
    ds: DatasetFiles,
    profiles: pd.DataFrame,
    ranges: pd.DataFrame,
    top_n: int,
    target_x: np.ndarray,
    output_base: Path,
    draw_sd: bool,
    show: bool,
) -> Tuple[List[Dict[str, object]], Dict[str, np.ndarray], List[Dict[str, object]]]:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    audit: List[Dict[str, object]] = []
    animal_curves: Dict[str, np.ndarray] = {}
    profile_rows: List[Dict[str, object]] = []

    for ax, (joint, cfg) in zip(axes, JOINTS.items()):
        selected = select_top_cycles(ranges, joint, top_n)
        mean, sd, arr = mean_top_profile(profiles, selected, cfg["profile_col"], target_x)

        if arr.shape[0] == 0:
            ax.text(0.5, 0.5, "Sin ciclos válidos", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(cfg["label"])
            continue

        for curve in arr:
            ax.plot(target_x, curve, color=cfg["color"], linewidth=1.0, alpha=0.28)
        if draw_sd and np.isfinite(sd).any():
            ax.fill_between(target_x, mean - sd, mean + sd, color=cfg["color"], alpha=0.16, linewidth=0)
        ax.plot(target_x, mean, color=cfg["color"], linewidth=3.0, label=f"Promedio top {arr.shape[0]}")
        ax.set_title(f"{cfg['label']} | {arr.shape[0]} ciclos con mayor máximo", fontweight="bold")
        ax.set_xlabel("Ciclo de marcha (%)")
        ax.set_ylabel("Ángulo (°)")
        ax.set_xlim(0, 100)
        ax.grid(True, alpha=0.22)
        ax.legend(loc="best", frameon=False)

        animal_curves[joint] = mean
        for p, value in zip(target_x, mean):
            profile_rows.append({
                "animal_id": ds.animal_id,
                "dataset_id": ds.dataset_id,
                "joint": joint,
                "joint_label": cfg["label"],
                "percent_gait_cycle": float(p),
                "mean_angle_deg": float(value),
                "n_selected_cycles": int(arr.shape[0]),
            })

        for _, row in selected.iterrows():
            item: Dict[str, object] = {
                "animal_id": ds.animal_id,
                "dataset_id": ds.dataset_id,
                "joint": joint,
                "joint_label": cfg["label"],
                "selection_rule": f"top_{top_n}_by_{cfg['max_col']}",
                "selection_rank": int(row["selection_rank"]),
                "cycle_id": int(row["cycle_id"]),
                "max_angle_deg": float(row[cfg["max_col"]]),
            }
            for col in ["start_frame", "end_frame", "duration_frames", "duration_s"]:
                if col in row.index and pd.notna(row[col]):
                    item[col] = row[col]
            audit.append(item)

    fig.suptitle(
        f"Goniometría lateral — Top {top_n} por articulación\nAnimal {ds.animal_id}",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return audit, animal_curves, profile_rows


def collapse_datasets_to_animals(
    dataset_curves: Dict[str, Tuple[str, Dict[str, np.ndarray]]]
) -> Dict[str, Dict[str, np.ndarray]]:
    """Si un animal tiene varios datasets, primero los promedia para darle peso 1."""
    by_animal: Dict[str, Dict[str, List[np.ndarray]]] = {}
    for _, (animal_id, joint_curves) in dataset_curves.items():
        for joint, curve in joint_curves.items():
            by_animal.setdefault(animal_id, {}).setdefault(joint, []).append(curve)

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for animal_id, joint_map in by_animal.items():
        out[animal_id] = {}
        for joint, curves in joint_map.items():
            arr = np.vstack(curves)
            out[animal_id][joint] = np.nanmean(arr, axis=0)
    return out


def plot_group(
    group_name: str,
    animal_curves: Dict[str, Dict[str, np.ndarray]],
    target_x: np.ndarray,
    top_n: int,
    output_base: Path,
    draw_sd: bool,
    show: bool,
) -> List[Dict[str, object]]:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    summary_rows: List[Dict[str, object]] = []

    for ax, (joint, cfg) in zip(axes, JOINTS.items()):
        valid = [(animal, curves[joint]) for animal, curves in animal_curves.items() if joint in curves]
        if not valid:
            ax.text(0.5, 0.5, "Sin animales válidos", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(cfg["label"])
            continue
        arr = np.vstack([curve for _, curve in valid])
        mean = np.nanmean(arr, axis=0)
        sd = np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.full_like(mean, np.nan)

        for _, curve in valid:
            ax.plot(target_x, curve, color=cfg["color"], linewidth=1.0, alpha=0.22)
        if draw_sd and arr.shape[0] > 1 and np.isfinite(sd).any():
            ax.fill_between(target_x, mean - sd, mean + sd, color=cfg["color"], alpha=0.18, linewidth=0)
        ax.plot(target_x, mean, color=cfg["color"], linewidth=3.2, label=f"Media {group_name} (n={arr.shape[0]} animales)")
        ax.set_title(cfg["label"], fontweight="bold")
        ax.set_xlabel("Ciclo de marcha (%)")
        ax.set_ylabel("Ángulo (°)")
        ax.set_xlim(0, 100)
        ax.grid(True, alpha=0.22)
        ax.legend(loc="best", frameon=False)

        for p, m, s in zip(target_x, mean, sd):
            summary_rows.append({
                "group": group_name,
                "joint": joint,
                "joint_label": cfg["label"],
                "percent_gait_cycle": float(p),
                "mean_angle_deg": float(m),
                "sd_between_animals_deg": float(s) if np.isfinite(s) else np.nan,
                "n_animals": int(arr.shape[0]),
                "top_n_per_joint_per_animal": int(top_n),
            })

    fig.suptitle(
        f"Goniometría lateral — {group_name}\n"
        f"Cada animal aporta el promedio de sus Top {top_n} ciclos",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return summary_rows


def plot_group_comparison(
    group_animals: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    target_x: np.ndarray,
    top_n: int,
    output_base: Path,
    draw_sd: bool,
    show: bool,
) -> None:
    """Superpone los promedios grupales. Figura descriptiva, sin p-values."""
    groups = list(group_animals.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    fallback_colors = ["tab:blue", "tab:red", "tab:green", "tab:purple", "tab:brown"]
    for ax, (joint, cfg) in zip(axes, JOINTS.items()):
        plotted = 0
        for g_idx, group in enumerate(groups):
            curves = [jc[joint] for jc in group_animals[group].values() if joint in jc]
            if not curves:
                continue
            arr = np.vstack(curves)
            mean = np.nanmean(arr, axis=0)
            sd = np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.full_like(mean, np.nan)
            color = GROUP_COLORS.get(group, fallback_colors[g_idx % len(fallback_colors)])
            if draw_sd and arr.shape[0] > 1 and np.isfinite(sd).any():
                ax.fill_between(target_x, mean - sd, mean + sd, color=color, alpha=0.12, linewidth=0)
            ax.plot(target_x, mean, color=color, linewidth=3.0, label=f"{group} (n={arr.shape[0]})")
            plotted += 1

        if plotted == 0:
            ax.text(0.5, 0.5, "Sin grupos válidos", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(cfg["label"], fontweight="bold")
        ax.set_xlabel("Ciclo de marcha (%)")
        ax.set_ylabel("Ángulo (°)")
        ax.set_xlim(0, 100)
        ax.grid(True, alpha=0.22)
        if plotted:
            ax.legend(loc="best", frameon=False)

    fig.suptitle(
        f"Comparación descriptiva de goniometría — Top {top_n}\n"
        "Promedio por animal antes del promedio grupal | Sin estadística inferencial",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_all_animals(
    animal_curves: Dict[str, Dict[str, np.ndarray]],
    target_x: np.ndarray,
    top_n: int,
    output_base: Path,
    draw_sd: bool,
    show: bool,
) -> None:
    """Resumen descriptivo general, ignorando grupo experimental."""
    plot_group(
        "Todos los animales",
        animal_curves,
        target_x,
        top_n,
        output_base,
        draw_sd,
        show,
    )


def run_pipeline(
    input_dir: Path,
    outdir: Path,
    groups_csv: Optional[Path] = None,
    top_n: int = 10,
    plot_points: int = 101,
    draw_sd: bool = True,
    show: bool = False,
) -> Dict[str, object]:
    input_dir = Path(input_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    individual_dir = outdir / "Individual"
    group_dir = outdir / "Grupal comparativo"
    tables_dir = outdir / "Tablas"
    individual_dir.mkdir(parents=True, exist_ok=True)
    group_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    datasets = discover_datasets(input_dir)
    group_map = read_groups(groups_csv, input_dir)
    target_x = np.linspace(0.0, 100.0, int(plot_points))

    print("\n=== 05 GONIOMETRÍA TOP-N: INDIVIDUAL + GRUPAL ===")
    print(f"Entrada:                  {input_dir}")
    print(f"Salida:                   {outdir}")
    print(f"Datasets encontrados:     {len(datasets)}")
    print(f"Top ciclos/articulación:  {top_n}")
    print("Regla:                    mayor ángulo MÁXIMO por articulación")
    print("Unidad grupal:            animal")
    print("Estadística inferencial:  NINGUNA")
    print("Datos sintéticos:         NINGUNO")
    print(f"Mapa de grupos:           {'sí' if group_map else 'no'}\n")

    audit_rows: List[Dict[str, object]] = []
    animal_profile_rows: List[Dict[str, object]] = []
    dataset_curves: Dict[str, Tuple[str, Dict[str, np.ndarray]]] = {}

    for i, ds in enumerate(datasets, start=1):
        print(f"[{i}/{len(datasets)}] Animal {ds.animal_id} | {ds.dataset_id}")
        profiles = read_profiles(ds.profiles_path)
        ranges = read_ranges(ds.ranges_path)

        animal_out = individual_dir / safe_name(ds.animal_id)
        output_base = animal_out / f"goniometria_top{top_n}"
        audit, curves, profile_rows = plot_individual(
            ds, profiles, ranges, top_n, target_x, output_base, draw_sd, show
        )
        group = group_map.get(ds.animal_id, "")
        for row in audit:
            row["group"] = group
            row["profiles_file"] = str(ds.profiles_path)
            row["ranges_file"] = str(ds.ranges_path)
        for row in profile_rows:
            row["group"] = group

        audit_rows.extend(audit)
        animal_profile_rows.extend(profile_rows)
        dataset_curves[ds.dataset_id] = (ds.animal_id, curves)

    # Promedia datasets repetidos dentro del mismo animal antes de cualquier resumen grupal.
    animal_curves = collapse_datasets_to_animals(dataset_curves)

    audit_df = pd.DataFrame(audit_rows)
    audit_csv = tables_dir / "ciclos_top10_seleccionados.csv"
    audit_df.to_csv(audit_csv, index=False)

    animal_profiles_df = pd.DataFrame(animal_profile_rows)
    animal_profiles_csv = tables_dir / "perfiles_promedio_por_animal.csv"
    animal_profiles_df.to_csv(animal_profiles_csv, index=False)

    general_base = outdir / f"resumen_goniometria_top{top_n}_todos_animales"
    plot_all_animals(animal_curves, target_x, top_n, general_base, draw_sd, show)

    group_summary_rows: List[Dict[str, object]] = []
    grouped_animals: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

    if group_map:
        missing_group_ids = sorted(a for a in animal_curves if a not in group_map)
        if missing_group_ids:
            print(
                "ADVERTENCIA: estos animales no tienen grupo y se excluyen SOLO de las "
                "figuras grupales: " + ", ".join(missing_group_ids)
            )

        for animal_id, curves in animal_curves.items():
            group = group_map.get(animal_id)
            if not group:
                continue
            grouped_animals.setdefault(group, {})[animal_id] = curves

        for group, curves_by_animal in sorted(grouped_animals.items()):
            group_safe = safe_name(group)
            group_base = group_dir / group_safe / f"goniometria_top{top_n}_{group_safe}"
            rows = plot_group(
                group, curves_by_animal, target_x, top_n, group_base, draw_sd, show
            )
            group_summary_rows.extend(rows)

        if len(grouped_animals) >= 2:
            names = list(sorted(grouped_animals.keys()))
            if "WT" in grouped_animals and "SOD1" in grouped_animals:
                comparison_name = "WT vs SOD1"
                comparison_groups = {
                    "WT": grouped_animals["WT"],
                    "SOD1": grouped_animals["SOD1"],
                }
            else:
                comparison_name = " vs ".join(names[:2])
                comparison_groups = {name: grouped_animals[name] for name in names[:2]}

            comp_base = group_dir / comparison_name / f"comparacion_goniometria_top{top_n}"
            plot_group_comparison(
                comparison_groups, target_x, top_n, comp_base, draw_sd, show
            )

    group_summary_df = pd.DataFrame(group_summary_rows)
    group_summary_csv = tables_dir / "perfiles_promedio_por_grupo.csv"
    group_summary_df.to_csv(group_summary_csv, index=False)

    params = outdir / "metodo_goniometria_top10.txt"
    with open(params, "w", encoding="utf-8") as f:
        f.write("05_graficos_goniometria_top10.py\n")
        f.write(f"input_dir = {input_dir}\n")
        f.write(f"outdir = {outdir}\n")
        f.write(f"groups_csv = {groups_csv}\n")
        f.write(f"top_n = {top_n}\n")
        f.write(f"plot_points = {plot_points}\n")
        f.write(f"draw_sd = {draw_sd}\n")
        f.write(f"n_datasets = {len(datasets)}\n")
        f.write(f"n_animals = {len(animal_curves)}\n")
        f.write("selection = top N cycles by joint-specific maximum angle\n")
        f.write("joint_selection_is_independent = True\n")
        f.write("group_aggregation = cycles_to_animal_mean_then_group_mean\n")
        f.write("inferential_statistics = None\n")
        f.write("synthetic_random_data = None\n")
        f.write("angle_cap_120_deg = False\n")
        f.write("temporal_qc_used_for_goniometry_selection = False\n")

    print("\nGenerado:")
    print(f"  {individual_dir}")
    print(f"  {group_dir}")
    print(f"  {audit_csv}")
    print(f"  {animal_profiles_csv}")
    print(f"  {group_summary_csv}")
    print(f"  {general_base.with_suffix('.png')}")
    print(f"  {general_base.with_suffix('.pdf')}")

    return {
        "individual_dir": individual_dir,
        "group_dir": group_dir,
        "audit_csv": audit_csv,
        "animal_profiles_csv": animal_profiles_csv,
        "group_profiles_csv": group_summary_csv,
        "general_png": general_base.with_suffix(".png"),
        "general_pdf": general_base.with_suffix(".pdf"),
        "n_animals": len(animal_curves),
        "n_groups": len(grouped_animals),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Genera gráficos goniométricos individuales y grupales usando los N ciclos "
            "con mayor ángulo máximo por articulación a partir de las salidas del script 02."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Carpeta raíz con *_cycle_angle_profiles.csv y *_cycle_angle_ranges.csv.")
    parser.add_argument("--groups-csv", type=Path, default=None,
                        help="CSV animal_id,group. Si se omite, busca groups.csv en --input-dir.")
    parser.add_argument("--outdir", type=Path, default=Path("resultados_05_goniometria"),
                        help="Carpeta de salida.")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Número de ciclos con mayor máximo por articulación (default: 10).")
    parser.add_argument("--plot-points", type=int, default=101,
                        help="Puntos del eje normalizado 0-100 (default: 101).")
    parser.add_argument("--no-sd", action="store_true",
                        help="No dibujar bandas descriptivas de ±1 D.E.")
    parser.add_argument("--show", action="store_true",
                        help="Mostrar figuras además de guardarlas.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    if args.top_n < 1:
        parser.error("--top-n debe ser >= 1")
    if args.plot_points < 2:
        parser.error("--plot-points debe ser >= 2")
    try:
        run_pipeline(
            input_dir=args.input_dir,
            outdir=args.outdir,
            groups_csv=args.groups_csv,
            top_n=args.top_n,
            plot_points=args.plot_points,
            draw_sd=not args.no_sd,
            show=args.show,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
