# P30 gait-analysis pipeline — audited final code

This package contains the frozen code after the Day 1 methodological audit.

## Experimental design

- Treadmill: **20 cm/s**
- Sex: **all male**
- Litter: **all siblings from one shared litter**
- WT: `860, 861, 863, 867`
- SOD1: `856, 857, 859, 862, 864, 865`
- Experimental acquisition: **60 Hz**

### Important consequence of the single litter

A random litter effect **must not** be fitted with only one litter level; it is not identifiable.
The genotype contrast is therefore a within-litter comparison among siblings. This controls the
litter environment inside this cohort, but population-level claims require replication in
independent litters.

## Scripts

1. `01_preprocesamiento_y_ciclos.py`
   - Frozen multichannel detector.
   - Toe relative to hip is the primary event signal.
   - Knee angle is strong confirmation.
   - Knee-relative + hip-angle is the alternative confirmation route.
   - No minimum target number of cycles and no synthetic events.
   - 5-frame smoothing for both detection and exported analysis coordinates.

2. `02_goniometria_lateral_por_ciclos.py`
   - Compatible without algorithmic change.
   - Internal angles: hip, knee, ankle, foot.
   - Uses the validated cycles from script 01.

3. `03_variables_temporales_y_toe_clearance.py`
   - Corrected for rapid 5–8 frame cycles.
   - Minimum estimated stance/swing: 2 frames.
   - Sustained toe-off confirmation: 2 frames.
   - `stride_duration_s` is always defined from cycle limits.
   - Stance/swing/toe-clearance are **kinematic estimates**, not force-plate ground truth.

4. `04_validacion_estadistica_y_excel.py`
   - Audit correction: `stride_duration_s` is retained for every validated cycle.
   - Only phase-dependent variables are masked when `accepted_temporal != 1`.
   - No automatic exclusion of animals or datasets.
   - Main descriptive/inferential unit: animal.

5. `05_analisis_WT_vs_SOD1.py`
   - Exact two-sided **studentized permutation** using Welch t as the primary statistic.
   - Enumerates all 210 allocations for 4 WT among 10 animals.
   - BH-FDR correction.
   - Reports effect sizes.
   - Recomputes intra-animal variability using all valid stride cycles.
   - Explicitly documents that one litter cannot support a litter random effect.


