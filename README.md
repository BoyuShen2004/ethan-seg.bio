# SegBioModel + nnU-Net v2 (SLURM) — Backend API & How-To

This repo wraps an nnU-Net v2 training/prediction workflow behind a small FastAPI “backend API” and a few SLURM templates. It’s designed for an HPC cluster with SLURM and nnU-Net v2 installed on the worker nodes.

---

## What’s in this repo

```
.
└── scripts/
    ├── SegBioModel.py                # Core pipeline class that submits SLURM jobs
    ├── api.py                        # FastAPI app exposing training/inference endpoints
    └── nnunet_slurm_templates/
        ├── nnunet_train_template.sl
        ├── nnunet_verify_template.sl
        └── nnunet_predict_template.sl
```

> If your copy doesn’t already have the `scripts/nnunet/` folder, create it and place the three `*.sl` templates there. Then **edit the hardcoded paths** at the top of `SegBioModel.py` to point to your copies (details below).

---

## Required environment (cluster side)

- **SLURM** available (`sbatch`, `srun`).
- **nnU-Net v2** available on compute nodes (`nnUNetv2_*` in `$PATH`), with working CUDA.
- A project storage layout (change these to your own paths as needed):

```
/projects/.../nnUNet/nnUNet_raw/          # training datasets (Dataset{ID}_*)
/projects/.../nnUNet/nnUNet_preprocessed/ # nnU-Net preprocessed outputs
/projects/.../nnUNet/nnUNet_results/      # trained models + logs
/projects/.../nnUNet/test_data/           # test/eval inputs (Dataset{ID}_*)
/projects/.../nnUNet/predictions/         # prediction outputs mirroring test_data names
/projects/.../scripts/nnunet_slurm_templates/             # the 3 SLURM templates (*.sl)
```

---

## Configure hardcoded paths (important)

Open `SegBioModel.py` and adjust these constants to your cluster paths:

```python
# SLURM templates
TRAIN_TPL   = Path("/projects/.../scripts/nnunet/nnunet_train_template.sl")
VERIFY_TPL  = Path("/projects/.../scripts/nnunet/nnunet_verify_template.sl")
PREDICT_TPL = Path("/projects/.../scripts/nnunet/nnunet_predict_template.sl")

# nnU-Net v2 directories
NNRAW_DEFAULT  = Path("/projects/.../nnUNet/nnUNet_raw")
NNPRE_DEFAULT  = Path("/projects/.../nnUNet/nnUNet_preprocessed")
NNRES_DEFAULT  = Path("/projects/.../nnUNet/nnUNet_results")
```

Also confirm (and adjust if needed) the locations used by prediction:

```python
TEST_DATA_ROOT    = Path("/projects/.../nnUNet/test_data")
PREDICT_DATA_ROOT = Path("/projects/.../nnUNet/predictions")
```

Finally, `_initialize_model()` writes an init report to:

```python
scripts_dir = Path("/projects/.../scripts")
# produces SegBioModel_init_Dataset{ID}.txt here
```

Change that path if you want the init TXT elsewhere.

---

## Expected dataset layout & naming

- **Training data** must live under `nnUNet_raw/` in folders named like:
  ```
  Dataset001_foo/
  Dataset009_bar/
  ```
  The code searches by pattern `Dataset{dataset_id}_*` (e.g. `Dataset001_*`).

- **Test/eval data** must live under `test_data/` with the same `Dataset{ID}_*` naming. The prediction output folder will mirror the input folder name inside `predictions/`.

---

## What the SLURM templates expect

The code submits `sbatch` with exported variables:

- **Verify (preprocess check)**: `DATASET_ID`, `MODALITY`, `TARGET`
- **Train**: `DATASET_ID`, `CONFIG` (`"2d"` or `"3d"`), `MODALITY`, `TARGET`
- **Predict**: `DATASET_ID`, `CONFIG`, `IN_DIR`, `OUT_DIR`, `MODALITY`, `TARGET`

Make sure your `*.sl` scripts read these environment variables and run the right nnU-Net commands. For example, a training template typically runs something like:

```bash
# inside nnunet_train_template.sl (example sketch)
echo "DATASET_ID=$DATASET_ID CONFIG=$CONFIG"
srun nnUNetv2_train \
  $DATASET_ID $CONFIG \
  -p nnUNetPlans \
  -f all
```

Likewise, a predict template should use `IN_DIR`, `OUT_DIR`, and `CONFIG` with `nnUNetv2_predict`.

> The sample templates in this repo are placeholders. Adjust them to your cluster (partition, time, mail, module loads, conda env, and the exact `nnUNetv2_*` commands you use).

---

## Run the backend API

### 1) Install Python deps (login or head node)

```bash
pip install fastapi uvicorn pydantic
# (and anything else you need; nnU-Net itself must be available on compute nodes)
```

### 2) Start the API server (from repo root)

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

---

## API overview

### Schemas

- `ModelSpec`:
  - `modality` (e.g., `"EM"`)
  - `target` (e.g., `"mitochondria"`)
  - `dataset_id` (e.g., `"001"`)
  - `config` (optional, default `"3d"`, can be `"2d"`)

- `InferenceSpec` extends `ModelSpec` with:
  - `test_data` — the **dataset id string** for test data (e.g., `"001"`). The code will look for `TEST_DATA_ROOT/Dataset001_*`.

### Endpoints

- `POST /train` — **non-blocking** training pipeline:
  1) `_prepare_training_data` (submits verify job unless preprocessed outputs already exist)
  2) `_initialize_model` (writes `SegBioModel_init_Dataset{ID}.txt`)
  3) `_train_with_monitoring` (submits training unless checkpoints already exist)
  4) `_validate_model` (parses logs later; also available as a separate `GET /validate/{dataset_id}`)

  **Response** returns submission status and (if submitted) SLURM job id.

- `POST /inference` — **non-blocking** prediction job for `test_data` id.
  Skips if outputs already exist in `predictions/Dataset{ID}_*`.

- `POST /prepare` — submit only the verify/preprocess check.

- `POST /initialize` — write the init TXT without submitting jobs.

- `POST /train_submit` — submit only the training job (honors dependency on verify if present).

- `GET /validate/{dataset_id}` — parse newest training logs under `nnUNet_results/Dataset{ID}_*` and return `mean_dice`. Also writes `SegBioModel_validate_Dataset{ID}.txt` to `nnUNet_results/`.

- `GET /` — service info.

---

## Example calls

> Replace values to match your data IDs and cluster paths. These calls return quickly because job execution happens on SLURM.

### Train

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
        "modality": "EM",
        "target": "mitochondria",
        "dataset_id": "001",
        "config": "3d"
      }'
```

**What happens:**  
- Verifies/preprocess-checks `nnUNet_raw/Dataset001_*` → looks for signals in `nnUNet_preprocessed/`.  
- Writes `SegBioModel_init_Dataset001.txt`.  
- Submits training to SLURM unless checkpoints already exist in `nnUNet_results/Dataset001_*`.

### Validate (get mean Dice later)

```bash
curl http://localhost:8000/validate/001
```

Returns something like:

```json
{
  "dataset_id": "001",
  "mean_dice": 0.73,
  "validate_txt": "/projects/.../nnUNet/nnUNet_results/SegBioModel_validate_Dataset001.txt"
}
```

### Predict

First, ensure a test dataset exists as `test_data/Dataset001_something/` (the name must match `Dataset{test_data}_*`).

```bash
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{
        "modality": "EM",
        "target": "mitochondria",
        "dataset_id": "001",
        "config": "3d",
        "test_data": "001"
      }'
```

The job writes results to `predictions/Dataset001_something/`. If files (`*.nii.gz`, `*.npz`, or `*.tif`) already exist there, the API will **skip** re-submitting.

---

## Skipping rules (idempotence)

- **Prepare** skips if it finds any of:
  - `nnUNet_preprocessed/Dataset{ID}_*/{plans.json,dataset.json,dataset_fingerprint.json,*.npz}`

- **Train** skips if it finds any `checkpoint_*.pth` under:
  - `nnUNet_results/Dataset{ID}_*/(3d|3d_fullres|3d_lowres|2d)/fold_*` (or `fold_all`)

- **Predict** skips if output dir already contains any of:
  - `*.nii.gz`, `*.npz`, or `*.tif`

---

## Logs & small artifacts

- Init report: `SegBioModel_init_Dataset{ID}.txt` (path configurable in `_initialize_model()`).
- Validation summary: `SegBioModel_validate_Dataset{ID}.txt` under `nnUNet_results/`.
- SLURM stdout/err from the templates (filenames controlled by `#SBATCH --output/--error`).

---

## Quick start checklist

1. Edit the **paths** in `SegBioModel.py` to your cluster.
2. Put your SLURM templates under `scripts/nnunet/` and make sure they use the exported env vars.
3. Place training data under `nnUNet_raw/Dataset{ID}_*` and test data under `test_data/Dataset{ID}_*`.
4. Run the API: `uvicorn api:app --host 0.0.0.0 --port 8000`.
5. `POST /train`, then `GET /validate/{ID}`, and `POST /inference` when ready.

---
