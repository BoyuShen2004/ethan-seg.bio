from pathlib import Path
import subprocess
import os
import re, json

# Reusable SLURM templates
TRAIN_TPL   = Path("/projects/weilab/seg.bio/ethan/scripts/nnunet/nnunet_train_template.sl")
VERIFY_TPL  = Path("/projects/weilab/seg.bio/ethan/scripts/nnunet/nnunet_verify_template.sl")
PREDICT_TPL = Path("/projects/weilab/seg.bio/ethan/scripts/nnunet/nnunet_predict_template.sl")

# Default nnU-Net v2 directories
NNRAW_DEFAULT  = Path("/projects/weilab/seg.bio/ethan/nnUNet/nnUNet_raw")
NNPRE_DEFAULT  = Path("/projects/weilab/seg.bio/ethan/nnUNet/nnUNet_preprocessed")
NNRES_DEFAULT  = Path("/projects/weilab/seg.bio/ethan/nnUNet/nnUNet_results")


class SegBioModel:
    def __init__(self, modality, target, dataset_id: str, config: str | None = "3d"):
        """
        dataset_id: numeric string, e.g., "001".
        config: "2d" or "3d" (default "3d"). Controls which model config is returned.
        data_path will automatically resolve to nnUNet_raw/Dataset{ID}_*
        """
        self.modality = modality
        self.target = target
        self.dataset_id = dataset_id
        self.config = (config or "3d").lower()

        # Resolve data_path dynamically from nnUNet_raw
        pattern = f"Dataset{dataset_id}_*"
        matches = list(NNRAW_DEFAULT.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"No dataset folder found under {NNRAW_DEFAULT} with pattern {pattern}"
            )
        self.data_path = matches[0]

        self.model_config = self._get_model_config()

    def _get_model_config(self):
        # 3D case
        if (
            self.modality == "EM"
            and self.target == "mitochondria"
            and self.config == "3d"
        ):
            return {
                "architecture": "nnUNetV2",
                "batch_size": 2,
                "patch_size": [128, 128, 128],  # 3D
                "num_epochs": 1000,
                "learning_rate": 0.01,
            }
        # 2D case
        if (
            self.modality == "EM"
            and self.target == "mitochondria"
            and self.config == "2d"
        ):
            return {
                "architecture": "nnUNetV2",
                "batch_size": 2,
                "patch_size": [128, 128],  # 2D
                "num_epochs": 1000,
                "learning_rate": 0.01,
            }
        # Fallback (default to 2D if an unexpected combination is given)
        return {
            "architecture": "nnUNetV2",
            "batch_size": 2,
            "patch_size": [128, 128],
            "num_epochs": 1000,
            "learning_rate": 0.01,
        }

    def train(self):
        """
        Non-blocking:
        1) Submit verify/preprocess (skip if already preprocessed)
        2) Initialize model config (writes init TXT)
        3) Submit training with dependency after verify (skip if already trained)
        4) Parse validation now (writes validate TXT; returns mean dice or None)
        Returns: dict with stage results.
        """
        verify_res = self._prepare_training_data()   # may skip or submit
        self._initialize_model()                     # writes SegBioModel_init_Dataset{ID}.txt
        train_res  = self._train_with_monitoring()   # may skip or submit
        mean_dice  = self._validate_model()          # writes SegBioModel_validate_Dataset{ID}.txt
        return {"verify": verify_res, "train": train_res, "mean_dice": mean_dice}

    def inference(self, test_data):
        return self._run_inference(test_data)

    # ------------------- private implementations -------------------

    def _prepare_training_data(self):
        """
        Submit a SLURM job to verify dataset integrity for the given dataset_id.
        Non-blocking; returns dict. Skips if looks already preprocessed.
        Robust 'done' check: any matching folder in nnUNet_preprocessed having
        one of {plans.json, dataset.json, dataset_fingerprint.json, *.npz}.
        """
        assert VERIFY_TPL.exists(), f"Missing SLURM template: {VERIFY_TPL}"

        # already-done check
        for pp_dir in NNPRE_DEFAULT.glob(f"Dataset{self.dataset_id}_*"):
            if not pp_dir.is_dir():
                continue
            if ((pp_dir / "plans.json").exists()
                or (pp_dir / "dataset.json").exists()
                or (pp_dir / "dataset_fingerprint.json").exists()
                or any(pp_dir.glob("*.npz"))):
                return {"status": "skipped_already_preprocessed", "job_id": None, "preprocessed_dir": str(pp_dir)}

        # submit
        out = subprocess.run(
            ["sbatch", f"--export=ALL,DATASET_ID={self.dataset_id},MODALITY={self.modality},TARGET={self.target}",
             str(VERIFY_TPL)],
            check=True, capture_output=True, text=True
        ).stdout.strip()
        job_id = out.split()[-1]
        self._verify_job_id = job_id
        return {"status": "submitted", "job_id": job_id}

    def _initialize_model(self):
        """
        Decide the nnU-Net configuration to use ("2d" or "3d") from the model config,
        and write an init report TXT.
        """
        cfg = self.model_config
        self._nnunet_config = "3d" if len(cfg["patch_size"]) == 3 else "2d"
        
        # Write TXT to the scripts directory
        scripts_dir = Path("/projects/weilab/seg.bio/ethan/scripts")
        init_txt = scripts_dir / f"SegBioModel_init_Dataset{self.dataset_id}.txt"
        init_txt.write_text(
            "\n".join([
                f"dataset=Dataset{self.dataset_id}",
                f"modality={self.modality}",
                f"target={self.target}",
                f"config={self._nnunet_config}",
                f"architecture={cfg['architecture']}",
                f"batch_size={cfg['batch_size']}",
                f"patch_size={cfg['patch_size']}",
                f"num_epochs={cfg['num_epochs']}",
                f"learning_rate={cfg['learning_rate']}",
            ]) + "\n"
        )

    def _train_with_monitoring(self):
        """
        Submit the nnUNet training SLURM job with dependency on verify (if present).
        Non-blocking; returns dict. Skips if trained results are already there.
        Robust 'done' check: any checkpoint_*.pth anywhere under a trainer/plans
        directory that matches the current config (2d or 3d*).
        """
        assert TRAIN_TPL.exists(), f"Missing SLURM template: {TRAIN_TPL}"

        # Ensure CONFIG is available even if _initialize_model was not called
        if not hasattr(self, "_nnunet_config"):
            cfg = self.model_config
            self._nnunet_config = "3d" if len(cfg["patch_size"]) == 3 else "2d"

        # already-trained check
        config_tags = ["2d"] if self._nnunet_config == "2d" else ["3d", "3d_fullres", "3d_lowres"]
        for rdir in NNRES_DEFAULT.glob(f"Dataset{self.dataset_id}_*"):
            if not rdir.is_dir():
                continue
            # candidate trainer/plan dirs; fallback is rdir itself
            tdirs = [p for p in rdir.glob("*/") if any(tag in p.name for tag in config_tags)] or [rdir]
            for tdir in tdirs:
                # Any checkpoint under fold_all or fold_* is enough to mark as 'done'
                if list((tdir / "fold_all").glob("checkpoint_*.pth")):
                    return {"status": "skipped_already_trained", "job_id": None, "results_dir": str(rdir)}
                for fdir in tdir.glob("fold_*"):
                    if fdir.is_dir() and list(fdir.glob("checkpoint_*.pth")):
                        return {"status": "skipped_already_trained", "job_id": None, "results_dir": str(rdir)}
                # Extra safety: deep search (covers unusual layouts)
                if list(tdir.rglob("checkpoint_*.pth")):
                    return {"status": "skipped_already_trained", "job_id": None, "results_dir": str(rdir)}

        export_vars = ",".join([
            f"DATASET_ID={self.dataset_id}",
            f"CONFIG={self._nnunet_config}",
            f"MODALITY={self.modality}",
            f"TARGET={self.target}",
        ])

        # Build sbatch command with optional dependency
        sbatch_cmd = ["sbatch"]
        if hasattr(self, "_verify_job_id") and self._verify_job_id:
            sbatch_cmd += [f"--dependency=afterok:{self._verify_job_id}"]
        sbatch_cmd += [f"--export=ALL,{export_vars}", str(TRAIN_TPL)]

        out = subprocess.run(sbatch_cmd, check=True, capture_output=True, text=True).stdout.strip()
        job_id = out.split()[-1]
        self._train_job_id = job_id
        return {"status": "submitted", "job_id": job_id}

    def _validate_model(self):
        """
        Parse the newest training logs under NNRES_DEFAULT/Dataset{ID}_* to get
        the final mean validation Dice. Also writes:
          NNRES_DEFAULT/SegBioModel_validate_Dataset{ID}.txt
        Returns mean_dice (float) or None.
        """
        results_root = NNRES_DEFAULT
        cand_dirs = [p for p in results_root.glob(f"Dataset{self.dataset_id}_*") if p.is_dir()]
       # Write TXT to the scripts directory
        scripts_dir = Path("/projects/weilab/seg.bio/ethan/scripts") 
        val_txt = scripts_dir / f"SegBioModel_validate_Dataset{self.dataset_id}.txt"

        if not cand_dirs:
            val_txt.write_text(
                "\n".join([
                    f"dataset=Dataset{self.dataset_id}",
                    "results_dir=NA",
                    "log=NA",
                    "mean_dice=NA",
                ]) + "\n"
            )
            return None

        res_dir = max(cand_dirs, key=lambda p: p.stat().st_mtime)

        log_files = sorted(
            res_dir.rglob("training_log_*.txt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        dice_rx = re.compile(r"Mean\s+Validation\s+Dice\s*[:=]\s*([0-9]*\.?[0-9]+)")

        mean = None
        source_log = None
        for lf in log_files:
            try:
                m = dice_rx.search(lf.read_text(errors="ignore"))
                if m:
                    mean = float(m.group(1))
                    source_log = lf
                    break
            except Exception:
                continue

        val_txt.write_text(
            "\n".join([
                f"dataset=Dataset{self.dataset_id}",
                f"results_dir={res_dir}",
                f"log={source_log if source_log else 'NA'}",
                f"mean_dice={mean if mean is not None else 'NA'}",
            ]) + "\n"
        )

        return mean

    def _run_inference(self, test_data):
        """
        Submit a SLURM prediction job for a test dataset by ID.
        Non-blocking; returns a dict with job id and paths. Skips if output exists.
        Robust 'done' check: output dir contains any *.nii.gz / *.npz / *.tif.
        """
        TEST_DATA_ROOT     = Path("/projects/weilab/seg.bio/ethan/nnUNet/test_data")
        PREDICT_DATA_ROOT  = Path("/projects/weilab/seg.bio/ethan/nnUNet/predictions")

        assert isinstance(test_data, str) and test_data.strip(), "test_data must be a dataset id string like '001'"
        assert PREDICT_TPL.exists(), f"Missing SLURM template: {PREDICT_TPL}"

        # Resolve input folder: Dataset{ID}_**
        pattern = f"Dataset{test_data}_*"
        matches = sorted(TEST_DATA_ROOT.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No test-data folder under {TEST_DATA_ROOT} matching '{pattern}'")
        in_dir = matches[0]
        if not in_dir.is_dir():
            raise FileNotFoundError(f"Resolved test-data path is not a directory: {in_dir}")

        # Output folder: mirror name under predictions/
        out_dir = PREDICT_DATA_ROOT / in_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Skip if output already contains predicted files
        if any(out_dir.glob("*.nii.gz")) or any(out_dir.glob("*.npz")) or any(out_dir.glob("*.tif")):
            return {
                "status": "skipped_already_predicted",
                "job_id": None,
                "in_dir": str(in_dir),
                "out_dir": str(out_dir),
                "config": getattr(self, "_nnunet_config", self.config),
            }

        # Ensure CONFIG is available ('2d' or '3d')
        if not hasattr(self, "_nnunet_config"):
            cfg = self.model_config
            self._nnunet_config = "3d" if len(cfg["patch_size"]) == 3 else "2d"

        export_vars = ",".join([
            f"DATASET_ID={self.dataset_id}",  # training dataset id
            f"CONFIG={self._nnunet_config}",
            f"IN_DIR={in_dir}",
            f"OUT_DIR={out_dir}",
            f"MODALITY={self.modality}",
            f"TARGET={self.target}",
        ])

        out = subprocess.run(
            ["sbatch", f"--export=ALL,{export_vars}", str(PREDICT_TPL)],
            check=True, capture_output=True, text=True
        ).stdout.strip()
        job_id = out.split()[-1]

        return {
            "status": "submitted",
            "job_id": job_id,
            "in_dir": str(in_dir),
            "out_dir": str(out_dir),
            "config": self._nnunet_config,
        }
