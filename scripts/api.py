# api.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

# IMPORTANT: this assumes SegBioModel.py is in the same folder
from SegBioModel import SegBioModel, NNRES_DEFAULT

app = FastAPI(title="SegBioModel API", version="1.0")

# --------- Request schemas ---------
class ModelSpec(BaseModel):
    modality: str = Field(..., example="EM")
    target: str = Field(..., example="mitochondria")
    dataset_id: str = Field(..., example="001")
    config: Optional[str] = Field("3d", example="2d")

class InferenceSpec(ModelSpec):
    test_data: str = Field(..., example="001")

# --------- Helpers ---------
def _mk_model(spec: ModelSpec) -> SegBioModel:
    return SegBioModel(
        modality=spec.modality,
        target=spec.target,
        dataset_id=spec.dataset_id,
        config=spec.config,
    )

# --------- Public high-level endpoints ---------
@app.post("/train")
def train(spec: ModelSpec) -> Dict[str, Any]:
    """
    Non-blocking pipeline:
      1) _prepare_training_data (skip if done)
      2) _initialize_model (writes init txt)
      3) _train_with_monitoring (skip if done)
      4) _validate_model (writes validation txt, returns mean_dice or None)
    """
    m = _mk_model(spec)
    return m.train()

@app.post("/inference")
def inference(spec: InferenceSpec) -> Dict[str, Any]:
    """
    Submit prediction job (skips if outputs already exist).
    """
    m = _mk_model(spec)
    return m.inference(spec.test_data)

# --------- Expose all individual methods too ---------
@app.post("/prepare")
def prepare(spec: ModelSpec) -> Dict[str, Any]:
    m = _mk_model(spec)
    return m._prepare_training_data()

@app.post("/initialize")
def initialize(spec: ModelSpec) -> Dict[str, Any]:
    m = _mk_model(spec)
    m._initialize_model()
    # return where the init .txt is written
    return {
        "status": "ok",
        "init_txt": str(NNRES_DEFAULT / f"SegBioModel_init_Dataset{spec.dataset_id}.txt"),
    }

@app.post("/train_submit")
def train_submit(spec: ModelSpec) -> Dict[str, Any]:
    m = _mk_model(spec)
    return m._train_with_monitoring()

@app.get("/validate/{dataset_id}")
def validate(dataset_id: str, modality: str = "EM", target: str = "mitochondria", config: str = "3d") -> Dict[str, Any]:
    """
    Parse logs and write SegBioModel_validate_Dataset{ID}.txt.
    """
    m = SegBioModel(modality=modality, target=target, dataset_id=dataset_id, config=config)
    mean = m._validate_model()
    return {
        "dataset_id": dataset_id,
        "mean_dice": mean,
        "validate_txt": str(NNRES_DEFAULT / f"SegBioModel_validate_Dataset{dataset_id}.txt"),
    }

# --------- Utility ---------
@app.get("/")
def root():
    return {"service": "SegBioModel API", "version": "1.0"}
