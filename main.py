# pyright: reportMissingImports=false
import numpy as np
import pandas as pd
import math
import json
import pickle
import os

from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, Tuple, Optional

from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    Form,
    Request,
    Query,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from pydantic import BaseModel

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import LabelEncoder

from train_models import train_models
from auth_utils import require_admin

# ------------------------------------------------------
# ENV + PATH SETUP
# ------------------------------------------------------

load_dotenv()
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",")]

BASE_DIR = Path(__file__).parent

# NEW: directories for multi-tenant artifacts
MODELS_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "metrics"
DATASETS_DIR = BASE_DIR / "datasets"

for d in (MODELS_DIR, METRICS_DIR, DATASETS_DIR):
    d.mkdir(exist_ok=True)

DEFAULT_PREFIX = "default"
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")

ROWS_PER_PAGE = 50

# ------------------------------------------------------
# Default dataset (Breast Cancer)
# ------------------------------------------------------

data = load_breast_cancer()
X_DEFAULT = data.data
y_DEFAULT = data.target
feature_names = list(data.feature_names)

DEFAULT_DATASET_NAME = "Breast Cancer (Default)"

# ------------------------------------------------------
# FastAPI app setup
# ------------------------------------------------------

app = FastAPI(title="ML Model Comparison API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------
# Simple in-memory caches
# ------------------------------------------------------

DEFAULT_MODELS: Dict[str, Any] = {}
DEFAULT_METRICS: Dict[str, Any] = {}
DEFAULT_BEST_MODEL_NAME: Optional[str] = None

USER_MODELS_CACHE: Dict[str, Dict[str, Any]] = {}  # user_id -> models dict
USER_METRICS_CACHE: Dict[str, Dict[str, Any]] = {}  # user_id -> metrics json
USER_DF_CACHE: Dict[str, pd.DataFrame] = {}  # user_id or "default" -> DataFrame


# ------------------------------------------------------
# Helper functions
# ------------------------------------------------------


def build_prefix(user_id: Optional[str]) -> str:
    """Return the storage prefix for a given owner."""
    return user_id if user_id else DEFAULT_PREFIX


def get_paths(prefix: str) -> Tuple[Path, Path, Path, Path]:
    """
    For a given prefix (default or user_id), return:
      best_model_path, all_models_path, metrics_path, dataset_path
    """
    best_model_path = MODELS_DIR / f"{prefix}_best_model.pkl"
    all_models_path = MODELS_DIR / f"{prefix}_models.pkl"
    metrics_path = METRICS_DIR / f"{prefix}_metrics.json"
    dataset_path = DATASETS_DIR / f"{prefix}_dataset.csv"
    return best_model_path, all_models_path, metrics_path, dataset_path


def effective_prefix(user_id: Optional[str]) -> str:
    """
    Decide which prefix to use:
    - if user_id is given and that user has a metrics file => their dataset
    - otherwise => default dataset
    """
    if user_id:
        _, _, user_metrics_path, _ = get_paths(build_prefix(user_id))
        if user_metrics_path.exists():
            return build_prefix(user_id)

    # fallback to default
    return DEFAULT_PREFIX


def require_admin(request: Request):
    """Simple header-based admin protection using ADMIN_API_KEY."""
    if not ADMIN_API_KEY:
        raise HTTPException(
            500,
            "ADMIN_API_KEY not set in backend .env (admin endpoints disabled).",
        )

    header_val = request.headers.get("x-admin-key") or request.headers.get(
        "X-Admin-Key"
    )
    if header_val != ADMIN_API_KEY:
        raise HTTPException(403, "Admin access required.")


def train_and_persist(
    df: pd.DataFrame,
    dataset_name: str,
    owner_id: Optional[str],
) -> Dict[str, Any]:
    """
    Common function to train models on a DataFrame and save:
    - best_model.pkl
    - all_models.pkl
    - metrics.json
    - dataset.csv
    For either default (owner_id=None) or a specific user.
    """
    if df.shape[1] < 2:
        raise HTTPException(
            400, "Dataset must contain at least 1 feature + 1 label column"
        )

    # Split features/label
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    MODELS, MODEL_METRICS, best_name, best_confusion = train_models(X, y)

    prefix = build_prefix(owner_id)
    best_model_path, all_models_path, metrics_path, dataset_path = get_paths(prefix)

    # Save best model only
    with open(best_model_path, "wb") as f:
        pickle.dump(MODELS[best_name], f)

    # Save all models (so we can predict with ANY model later)
    with open(all_models_path, "wb") as f:
        pickle.dump(MODELS, f)

    # Save dataset to CSV (cleaned version)
    df.to_csv(dataset_path, index=False)

    # Save metrics JSON
    metrics_payload = {
        "best_model": best_name,
        "metrics": MODEL_METRICS,
        "dataset_name": dataset_name,
        "rows": int(df.shape[0]),
        "columns": list(df.columns),
        "owner_id": owner_id,
        "dataset_path": str(dataset_path),
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=4)

    # Update caches
    if owner_id is None:
        global DEFAULT_MODELS, DEFAULT_METRICS, DEFAULT_BEST_MODEL_NAME
        DEFAULT_MODELS = MODELS
        DEFAULT_METRICS = metrics_payload
        DEFAULT_BEST_MODEL_NAME = best_name
        USER_DF_CACHE[DEFAULT_PREFIX] = df
    else:
        USER_MODELS_CACHE[owner_id] = MODELS
        USER_METRICS_CACHE[owner_id] = metrics_payload
        USER_DF_CACHE[owner_id] = df

    return metrics_payload


def load_default_if_needed():
    """Ensure default dataset models & metrics exist; train them from sklearn data if not."""
    global DEFAULT_MODELS, DEFAULT_METRICS, DEFAULT_BEST_MODEL_NAME

    prefix = DEFAULT_PREFIX
    best_model_path, all_models_path, metrics_path, dataset_path = get_paths(prefix)

    # If metrics already exist, load them (and models)
    if metrics_path.exists() and all_models_path.exists():
        with open(metrics_path, "r") as f:
            DEFAULT_METRICS = json.load(f)

        DEFAULT_BEST_MODEL_NAME = DEFAULT_METRICS["best_model"]

        with open(all_models_path, "rb") as f:
            DEFAULT_MODELS = pickle.load(f)

        # load dataset if present, else reconstruct from X_DEFAULT
        if dataset_path.exists():
            df = pd.read_csv(dataset_path)
        else:
            df = pd.DataFrame(data=X_DEFAULT, columns=feature_names)
            df["target"] = y_DEFAULT

        USER_DF_CACHE[DEFAULT_PREFIX] = df
        print("✅ Loaded default dataset & models from disk.")
        return

    # Otherwise train from sklearn dataset and persist:
    df_default = pd.DataFrame(data=X_DEFAULT, columns=feature_names)
    df_default["target"] = y_DEFAULT

    print("⚠️ No default artifacts found. Training default dataset...")
    DEFAULT_METRICS = train_and_persist(
        df=df_default,
        dataset_name=DEFAULT_DATASET_NAME,
        owner_id=None,
    )
    print("✅ Default dataset trained & saved.")


def load_models_for_prefix(prefix: str) -> Dict[str, Any]:
    """Load all models dict for a given prefix from cache or disk."""
    global DEFAULT_MODELS  # ✅ MOVED TO TOP

    # default
    if prefix == DEFAULT_PREFIX:
        if DEFAULT_MODELS:
            return DEFAULT_MODELS

    # user
    owner_id = None if prefix == DEFAULT_PREFIX else prefix
    if owner_id and owner_id in USER_MODELS_CACHE:
        return USER_MODELS_CACHE[owner_id]

    _, all_models_path, metrics_path, _ = get_paths(prefix)
    if not all_models_path.exists() or not metrics_path.exists():
        if prefix != DEFAULT_PREFIX:
            # Fallback to default
            return load_models_for_prefix(DEFAULT_PREFIX)
        raise HTTPException(500, "Model artifacts not found.")

    with open(all_models_path, "rb") as f:
        models = pickle.load(f)

    if prefix == DEFAULT_PREFIX:
        DEFAULT_MODELS = models
    else:
        USER_MODELS_CACHE[owner_id] = models  # type: ignore

    return models



def load_metrics_for_prefix(prefix: str) -> Dict[str, Any]:
    """Load metrics json for given prefix from cache or disk."""
    global DEFAULT_METRICS, DEFAULT_BEST_MODEL_NAME  # ✅ MOVED TO TOP

    if prefix == DEFAULT_PREFIX and DEFAULT_METRICS:
        return DEFAULT_METRICS

    owner_id = None if prefix == DEFAULT_PREFIX else prefix
    if owner_id and owner_id in USER_METRICS_CACHE:
        return USER_METRICS_CACHE[owner_id]

    _, _, metrics_path, _ = get_paths(prefix)
    if not metrics_path.exists():
        if prefix != DEFAULT_PREFIX:
            return load_metrics_for_prefix(DEFAULT_PREFIX)
        raise HTTPException(500, "Metrics not found for default dataset.")

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    if prefix == DEFAULT_PREFIX:
        DEFAULT_METRICS = metrics
        DEFAULT_BEST_MODEL_NAME = metrics["best_model"]
    else:
        USER_METRICS_CACHE[owner_id] = metrics  # type: ignore

    return metrics



def load_df_for_prefix(prefix: str) -> pd.DataFrame:
    """Load dataset DataFrame from cache or CSV, or fallback to default."""
    if prefix in USER_DF_CACHE:
        return USER_DF_CACHE[prefix]

    _, _, metrics_path, dataset_path = get_paths(prefix)

    if dataset_path.exists():
        df = pd.read_csv(dataset_path)
        USER_DF_CACHE[prefix] = df
        return df

    # If dataset not found but metrics exist, something is off; fallback to default
    if prefix != DEFAULT_PREFIX:
        return load_df_for_prefix(DEFAULT_PREFIX)

    # default: reconstruct from sklearn if CSV missing
    df_default = pd.DataFrame(data=X_DEFAULT, columns=feature_names)
    df_default["target"] = y_DEFAULT
    USER_DF_CACHE[DEFAULT_PREFIX] = df_default
    return df_default


# ------------------------------------------------------
# Pydantic Models
# ------------------------------------------------------


class PredictRequest(BaseModel):
    model_name: str
    features: list[float]
    user_id: Optional[str] = None  # NEW: whose dataset to use


class PredictResponse(BaseModel):
    model_name: str
    predicted_class: int
    predicted_proba: float | None


class ModelInfo(BaseModel):
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    confusion_matrix: dict | None = None


# ------------------------------------------------------
# LOAD OR TRAIN DEFAULT ON STARTUP
# ------------------------------------------------------


@app.on_event("startup")
def on_startup():
    print("\n🚀 Server starting...")
    load_default_if_needed()
    print("✅ Ready\n")


# ------------------------------------------------------
# ROUTES
# ------------------------------------------------------


@app.get("/")
def root():
    return {"status": "ML Model API running"}


@app.get("/features")
def get_features():
    return {"feature_names": feature_names}


@app.get("/models", response_model=list[ModelInfo])
def get_models(userId: Optional[str] = Query(None)):
    """
    Return model metrics for either:
    - user dataset (if userId has its own metrics)
    - otherwise default dataset
    """
    prefix = effective_prefix(userId)
    metrics_payload = load_metrics_for_prefix(prefix)
    metrics_dict = metrics_payload["metrics"]

    return [
        {
            "name": name,
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "roc_auc": m.get("roc_auc"),
            "confusion_matrix": m.get("confusion_matrix"),
        }
        for name, m in metrics_dict.items()
    ]


@app.get("/confusion-matrix/{model_name}")
def get_confusion_matrix(model_name: str, userId: Optional[str] = Query(None)):
    prefix = effective_prefix(userId)
    metrics_payload = load_metrics_for_prefix(prefix)
    metrics_dict = metrics_payload["metrics"]

    if model_name not in metrics_dict:
        raise HTTPException(404, detail="Model not found")

    cm = metrics_dict[model_name].get("confusion_matrix")
    if not cm:
        raise HTTPException(404, detail="Confusion matrix not found")

    return {"model_name": model_name, "confusion_matrix": cm}


@app.get("/best-model")
def best_model(userId: Optional[str] = Query(None)):
    """
    Return best model info for user or default dataset.
    """
    prefix = effective_prefix(userId)
    metrics_payload = load_metrics_for_prefix(prefix)
    best_name = metrics_payload["best_model"]
    metrics = metrics_payload["metrics"][best_name]

    return {
        "best_model_name": best_name,
        "metrics": metrics,
        "confusion_matrix": metrics.get("confusion_matrix"),
    }


@app.get("/download-best-model")
def download_best_model(userId: Optional[str] = Query(None)):
    prefix = effective_prefix(userId)
    best_model_path, _, _, _ = get_paths(prefix)

    if not best_model_path.exists():
        raise HTTPException(404, "Model file not found")

    filename = (
        "default_best_model.pkl" if prefix == DEFAULT_PREFIX else f"{prefix}_best_model.pkl"
    )

    return FileResponse(
        path=str(best_model_path),
        filename=filename,
        media_type="application/octet-stream",
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    prefix = effective_prefix(req.user_id)
    models = load_models_for_prefix(prefix)

    if req.model_name not in models:
        raise HTTPException(404, "Model not found for this dataset")

    if len(req.features) != len(feature_names):
        raise HTTPException(400, f"Expected {len(feature_names)} features")

    model = models[req.model_name]
    x = np.array(req.features).reshape(1, -1)

    pred = model.predict(x)[0]
    proba = None

    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(x)[0][1])

    return {
        "model_name": req.model_name,
        "predicted_class": int(pred),
        "predicted_proba": proba,
    }


@app.post("/train-from-file")
async def train_from_file(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    user_email: Optional[str] = Form(None),
):
    """
    Train a *personal* dataset for a logged-in user.
    Frontend must send:
      - file (CSV)
      - user_id (string, Mongo _id)
      - user_email (optional)

    This does NOT change the default dataset.
    """
    # Load raw CSV
    df = pd.read_csv(file.file)

    if df.shape[1] < 2:
        raise HTTPException(
            400, "Dataset must contain at least 1 feature + 1 label column"
        )

    # Clean data
    df.dropna(how="all", inplace=True)

    encoders = {}
    for col in df.columns:
        if df[col].dtype == "object":
            encoder = LabelEncoder()
            df[col] = df[col].astype(str)
            df[col] = encoder.fit_transform(df[col])
            encoders[col] = encoder

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.fillna(df.mean(numeric_only=True), inplace=True)

    if df.isnull().sum().sum() > 0:
        df = df.dropna()

    if df.shape[0] < 10:
        raise HTTPException(400, "Dataset too small after cleaning")

    # Train and persist for this user
    dataset_name = file.filename or f"user_dataset_{user_id}"
    metrics_payload = train_and_persist(df=df, dataset_name=dataset_name, owner_id=user_id)

    # attach email in metrics cache/json for admin viewing
    owner_metrics = USER_METRICS_CACHE.get(user_id, metrics_payload)
    owner_metrics["user_email"] = user_email
    USER_METRICS_CACHE[user_id] = owner_metrics

    # Also patch metrics file with email (optional)
    _, _, metrics_path, _ = get_paths(build_prefix(user_id))
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            existing = json.load(f)
        existing["user_email"] = user_email
        with open(metrics_path, "w") as f:
            json.dump(existing, f, indent=4)

    return {
        "message": "✅ Training completed safely on your dataset",
        "best_model": metrics_payload["best_model"],
        "dataset_name": metrics_payload["dataset_name"],
        "rows_used": metrics_payload["rows"],
        "columns": metrics_payload["columns"],
    }


@app.get("/current-dataset")
def get_current_dataset(userId: Optional[str] = Query(None)):
    """
    Returns the dataset name for this user, or default name.
    """
    prefix = effective_prefix(userId)
    metrics_payload = load_metrics_for_prefix(prefix)
    return {"name": metrics_payload.get("dataset_name", DEFAULT_DATASET_NAME)}


@app.get("/dataset-preview")
def get_dataset_preview(page: int = 1, userId: Optional[str] = Query(None)):
    prefix = effective_prefix(userId)
    df = load_df_for_prefix(prefix)

    total_rows = len(df)
    total_pages = max(1, math.ceil(total_rows / ROWS_PER_PAGE))

    if page < 1 or page > total_pages:
        raise HTTPException(400, "Invalid page number")

    start = (page - 1) * ROWS_PER_PAGE
    end = start + ROWS_PER_PAGE

    page_data = df.iloc[start:end].to_dict(orient="records")

    metrics_payload = load_metrics_for_prefix(prefix)

    return {
        "dataset": metrics_payload.get("dataset_name", DEFAULT_DATASET_NAME),
        "page": page,
        "per_page": ROWS_PER_PAGE,
        "total_rows": total_rows,
        "total_pages": total_pages,
        "data": page_data,
    }


# ------------------------------------------------------
# ADMIN ENDPOINTS
# ------------------------------------------------------


@app.get("/admin/datasets")
def get_all_datasets(request: Request):
    require_admin(request)
    """
    Admin-only: list all datasets (default + user-specific).
    Protected by X-Admin-Key header.
    """
    require_admin(request)

    datasets = []

    # Include default
    default_metrics = load_metrics_for_prefix(DEFAULT_PREFIX)
    datasets.append(
        {
            "owner_id": None,
            "prefix": DEFAULT_PREFIX,
            "type": "default",
            "dataset_name": default_metrics.get("dataset_name", DEFAULT_DATASET_NAME),
            "rows": default_metrics.get("rows"),
            "columns": default_metrics.get("columns"),
        }
    )

    # Include all user metrics json files
    for metrics_file in METRICS_DIR.glob("*_metrics.json"):
        name = metrics_file.name  # e.g. "abc123_metrics.json"
        prefix = name.replace("_metrics.json", "")
        if prefix == DEFAULT_PREFIX:
            continue

        with open(metrics_file, "r") as f:
            m = json.load(f)

        datasets.append(
            {
                "owner_id": m.get("owner_id") or prefix,
                "prefix": prefix,
                "type": "user",
                "dataset_name": m.get("dataset_name"),
                "rows": m.get("rows"),
                "columns": m.get("columns"),
                "user_email": m.get("user_email"),
            }
        )

    return {"datasets": datasets}


@app.delete("/admin/datasets/{owner_id}")
def admin_delete_user_dataset(owner_id: str, request: Request):
    require_admin(request)
    """
    Admin-only: deletes a user's dataset + models + metrics.
    After this, that user will fall back to the default dataset.
    """
    require_admin(request)

    prefix = build_prefix(owner_id)
    best_model_path, all_models_path, metrics_path, dataset_path = get_paths(prefix)

    # Delete files if exist
    for p in (best_model_path, all_models_path, metrics_path, dataset_path):
        if p.exists():
            p.unlink()

    # Clear caches
    USER_MODELS_CACHE.pop(owner_id, None)
    USER_METRICS_CACHE.pop(owner_id, None)
    USER_DF_CACHE.pop(owner_id, None)

    return {"message": f"✅ Dataset for user {owner_id} removed. Fallback to default."}


@app.post("/admin/default-dataset")
async def admin_set_default_dataset(
    request: Request,
    file: UploadFile = File(...),
):
    require_admin(request)
    """
    Admin-only: upload a NEW default dataset.
    This replaces the previous default dataset for ALL users who do not have
    their own personal dataset.
    """
    require_admin(request)

    df = pd.read_csv(file.file)

    if df.shape[1] < 2:
        raise HTTPException(
            400, "Dataset must contain at least 1 feature + 1 label column"
        )

    # Clean data
    df.dropna(how="all", inplace=True)

    encoders = {}
    for col in df.columns:
        if df[col].dtype == "object":
            encoder = LabelEncoder()
            df[col] = df[col].astype(str)
            df[col] = encoder.fit_transform(df[col])
            encoders[col] = encoder

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.fillna(df.mean(numeric_only=True), inplace=True)

    if df.isnull().sum().sum() > 0:
        df = df.dropna()

    if df.shape[0] < 10:
        raise HTTPException(400, "Dataset too small after cleaning")

    dataset_name = file.filename or "Admin Default Dataset"
    metrics_payload = train_and_persist(df=df, dataset_name=dataset_name, owner_id=None)

    return {
        "message": "✅ Default dataset updated successfully",
        "best_model": metrics_payload["best_model"],
        "dataset_name": metrics_payload["dataset_name"],
        "rows_used": metrics_payload["rows"],
        "columns": metrics_payload["columns"],
    }
