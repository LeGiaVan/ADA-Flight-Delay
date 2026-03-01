import os
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ======================================================
# PATH CONFIG
# ======================================================
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BACKEND_DIR)

FRONTEND_HTML = os.path.join(ROOT_DIR, "frontend", "uxui.html")
MODEL_DIR = os.path.join(BACKEND_DIR, "models")
TE_DIR = os.path.join(BACKEND_DIR, "te_maps")

# ======================================================
# LOAD STATIC MODEL ONLY
# ======================================================
model_static = xgb.Booster()
model_static.load_model(
    os.path.join(MODEL_DIR, "xgboost_model_1_static_ver2.json")
)

print("✅ Static model loaded")

# ======================================================
# LOAD TE MAPS (STATIC)
# ======================================================
def load_te(col):
    path = os.path.join(
        TE_DIR, f"TE_mapping_{col}_xgboost_model_1_static_ver2.csv"
    )
    df = pd.read_csv(path)
    te_col = f"TE_{col}"
    mapping = dict(zip(df[col], df[te_col]))
    mapping["_GLOBAL_"] = df[te_col].mean()
    return mapping


TE_MAPS = {
    "OP_CARRIER": load_te("OP_CARRIER"),
    "ORIGIN": load_te("ORIGIN"),
    "DEST": load_te("DEST"),
}

print("✅ TE maps loaded")

# ======================================================
# FASTAPI
# ======================================================
app = FastAPI()

# ======================================================
# INPUT SCHEMA – ĐÚNG UI
# ======================================================
class PredictInput(BaseModel):
    MONTH: int
    DAY_OF_WEEK: int
    HOUR: int
    TIME_OF_DAY: int
    ROUTE_COUNT: int
    CRS_ELAPSED_TIME: int
    OP_CARRIER: str
    ORIGIN: str
    DEST: str

# ======================================================
# FEATURE ORDER – ĐÚNG LÚC TRAIN
# ======================================================
FEATURES = [
    "MONTH",
    "DAY_OF_WEEK",
    "HOUR",
    "TIME_OF_DAY",
    "ROUTE_COUNT",
    "CRS_ELAPSED_TIME",
    "TE_OP_CARRIER",
    "TE_ORIGIN",
    "TE_DEST",
]

def build_feature_df(data: dict):
    row = {
        "MONTH": data["MONTH"],
        "DAY_OF_WEEK": data["DAY_OF_WEEK"],
        "HOUR": data["HOUR"],
        "TIME_OF_DAY": data["TIME_OF_DAY"],
        "ROUTE_COUNT": data["ROUTE_COUNT"],
        "CRS_ELAPSED_TIME": data["CRS_ELAPSED_TIME"],
    }

    for col in ["OP_CARRIER", "ORIGIN", "DEST"]:
        row[f"TE_{col}"] = TE_MAPS[col].get(
            data[col], TE_MAPS[col]["_GLOBAL_"]
        )

    df = pd.DataFrame([[row[f] for f in FEATURES]], columns=FEATURES)
    return df.astype("float32")

# ======================================================
# ROUTES
# ======================================================
@app.get("/")
def ui():
    return FileResponse(FRONTEND_HTML)

@app.post("/predict/static")
def predict_static(data: PredictInput):
    X = build_feature_df(data.dict())
    dmat = xgb.DMatrix(X)

    print("\n====== DEBUG STATIC ======")
    print(X)
    print("==========================\n")

    prob = model_static.predict(dmat)[0]

    return {
        "class": int(prob.argmax()),
        "prob": prob.tolist(),
    }