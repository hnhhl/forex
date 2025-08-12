from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path so we can import the 'src' package under reloader subprocesses
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure src package import works
from src.core.ultimate_xau_system import UltimateXAUSystem, SystemConfig

logger = logging.getLogger(__name__)

app = FastAPI(title="Ultimate XAU Super System API", version="1.0.0")

# CORS for Live Server/Live Preview
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class OHLCPoint(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None
    timestamp: Optional[str] = None

class SignalRequest(BaseModel):
    ohlc: Optional[List[OHLCPoint]] = None
    data: Optional[Dict[str, Any]] = None

# Singleton system instance
_system: Optional[UltimateXAUSystem] = None

def get_system() -> UltimateXAUSystem:
    global _system
    if _system is None:
        logger.info("Initializing UltimateXAUSystem singleton...")
        cfg = SystemConfig()
        # Safe defaults; real trading stays disabled unless configured otherwise
        cfg.live_trading = False
        cfg.paper_trading = True
        _system = UltimateXAUSystem(cfg)
    return _system

@app.get("/")
def root():
    return {"service": "ultimate-xau-system-api", "status": "ok"}

@app.get("/health")
def health():
    sysobj = get_system()
    return {
        "status": "ok",
        "ensemble_loaded": sysobj.ensemble_loaded,
        "models_count": getattr(sysobj, "models_count", 0),
    }

@app.get("/system/status")
def system_status():
    sysobj = get_system()
    return sysobj.get_system_status()

@app.post("/signal")
def generate_signal(payload: SignalRequest = Body(default=None)):
    sysobj = get_system()
    data_dict: Optional[Dict[str, Any]] = None
    if payload is not None:
        data_dict = payload.model_dump()
    return sysobj.generate_signal(data=data_dict)

@app.get("/training/status")
def training_status():
    sysobj = get_system()
    return sysobj.get_training_status()

@app.post("/training/start")
def training_start():
    sysobj = get_system()
    return sysobj.start_training()

@app.get("/ensemble/status")
def ensemble_status():
    sysobj = get_system()
    if not sysobj.ensemble_manager:
        return {"status": "no_ensemble"}
    return sysobj.ensemble_manager.get_parliament_status()

@app.get("/ensemble/top")
def ensemble_top(limit: int = 5):
    sysobj = get_system()
    if not sysobj.ensemble_manager:
        return {"status": "no_ensemble"}
    return {"top": sysobj.ensemble_manager.get_top_performers(limit)}

@app.get("/models/list")
def list_models():
    sysobj = get_system()
    if not sysobj.ensemble_manager:
        return {"registered": [], "active": []}
    registered = list(sysobj.ensemble_manager.model_registry.keys())
    active = list(sysobj.ensemble_manager.loaded_models.keys())
    return {"registered": registered, "active": active} 