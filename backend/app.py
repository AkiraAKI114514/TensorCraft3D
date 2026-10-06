import asyncio
import importlib.util
import queue
import threading
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict
from .graph import analyze_graph

app = FastAPI(title="TensorLab 3D", version="1.0.1")
training_lock = threading.Lock()
ROOT = Path(__file__).resolve().parent.parent


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    epochs: int = Field(30, ge=1, le=500)
    learningRate: float = Field(0.001, ge=0.000001, le=1)
    batchSize: int = Field(32, ge=2, le=256)
    samples: int = Field(512, ge=64, le=4096)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    dataset: Literal["synthetic", "csv"] = "synthetic"
    csv: str | None = Field(None, max_length=8_000_000)
    validationFraction: float = Field(0.2, ge=0.1, le=0.4)
    earlyStopping: bool = True
    patience: int = Field(5, ge=3, le=30)


@app.get("/api/health")
def health():
    available = importlib.util.find_spec("torch") is not None
    cuda, device = False, "CPU"
    if available:
        try:
            import torch
            cuda = torch.cuda.is_available()
            device = torch.cuda.get_device_name(0) if cuda else "CPU"
        except Exception:
            available = False
    return {"torch": available, "cuda": cuda, "device": device, "apiVersion": "1.0.1", "pytorchImport": True}


@app.post("/api/analyze")
def analyze(graph: dict):
    from fastapi import HTTPException
    try: return analyze_graph(graph)
    except (ValueError, TypeError, KeyError) as error: raise HTTPException(422, str(error)) from error


class PyTorchImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = Field(max_length=512_000)
    model_name: str | None = Field(None, max_length=128)
    input_shapes: dict[str, list[int]] = Field(default_factory=dict, max_length=8)


@app.post("/api/import/pytorch")
def import_code(payload: PyTorchImportRequest):
    from .pytorch_import import import_pytorch
    return import_pytorch(payload.source, payload.model_name, payload.input_shapes)


@app.websocket("/api/train")
async def training_socket(ws: WebSocket):
    origin = ws.headers.get("origin")
    if origin and urlparse(origin).hostname not in ("localhost", "127.0.0.1", "::1"):
        await ws.close(code=1008); return
    await ws.accept()
    acquired = False; stop = threading.Event(); worker = None
    try:
        payload = await asyncio.wait_for(ws.receive_json(), timeout=15)
        config = TrainingConfig.model_validate(payload.get("config", {})).model_dump()
        graph = payload.get("graph"); analyze_graph(graph)
        if not health()["torch"]: raise ValueError("PyTorch is not installed. Run install.ps1 -Training")
        if not training_lock.acquire(blocking=False): raise ValueError("Another training run is already active")
        acquired = True; messages = queue.Queue()
        from .training import train
        def run():
            try: train(graph, config, messages.put, stop)
            except Exception as error: messages.put({"type": "error", "message": str(error)})
            finally: training_lock.release()
        worker = threading.Thread(target=run, daemon=True); worker.start()
        async def receive_commands():
            try:
                while True:
                    command = await ws.receive_json()
                    if command.get("type") == "stop": stop.set()
            except WebSocketDisconnect: stop.set()
        listener = asyncio.create_task(receive_commands())
        try:
            while True:
                if stop.is_set() and not worker.is_alive() and messages.empty(): break
                try: message = messages.get_nowait()
                except queue.Empty: await asyncio.sleep(0.05); continue
                await ws.send_json(message)
                if message["type"] in ("done", "error"): break
        finally:
            listener.cancel(); stop.set()
    except WebSocketDisconnect: stop.set()
    except Exception as error:
        try: await ws.send_json({"type": "error", "message": str(error)})
        except (RuntimeError, WebSocketDisconnect): pass
    finally:
        stop.set()
        if acquired and worker is None: training_lock.release()
        try: await ws.close()
        except RuntimeError: pass


dist = ROOT / "dist"
if dist.exists():
    app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")
    @app.get("/{path:path}")
    def frontend(path: str):
        target = (dist / path).resolve()
        if target.is_relative_to(dist.resolve()) and target.is_file(): return FileResponse(target)
        return FileResponse(dist / "index.html")
