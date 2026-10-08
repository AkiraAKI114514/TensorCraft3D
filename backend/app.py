import asyncio
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
from .cuda_environment import inspect_environment
from .inference import InferenceError, TensorInferenceRequest, run_inference
from .trained_models import trained_models

app = FastAPI(title="TensorCraft3D · Build and Explore Neural Networks in 3D", version="1.0.1")
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
    inspection = inspect_environment(full=False)
    return {"torch": inspection["torch"]["installed"], "cuda": inspection["cuda"]["available"], "device": inspection["cuda"]["devices"][0]["name"] if inspection["cuda"]["devices"] else "CPU", "apiVersion": "1.0.1", "pytorchImport": True, "tensorInference": True, "trainedInference": True, "attentionInference": True}


@app.post("/api/infer")
def infer(payload: TensorInferenceRequest):
    from fastapi import HTTPException
    if not training_lock.acquire(blocking=False):
        raise HTTPException(409, "Another training or inference run is already active")
    try:
        return run_inference(payload)
    except (InferenceError, ValueError, TypeError, KeyError) as error:
        raise HTTPException(422, str(error)) from error
    except RuntimeError as error:
        if "not installed" in str(error):
            raise HTTPException(503, str(error)) from error
        raise HTTPException(422, str(error)) from error
    finally:
        training_lock.release()


@app.get("/api/environment")
def environment():
    if not training_lock.acquire(blocking=False):
        from fastapi import HTTPException
        raise HTTPException(409, "训练期间无法执行 CUDA 环境检查")
    try:
        return inspect_environment(full=True)
    finally:
        training_lock.release()


@app.post("/api/environment/smoke")
def environment_smoke():
    from fastapi import HTTPException
    if not training_lock.acquire(blocking=False):
        raise HTTPException(409, "训练期间无法执行 CUDA smoke probe")
    try:
        inspection = inspect_environment(full=False)
        if not inspection["cuda"]["available"]:
            raise HTTPException(422, "当前 CUDA 不可用，无法执行 smoke probe")
        try:
            import torch
            device = torch.device("cuda")
            left = torch.randn((32, 32), device=device, requires_grad=True)
            loss = (left @ left.T).mean()
            loss.backward()
            torch.cuda.synchronize()
            if left.grad is None or not torch.isfinite(loss).item() or not torch.isfinite(left.grad).all().item():
                raise ValueError("CUDA 运算产生无效损失或梯度")
            return {"ok": True, "device": torch.cuda.get_device_name(0), "loss": float(loss.detach().cpu())}
        except Exception as error:
            raise HTTPException(422, f"CUDA smoke probe 失败：{str(error)[:400]}") from error
    finally:
        training_lock.release()


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
            try: train(graph, config, messages.put, stop, retain_model=trained_models.save)
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
