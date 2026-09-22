import os
import sys
import json
import time
import argparse
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Support imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from inference.engine import VoxelInferenceEngine
from config.config import VoxelConfig

# Modèles Pydantic pour les requêtes
class ChatMessage(BaseModel):
    role: str = Field(..., description="Rôle ('system', 'user' ou 'assistant')")
    content: str = Field(..., description="Contenu du message")

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    repetition_penalty: Optional[float] = 1.15
    max_tokens: Optional[int] = 512
    enable_thinking: Optional[bool] = True

# Variable globale pour l'instance du moteur
engine: Optional[VoxelInferenceEngine] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Chargement du modèle unique en mémoire au démarrage
    global engine
    checkpoint = getattr(app.state, "checkpoint_path", "checkpoints/voxel_ai_final.pt")
    if not os.path.exists(checkpoint):
        alt = "checkpoints/latest_checkpoint.pt"
        if os.path.exists(alt):
            checkpoint = alt
        else:
            try:
                print(f"[*] Checkpoint introuvable localement ({checkpoint}).")
                print("[*] Téléchargement automatique des poids depuis Hugging Face Hub (tiagouille/voxel-ai)...")
                from huggingface_hub import hf_hub_download
                os.makedirs(os.path.dirname(checkpoint) or ".", exist_ok=True)
                hf_hub_download(repo_id="tiagouille/voxel-ai", filename="checkpoints/voxel_ai_final.pt", local_dir=".")
                print("[+] Poids téléchargés avec succès depuis Hugging Face !")
            except Exception as dl_err:
                print(f"[!] Téléchargement automatique Hugging Face impossible : {dl_err}")

    device = getattr(app.state, "device", "cpu")
    dtype = getattr(app.state, "dtype", "float32")
    tokenizer_dir = getattr(app.state, "tokenizer_dir", "tokenizer")

    print(f"[*] [Serveur Voxel AI] Initialisation du moteur d'inférence ({device}, {dtype})...")
    if os.path.exists(checkpoint):
        try:
            engine = VoxelInferenceEngine(
                checkpoint_path=checkpoint,
                tokenizer_dir=tokenizer_dir,
                device=device,
                dtype=dtype
            )
            print("[+] [Serveur Voxel AI] Modèle chargé en mémoire avec succès !")
        except Exception as e:
            print(f"[!] Erreur au chargement du modèle : {e}")
            engine = None
    else:
        print(f"[!] AVERTISSEMENT : Aucun checkpoint trouvé dans {checkpoint}. Le serveur démarre en attente de modèle.")
        engine = None

    yield
    print("[*] Arrêt du serveur Voxel AI...")

app = FastAPI(
    title="Voxel AI API",
    description="API locale haute performance pour le modèle conversationnel Voxel AI",
    version="1.0.0",
    lifespan=lifespan
)

# Configuration CORS pour autoriser l'accès depuis le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware pour désactiver tout cache navigateur sur le frontend et les fichiers statiques
@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.get("/api/health")
async def health_check():
    """Vérification de l'état de santé du serveur et de la présence du modèle."""
    return {
        "status": "online",
        "model_loaded": engine is not None,
        "device": str(engine.device) if engine else None,
        "dtype": engine.dtype_str if engine else None
    }

@app.get("/api/info")
async def model_info():
    """Informations sur l'architecture et les paramètres du modèle Voxel AI."""
    cfg_path = "config/model_config.json"
    cfg = VoxelConfig.load_from_json(cfg_path) if os.path.exists(cfg_path) else VoxelConfig()
    analytic = cfg.model.calculate_parameter_count()

    return {
        "name": cfg.model.name,
        "total_parameters": analytic["total_parameters"],
        "total_parameters_millions": round(analytic["total_parameters"] / 1e6, 2),
        "layers": cfg.model.n_layers,
        "d_model": cfg.model.d_model,
        "heads_q": cfg.model.n_heads,
        "heads_kv": cfg.model.n_kv_heads,
        "d_ff": cfg.model.d_ff,
        "max_context": cfg.model.max_seq_len,
        "model_loaded": engine is not None
    }

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """Génération de réponse conversationnelle classique (non-streaming)."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé. Veuillez vérifier la présence d'un checkpoint.")

    # Formatage de l'historique
    messages_dict = [{"role": m.role, "content": m.content} for m in req.messages]
    prompt = engine.tokenizer.apply_chat_template(messages_dict, add_generation_prompt=True)

    t0 = time.time()
    response_text = engine.generate(
        prompt=prompt,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        enable_thinking=bool(req.enable_thinking)
    )
    elapsed = time.time() - t0

    return {
        "role": "assistant",
        "content": response_text,
        "generation_time_sec": round(elapsed, 3)
    }

@app.post("/api/chat/stream")
async def chat_stream_endpoint(req: ChatRequest):
    """Génération de réponse conversationnelle en temps réel (Streaming SSE)."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé. Entraînez le modèle ou chargez un checkpoint.")

    messages_dict = [{"role": m.role, "content": m.content} for m in req.messages]
    prompt = engine.tokenizer.apply_chat_template(messages_dict, add_generation_prompt=True)

    def event_stream():
        for chunk in engine.stream_generate(
            prompt=prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            enable_thinking=bool(req.enable_thinking)
        ):
            payload = json.dumps({"token": chunk})
            yield f"data: {payload}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

# Distribution des fichiers statiques du site web
web_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web"))
if os.path.exists(web_dir):
    app.mount("/static", StaticFiles(directory=web_dir), name="static")

    @app.get("/")
    async def index_page():
        return FileResponse(os.path.join(web_dir, "index.html"))

def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    checkpoint: str = "checkpoints/voxel_ai_final.pt",
    device: str = "cpu",
    dtype: str = "float16",
    tokenizer_dir: str = "tokenizer"
):
    import uvicorn
    app.state.checkpoint_path = checkpoint
    app.state.device = device
    app.state.dtype = dtype
    app.state.tokenizer_dir = tokenizer_dir

    print("=" * 65)
    print(f"      DÉMARRAGE DU SERVEUR VOXEL AI")
    print("=" * 65)
    print(f" Adresse locale : http://localhost:{port}")
    print(f" Documentation API : http://localhost:{port}/docs")
    print(f" Interface Web : http://localhost:{port}/")
    print("=" * 65)

    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serveur API Voxel AI")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Hôte d'écoute")
    parser.add_argument("--port", type=int, default=8000, help="Port d'écoute")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/voxel_ai_final.pt", help="Chemin du checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="Périphérique ('cpu' ou 'cuda')")
    parser.add_argument("--dtype", type=str, default="float32", help="Précision ('float16', 'bfloat16' ou 'float32')")
    parser.add_argument("--tokenizer", type=str, default="tokenizer", help="Dossier du tokenizer")
    args = parser.parse_args()

    run_server(
        host=args.host,
        port=args.port,
        checkpoint=args.checkpoint,
        device=args.device,
        dtype=args.dtype,
        tokenizer_dir=args.tokenizer
    )
