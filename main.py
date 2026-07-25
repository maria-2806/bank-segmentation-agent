import os
import sys
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from agent_backend.agent_orchestrator import AgentOrchestrator
from agent_backend.eda_tool import run_full_eda

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2")
CSV_PATH = "banking_customers.csv"
SEGMENTED_CSV_PATH = "segmented_customers.csv"

orchestrator = None
ollama_status = {"status": "unknown"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator, ollama_status
    orchestrator = AgentOrchestrator(ollama_host=OLLAMA_HOST, model_name=MODEL_NAME)
    ollama_status = orchestrator.check_ollama_status() or {"status": "offline"}
    if ollama_status.get("status") != "online":
        print(f"[WARNING] Ollama not reachable at {OLLAMA_HOST}: {ollama_status}")
    yield
    orchestrator.close()


app = FastAPI(title="Bank Segmentation Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str


@app.get("/api/status")
def get_status():
    return {"ollama": ollama_status}


@app.post("/api/chat")
def chat(req: ChatRequest):
    if ollama_status.get("status") != "online":
        raise HTTPException(status_code=503, detail=f"Ollama is offline. Start it and pull '{MODEL_NAME}'.")
    try:
        return orchestrator.execute_query(req.query, csv_path=CSV_PATH)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data")
def get_data():
    if not os.path.exists(CSV_PATH):
        raise HTTPException(status_code=404, detail=f"{CSV_PATH} not found. Run data_generator.py first.")
    return run_full_eda(CSV_PATH)


@app.get("/api/customer/{customer_id}")
def get_customer(customer_id: str):
    path = SEGMENTED_CSV_PATH if os.path.exists(SEGMENTED_CSV_PATH) else CSV_PATH
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"{path} not found. Run data_generator.py first.")
    df = pd.read_csv(path)
    row = df[df["customer_id"] == customer_id.upper()]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found.")
    return row.iloc[0].to_dict()


@app.get("/api/download")
def download_segments():
    if not os.path.exists(SEGMENTED_CSV_PATH):
        raise HTTPException(status_code=404, detail="No segmented data yet. Run a segmentation query first.")
    return FileResponse(SEGMENTED_CSV_PATH, filename="segmented_customers.csv", media_type="text/csv")


FRONTEND_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")
if os.path.isdir(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
