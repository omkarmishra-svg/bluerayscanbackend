from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app.config import settings
from app.routers import scan

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION
)

# CORS - Allow everything for hackathon simplicity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(scan.router, prefix=settings.API_PREFIX, tags=["Scan"])

# WebSocket Endpoint
from fastapi import WebSocket, WebSocketDisconnect
from app.services.websocket_manager import manager

@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep the connection open and listen for any client messages (optional)
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WS Error: {e}")
        manager.disconnect(websocket)

# Mount uploads directory for easy debugging access (e.g. localhost:8000/uploads/foo.mp4)
# Warning: Be careful with security in production
if os.path.exists(settings.UPLOAD_DIR):
    app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

@app.get("/")
async def health_check():
    return {
        "status": "online",
        "service": "BlueRayScan Backend", 
        "version": settings.VERSION
    }
