from fastapi import WebSocket
from typing import List
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"Client connected. Active connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"Client disconnected. Active connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """
        Sends a JSON message to all connected clients.
        Ignores errors if a single client is disconnected ungracefully.
        """
        encoded_message = json.dumps(message)
        for connection in self.active_connections:
            try:
                await connection.send_text(encoded_message)
            except Exception as e:
                print(f"Failed to send to client: {e}")
                # Optional: cleanup dead connection here or let disconnect handle it
                
manager = ConnectionManager()
