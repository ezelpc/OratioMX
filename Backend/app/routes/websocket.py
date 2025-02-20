from fastapi import APIRouter, WebSocket
from typing import List


router = APIRouter()
active_connections: List[WebSocket] = []


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            for connection in active_connections:
                await connection.send_text(data)
    except Exception:
        active_connections.remove(websocket)
    