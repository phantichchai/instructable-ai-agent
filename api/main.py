import uvicorn
from fastapi import FastAPI
from api.routers.video_ws import video_ws_router
from api.routers.generate_api import generate_router

app = FastAPI()
app.include_router(video_ws_router)
app.include_router(generate_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
