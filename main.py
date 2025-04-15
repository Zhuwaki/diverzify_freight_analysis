from fastapi import FastAPI
from app.api.cleaning_api import router as cleaning_router
from app.api.process_api import router as process_router

app = FastAPI()

app.include_router(cleaning_router, prefix="/api", tags=["Cleaning"])
app.include_router(process_router, prefix="/api", tags=["Processing"])
