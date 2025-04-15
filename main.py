from fastapi import FastAPI
from app.api.cleaning_api import router as cleaning_router
from app.api.process_api import router as process_router
from app.api.freight_api import router as freight_router

app = FastAPI(title="Freight Cost Estimator API", version="1.0")

app.include_router(cleaning_router, prefix="/api", tags=["Cleaning"])
app.include_router(process_router, prefix="/api", tags=["Processing"])
app.include_router(freight_router, prefix="/api", tags=["Freight"])
