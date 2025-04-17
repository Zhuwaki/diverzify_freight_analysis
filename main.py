from fastapi import FastAPI
from app.api.cleaning_api import router as cleaning_router
from app.api.process_api import router as process_router
from app.api.model_api import router as freight_router
# from app.api.pipeline_api import router as pipeline_router

app = FastAPI(title="Freight Cost Estimator API", version="1.0")

app.include_router(cleaning_router, prefix="/api", tags=["Data Preprocessing"])
app.include_router(freight_router, prefix="/api", tags=["Freight Modelling"])
app.include_router(process_router, prefix="/api", tags=["Freight Comparison"])
# app.include_router(pipeline_router, prefix="/api")
