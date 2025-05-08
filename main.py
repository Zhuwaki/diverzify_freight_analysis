from fastapi import FastAPI
from app.api.cleaning_api import router as cleaning_router
from app.api.estimate_api import router as process_router
from app.api.ftl_model_api import router as ftl_model_router
from app.api.ltl_model_api import router as ltl_model_router
# from app.api.pipeline_api import router as pipeline_router

app = FastAPI(title="Freight Cost Estimator API", version="1.0")

app.include_router(cleaning_router, prefix="/api", tags=["Data Processing"])
app.include_router(ltl_model_router, prefix="/api",
                   tags=["XGS LTL Modelling - Version 1.0"])
app.include_router(ftl_model_router, prefix="/api",
                   tags=["XGS FTL Modelling - Version 1.0"])

app.include_router(process_router, prefix="/api", tags=["Ops v1"])
