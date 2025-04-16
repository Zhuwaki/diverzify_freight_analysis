from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import requests
import io

router = APIRouter()

API_BASE = "http://localhost:8000"  # adjust if deployed externally


@router.post("/pipeline")
async def run_full_pipeline(file: UploadFile = File(...)):
    try:
        # Step 1: Send to CLEANING API
        files = {"file": (file.filename, await file.read(), file.content_type)}
        clean_res = requests.post(f"{API_BASE}/clean", files=files)
        if clean_res.status_code != 200:
            return {"error": "Cleaning failed", "details": clean_res.json()}

        cleaned_data = clean_res.json()
        full_clean_json = cleaned_data.get(
            "preview")  # or full_data if returned

        # Step 2: Send to MODEL API
        model_res = requests.post(f"{API_BASE}/model", json=full_clean_json)
        if model_res.status_code != 200:
            return {"error": "Modeling failed", "details": model_res.json()}

        model_data = model_res.json()

        # Step 3: Send to PROCESS API
        process_payload = {
            "cleaned_data": full_clean_json,
            "model_output": model_data.get("preview")  # same strategy
        }
        process_res = requests.post(
            f"{API_BASE}/process", json=process_payload)
        if process_res.status_code != 200:
            return {"error": "Processing failed", "details": process_res.json()}

        final_output = process_res.json()

        return JSONResponse(content={
            "message": "🚀 Full pipeline complete",
            "clean_download": cleaned_data["download_url"],
            "model_download": model_data["download_url"],
            "final_download": final_output["download_url"],
            "final_preview": final_output["preview"]
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
