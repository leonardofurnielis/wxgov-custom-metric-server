import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import certifi
import nltk
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from utils.async_threading import run_in_thread
from wos.common import (
    extract_prompt_fields,
    get_dataset_records,
    log_metrics,
    log_record_metrics,
)
from wos.context_free_evaluator import (
    get_metrics_mean,
    get_records_metric,
    run_evaluator,
)

# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

try:
    nltk.download("cmudict", download_dir="./nltk_data")
    nltk.data.path.append("./nltk_data")
    logger.info("NLTK data initialized successfully.")
except Exception:
    logger.exception("Failed to initialize NLTK data.")
    raise

# ---------------------------------------------------------------------------
# Thread pool (single worker to protect shared resources)
# ---------------------------------------------------------------------------
executor = ThreadPoolExecutor(max_workers=1)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI()

# Add FastAPI CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def index() -> dict[str, str]:
    return {"status": "FastAPI is running!"}


@app.post("/compute/genai_context_free")
async def genai_context_free(request: Request):

    try:
        request_body: dict[str, Any] = await request.json()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON payload.",
        )
    input_data: list[dict[str, Any]] = request_body.get("input_data")

    if not input_data or not isinstance(input_data, list) or len(input_data) == 0:
        raise HTTPException(
            status_code=400,
            detail="'input_data' field is required and must be a non-empty list.",
        )

    data: dict[str, Any] = input_data[0].get("values")

    subscription_id = data.get("subscription_id")
    monitor_inst_id = data.get("custom_monitor_instance_id")
    payload_dataset_id = data.get("payload_dataset_id")
    run_id = data.get("custom_monitor_run_id")
    custom_dataset_id = data.get("custom_dataset_id")

    logger.info(
        "Processing | subscription_id=%s run_id=%s",
        subscription_id,
        run_id,
    )

    try:
        prompt_properties = extract_prompt_fields(subscription_id)
        payload_data = get_dataset_records(
            payload_dataset_id,
            monitor_inst_id,
            prompt_properties,
        )
    except Exception as exc:
        logger.exception("Failed to prepare evaluation payload.")
        raise HTTPException(status_code=500, detail=str(exc))

    try:
        loop = asyncio.get_event_loop()
        custom_metrics_result = await loop.run_in_executor(
            executor,
            run_in_thread,
            run_evaluator,
            payload_data,
            prompt_properties,
        )
    except Exception as exc:
        logger.exception("Custom evaluator execution failed.")
        raise HTTPException(status_code=500, detail=str(exc))

    try:
        metrics_mean = get_metrics_mean(custom_metrics_result)
        log_metrics(monitor_inst_id, run_id, metrics_mean)

        if custom_dataset_id:
            records_metrics = get_records_metric(custom_metrics_result)
            log_record_metrics(
                run_id,
                custom_dataset_id,
                payload_dataset_id,
                "payload",
                records_metrics,
            )
    except Exception as e:
        logger.exception("Failed to store metrics")
        return JSONResponse(
            content={"predictions": [], "errors": [str(e)]},
            status_code=500,
        )

    return JSONResponse(
        content={"predictions": [{"values": ["success"]}]}, status_code=200
    )
