import logging
import os
from typing import Any, Optional

import pandas as pd
from donkey.observability.watsonx import WatsonxGovClient
from ibm_watson_openscale import APIClient
from ibm_watson_openscale.utils import IAMAuthenticator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("OPENSCALE_API_KEY")
RECORDS_LIMIT = os.environ.get("RECORDS_LIMIT", 500)

if not API_KEY:
    raise RuntimeError("Environment variable OPENSCALE_API_KEY is not set")

# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ---------------------------------------------------------------------------
# OpenScale utilities
# ---------------------------------------------------------------------------
def _get_client() -> APIClient:
    authenticator = IAMAuthenticator(apikey=API_KEY)
    return APIClient(authenticator=authenticator)


# ---------------------------------------------------------------------------
# Subscription utilities
# ---------------------------------------------------------------------------
def extract_prompt_fields(subscription_id) -> dict[str, Any]:
    wos_client = _get_client()

    payload = wos_client.subscriptions.get(
        subscription_id=subscription_id
    ).result.to_dict()

    try:
        entity = payload["entity"]
        asset = entity.get("asset", {})
        asset_props = entity.get("asset_properties", {})
        problem_type = asset.get("problem_type")
    except Exception as exc:
        logger.exception(
            "Invalid subscription payload structure for subscription_id=%s",
            subscription_id,
        )
        raise RuntimeError("Invalid subscription payload structure") from exc

    # Always extract feature_fields
    feature_fields = set(asset_props.get("feature_fields", []))

    # Only extract context_fields for RAG
    if problem_type == "retrieval_augmented_generation":
        context_fields = set(asset_props.get("context_fields", []))
        question_field = asset_props.get("question_field", None)

        # Remove duplicates
        feature_fields = feature_fields - context_fields

        # Combined fields
        all_fields = feature_fields | context_fields

        return {
            "problem_type": problem_type,
            "feature_fields": sorted(feature_fields),
            "question_field": question_field,
            "context_fields": sorted(context_fields),
            "fields": sorted(all_fields),
        }

    return {
        "problem_type": problem_type,
        "feature_fields": sorted(feature_fields),
        "fields": sorted(feature_fields),
    }


# ---------------------------------------------------------------------------
# Monitor run utilities
# ---------------------------------------------------------------------------
def get_last_run_date(monitor_instance_id) -> Optional[str]:
    try:
        wos_client = _get_client()

        runs_dict = wos_client.monitor_instances.list_runs(
            monitor_instance_id=monitor_instance_id
        ).result.to_dict()

        finished_runs = [
            r
            for r in runs_dict["runs"]
            if r.get("entity", {}).get("status", {}).get("state") == "finished"
        ]

        if not finished_runs:
            return ""

        return finished_runs[0].get("metadata", {}).get("created_at", "")

    except (IndexError, AttributeError):
        return ""


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------
def get_dataset_records(
    dataset_id, monitor_instance_id, asset_properties
) -> pd.DataFrame:
    wos_client = _get_client()

    start_date = get_last_run_date(monitor_instance_id)
    start_date = start_date or None
    logger.info("Dataset ID: %s", dataset_id)
    logger.info("Fetching dataset records greater than or equal to: %s", start_date)

    data = wos_client.data_sets.get_list_of_records(
        data_set_id=dataset_id, start=start_date, limit=RECORDS_LIMIT
    ).result

    rows = []

    for record in data.get("records", []):
        values = record.get("entity", {}).get("values", {})

        row = {
            field: values.get(field)
            for field in asset_properties["fields"]
            if field in values
        }

        row["record_id"] = values.get("scoring_id")
        row["generated_text"] = values.get("generated_text")
        rows.append(row)

    logger.info("Number of records in dataset: %d", len(rows))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metrics storage
# ---------------------------------------------------------------------------
def log_metrics(monitor_instance_id, run_id, metrics) -> None:
    try:
        custom_metric_mgr = WatsonxGovClient(
            authenticator=IAMAuthenticator(apikey=API_KEY),
        )

        custom_metric_mgr.log_measurements(
            monitor_instance_id=monitor_instance_id,
            run_id=run_id,
            request_records=metrics,
        )
    except Exception as exc:
        logger.error(
            "Failed to store metrics for monitor_instance_id=%s, run_id=%s",
            monitor_instance_id,
            run_id,
        )
        raise RuntimeError("Failed to store metrics data") from exc


def log_record_metrics(
    run_id, custom_data_set_id, reference_data_set_id, computed_on, metrics
) -> None:
    try:
        custom_metric_mgr = WatsonxGovClient(
            authenticator=IAMAuthenticator(apikey=API_KEY),
        )

        custom_metric_mgr.log_record_measurements(
            custom_data_set_id=custom_data_set_id,
            reference_data_set_id=reference_data_set_id,
            computed_on=computed_on,
            run_id=run_id,
            request_records=metrics,
        )
    except Exception as exc:
        logger.error(
            "Failed to store record metrics for custom_data_set_id=%s, run_id=%s",
            custom_data_set_id,
            run_id,
        )
        raise RuntimeError("Failed to store record metrics data") from exc
