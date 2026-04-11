import logging
import os
from collections import defaultdict

from ibm_watsonx_gov.clients.api_client import APIClient as GovAPIClient
from ibm_watsonx_gov.config import Credentials, GenAIConfiguration
from ibm_watsonx_gov.evaluators import MetricsEvaluator
from ibm_watsonx_gov.metrics import (
    AnswerRelevanceMetric,
    JailbreakMetric,
    SocialBiasMetric,
    TextGradeLevelMetric,
    TextReadingEaseMetric,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("OPENSCALE_API_KEY")

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
# Utility functions
# ---------------------------------------------------------------------------
def _calc_mean(records) -> dict[str, float]:
    metric_sums = defaultdict(float)
    metric_counts = defaultdict(int)

    for record in records:
        name = record.get("name")
        value = record.get("value")

        if name is None or value is None:
            continue

        metric_sums[name] += value
        metric_counts[name] += 1

    return {
        name: metric_sums[name] / metric_counts[name]
        for name in metric_sums
        if metric_counts[name] > 0
    }


# ---------------------------------------------------------------------------
# Evaluator entry point
# ---------------------------------------------------------------------------
def run_evaluator(data, asset_properties) -> dict[str, float]:
    if data.empty:
        logger.info("Input data is empty; skipping evaluation.")
        return {}

    metrics = [
        AnswerRelevanceMetric(),
        TextReadingEaseMetric(),
        SocialBiasMetric(),
        JailbreakMetric(),
        TextGradeLevelMetric(),
    ]

    config = GenAIConfiguration(
        input_fields=asset_properties.get("feature_fields", []),
        context_fields=asset_properties.get("context_fields", [])
        if asset_properties.get("problem_type") == "retrieval_augmented_generation"
        else [],
        output_fields=["generated_text"],
        reference_fields=[],
    )

    logger.info("Starting metrics evaluation.")

    result = MetricsEvaluator(
        api_client=GovAPIClient(credentials=Credentials(api_key=API_KEY)),
        configuration=config,
    ).evaluate(data=data, metrics=metrics)

    result_dict = result.to_dict()

    if not result_dict:
        logger.warning("Evaluator returned no metric results.")
        return {}

    return _calc_mean(result_dict)
