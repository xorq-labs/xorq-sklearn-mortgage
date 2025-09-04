from attrs import evolve
from codetiming import Timer

from xorq_sklearn_mortgage.pipeline_lib import (
    ModelConfig,
    PipelineConfig,
    MLPipelineResult,
)


config = PipelineConfig(
    model=evolve(ModelConfig(), num_boost_round=50, max_depth=12),
)
with Timer("MLPipelineResult creation"):
    result = MLPipelineResult(config)
with Timer("generate predict_quickgrove expr"):
    # 30 seconds per 100k rows predicted
    # # floor of 6 seconds because of duckdb join
    expr = new_predictions_quickgrove = result.predict_quickgrove(result.raw_expr)
with Timer("execute predictions"):
    df = expr.limit(200_000).execute()
