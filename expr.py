import xorq.api as xo
from attrs import evolve
from codetiming import Timer

from xorq_sklearn_mortgage.pipeline_lib import (
    MLPipelineResult,
    ModelConfig,
    PipelineConfig,
)
from xorq_sklearn_mortgage.rewrite import rewrite_quickgrove_expr

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
with Timer("add filter to expr"):
    filtered_expr = expr.filter(xo._.orig_rate<6)
with Timer("execute predictions (orig_rate<6)"):
    filtered_df = filtered_expr.execute()
with Timer("rewrite quickgrove expr"):
    rewritten_filtered_expr = rewrite_quickgrove_expr(filtered_expr)
with Timer("execute predictions (orig_rate<6 with rewrite)"):
    rewritten_filtered_df = rewritten_filtered_expr.execute()
