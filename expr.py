from attrs import evolve
from xorq_sklearn_mortgage.pipeline_lib import (
    ModelConfig,
    PipelineConfig,
    MLPipelineResult,
)
from xorq_sklearn_mortgage.quickgrove_lib import (
    mortgage_xgboost_to_quickgrove_udf,
)


def predict_new_data_with_quickgrove(
    result: MLPipelineResult,
    new_data_expr,
):
    udf = mortgage_xgboost_to_quickgrove_udf(result.model)
    new_predicted = (
        result.process_expr(new_data_expr)
        .into_backend(result.ctx.con)
        .pipe(result.fitted_onehot.transform)
        # potential Filter to be pushed here
        .cache(storage=result.storage)
        # FIXME: If i try to cache wide table i get a `Compilation rule for
        # `TableUnnest` operation is not defined` error
        .mutate(prediction=udf.on_expr)
    )
    return new_predicted


def main():
    config = PipelineConfig(
        model=evolve(ModelConfig(), num_boost_round=50, max_depth=12),
    )
    result = MLPipelineResult(config)
    new_predictions_quickgrove = predict_new_data_with_quickgrove(
        result, result.raw_expr,
    )
    return result, new_predictions_quickgrove
