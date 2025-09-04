import os
import pathlib
import tempfile
from pathlib import Path
from typing import Any, List, Union

import pandas as pd
import xorq.api as xo
import xorq.expr.datatypes as dt
from attrs import evolve, field, frozen
from quickgrove import PyGradientBoostedDecisionTrees
from toolz import curry, pipe
from xorq.caching import ParquetStorage
from xorq.expr.ml import train_test_splits
from xorq.expr.ml.pipeline_lib import FittedPipeline
from xorq.expr.ml.quickgrove_lib import UDFWrapper
from xorq.expr.ml.structer import ENCODED
from xorq.vendor.ibis.common.annotations import Argument
from xorq.vendor.ibis.expr.rules import ValueOf

from xorq_sklearn_mortgage.pipeline_lib import (
    ConnectionContext,
    DataConfig,
    FeatureConfig,
    ModelConfig,
    MortgageXGBoost,
    OneHotHelper,
    PipelineConfig,
    load_data,
)

SUPPORTED_TYPES = {
    "float": dt.float64,
    "i": dt.boolean,
    "int": dt.int64,
}


@frozen
class MLPipelineResult:
    train_expr: Any = field()
    test_expr: Any = field()
    fitted_pipeline: FittedPipeline = field()
    predictions: Any = field()
    model: Any = field()
    deferred_model: Any = field(default=None)


def mortgage_xgboost_to_quickgrove(
    booster: "MortgageXGBoost", cleanup_temp_file: bool = True
):
    import quickgrove

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_file = f.name

    try:
        booster.model.save_model(temp_file)
        quickgrove_model = quickgrove.json_load(temp_file)
        return quickgrove_model
    finally:
        if cleanup_temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)


def mortgage_xgboost_to_quickgrove_udf(
    booster: "MortgageXGBoost",
    model_name: str = "xgboost_model",
    cleanup_temp_file: bool = True,
):
    quickgrove_model = mortgage_xgboost_to_quickgrove(
        booster, cleanup_temp_file=cleanup_temp_file
    )

    return make_quickgrove_udf(quickgrove_model, feature_names=booster.feature_names)


def make_df_cheap(series):
    values = [[dct["value"] for dct in lst] for lst in series]
    columns = [dct["key"] for dct in series.iloc[0]]
    df = pd.DataFrame(
        values,
        index=series.index,
        columns=columns,
    )
    return df


def make_quickgrove_udf(
    model_or_path: Union[str, Path, "PyGradientBoostedDecisionTrees"],
    feature_names: List[str],
    model_name="quickgrove_model_xgb",
) -> UDFWrapper:
    from xorq.expr.ml.quickgrove_lib import (
        SUPPORTED_TYPES,
        UDFWrapper,
        _create_udf_function,
        _create_udf_node,
        _load_quickgrove_model,
        _validate_model_features,
    )

    model, model_path = _load_quickgrove_model(model_or_path)
    model_path = "temp"

    _validate_model_features(model, SUPPORTED_TYPES)

    fields = {}
    for feature in feature_names:
        fields[feature] = Argument(
            pattern=ValueOf(dt.float64),
            typehint=dt.float64,
        )

    fields[ENCODED] = Argument(
        pattern=ValueOf(dt.Array(dt.Struct({"key": dt.string, "value": dt.float64}))),
        typehint=dt.Array(dt.Struct({"key": dt.string, "value": dt.float64})),
    )

    def fn_from_arrays(*arrays):
        if len(arrays) != len(feature_names):
            raise ValueError(f"Expected {len(feature_names)} arrays, got {len(arrays)}")

        encoded_array = arrays[-1]

        base_df = pd.DataFrame(
            {feature: array for feature, array in zip(feature_names, arrays)}
        ).drop(columns=ENCODED)

        encoded_series = pd.Series(encoded_array)
        onehot_df = make_df_cheap(encoded_series)

        combined_df = pd.concat([base_df, onehot_df], axis=1)

        feature_matrix = pd.DataFrame()
        all_features = list(combined_df.columns)
        for feature in list(all_features):
            feature_matrix[feature] = combined_df[feature].fillna(0.0)

        feature_arrays = [feature_matrix[feature].values for feature in all_features]
        return model.predict_arrays(feature_arrays)

    required_features = list(feature_names)

    udf_func = _create_udf_function(
        _create_udf_node(
            model=model,
            fn_from_arrays=fn_from_arrays,
            fields=fields,
            udf_name=model_name,
            extra_meta={"model_path": model_path},
        )
    )

    return UDFWrapper(
        func=udf_func,
        model=model,
        model_path=model_path,
        required_features=required_features,
    )


def create_features(expr):
    return expr.mutate(
        [
            xo._.current_loan_delinquency_status.fillna(0).name("delinq_status"),
            xo._.dti.fillna(0).name("dti"),
            xo._.borrower_credit_score.fillna(650).name("credit_score"),
            (xo._.orig_ltv > 80).name("high_ltv"),
            (xo._.dti > 43).name("high_dti"),
            (xo._.borrower_credit_score < 620).name("subprime"),
            (xo._.current_actual_upb / xo._.orig_upb).name("balance_ratio"),
            (xo._.current_loan_delinquency_status >= 3).name("target_delinquent"),
        ]
    )


def create_loan_summary(expr):
    return expr.group_by("loan_id").aggregate(
        [
            xo._.orig_interest_rate.first().name("orig_rate"),
            xo._.orig_ltv.first().cast("float64").name("orig_ltv"),
            xo._.dti.first().cast("float64").name("dti"),
            xo._.credit_score.first().cast("float64").name("credit_score"),
            xo._.orig_upb.first().cast("float64").name("orig_upb"),
            xo._.property_state.first().name("property_state"),
            xo._.loan_purpose.first().name("loan_purpose"),
            xo._.property_type.first().name("property_type"),
            xo._.delinq_status.max().cast("float64").name("max_delinquency"),
            xo._.target_delinquent.max().cast("float64").name("ever_90_delinq"),
            xo._.balance_ratio.last().name("current_balance_ratio"),
            xo._.high_ltv.any().cast("float64").name("high_ltv_flag"),
            xo._.high_dti.any().cast("float64").name("high_dti_flag"),
            xo._.subprime.any().cast("float64").name("subprime_flag"),
        ]
    )


@curry
def clean_features(config: FeatureConfig, expr):
    mutate_exprs = []

    for col in config.numeric_features:
        if col in expr.columns:
            fill_value = (
                0 if col in ["orig_ltv", "dti", "credit_score", "orig_upb"] else 0.0
            )
            mutate_exprs.append(getattr(xo._, col).fill_null(fill_value).name(col))

    for col in config.categorical_features:
        if col in expr.columns:
            mutate_exprs.append(getattr(xo._, col).fill_null("Unknown").name(col))

    for col in config.flag_features:
        if col in expr.columns:
            mutate_exprs.append(getattr(xo._, col).fill_null(0).name(col))

    mutate_exprs.append(xo._[config.target_col].fill_null(0).name(config.target_col))

    available_features = [f for f in config.all_features if f in expr.columns]

    return expr.select(["loan_id"] + available_features + [config.target_col]).mutate(
        mutate_exprs
    )


@curry
def create_train_test_split(config: PipelineConfig, ctx: ConnectionContext, expr):
    ml_with_row = expr.mutate(row_id=xo.row_number())
    train_expr, test_expr = train_test_splits(
        ml_with_row,
        "row_id",
        **config.split.split_kwargs,
    )

    storage = ParquetStorage(source=ctx.con)
    train_clean = train_expr.drop("row_id").cache(storage=storage)
    test_clean = test_expr.drop("row_id").cache(storage=storage)

    return train_clean, test_clean


def create_pipeline_steps(config: PipelineConfig):
    one_hot_step = OneHotHelper.get_step()
    xgb_step = MortgageXGBoost.get_step()
    return one_hot_step, xgb_step


def fit_pipeline(config: PipelineConfig, train_expr, test_expr):
    one_hot_step = OneHotHelper.get_step()
    xgb_step = MortgageXGBoost.get_step(config)
    con = xo.connect()
    storage = ParquetStorage(source=con)

    fitted_onehot = one_hot_step.fit(
        train_expr,
        features=config.features.categorical_features,
        dest_col=ENCODED,
        storage=storage,
    )

    fitted_xgb = xgb_step.fit(
        expr=train_expr.mutate(fitted_onehot.mutate),
        features=list(config.features.numeric_features)
        + list(config.features.flag_features)
        + [ENCODED],
        target=config.features.target_col,
        dest_col="predicted_prob",
        storage=storage,
    )

    pipeline = FittedPipeline((fitted_onehot, fitted_xgb), train_expr)

    predictions = pipeline.predict(test_expr).mutate(
        predicted_class=(xo._.predicted_prob >= config.model.prediction_threshold).cast(
            "int"
        )
    )
    return MLPipelineResult(
        train_expr=train_expr,
        test_expr=test_expr,
        fitted_pipeline=pipeline,
        predictions=predictions,
        model=fitted_xgb.model,
        deferred_model=fitted_xgb.model_udf,
    )


def create_quickgrove_predictions(result: MLPipelineResult, ctx: ConnectionContext):
    udf = mortgage_xgboost_to_quickgrove_udf(result.model)

    fitted_onehot = result.fitted_pipeline.fitted_steps[0]

    t = fitted_onehot.transform(result.test_expr).cache(
        storage=ParquetStorage(source=ctx.con)
    )

    test_predicted = t.mutate(prediction=udf.on_expr)

    return evolve(result, predictions=test_predicted)


def predict_new_data(result: MLPipelineResult, new_data_expr, config: PipelineConfig):
    # sklearn prediction

    # this takes a long time
    processed_new_data = (
        pipe(
            new_data_expr,
            create_features,
            create_loan_summary,
            lambda expr: clean_features(
                config.features, expr
            ),  # spurious into_backend?
        ).into_backend(con=xo.connect())
        # if I remove it I get InvalidInputException: Invalid Input Error:
        # Python exception occurred while executing the UDF: ValueError: Caller
        # must bind computed_arg to the output of computed_kwargs_expr
    )

    new_predictions = result.fitted_pipeline.predict(processed_new_data).mutate(
        predicted_class=(xo._.predicted_prob >= config.model.prediction_threshold).cast(
            "int"
        )
    )

    return new_predictions


def predict_new_data_with_quickgrove(
    result: MLPipelineResult,
    new_data_expr,
    config: PipelineConfig,
    ctx: ConnectionContext,
):
    processed_new_data = pipe(
        new_data_expr,
        create_features,
        create_loan_summary,
        lambda expr: clean_features(config.features, expr),
    )

    udf = mortgage_xgboost_to_quickgrove_udf(result.model)

    fitted_onehot = result.fitted_pipeline.fitted_steps[0]
    t = (
        fitted_onehot.transform(processed_new_data).cache(
            storage=ParquetStorage(source=ctx.con)
        )  # potential Filter to be pushed here
    )
    # FIXME: If i try to cache wide table i get a `Compilation rule for
    # `TableUnnest` operation is not defined` error
    new_predicted = t.mutate(prediction=udf.on_expr)

    return new_predicted


def create_mortgage_pipeline(config: PipelineConfig):
    def execute_pipeline():
        ctx = ConnectionContext.create()

        load_data_fn = load_data(config.data, ctx)
        clean_data_fn = clean_features(config.features)
        split_data_fn = create_train_test_split(config, ctx)

        return pipe(
            load_data_fn,
            create_features,
            create_loan_summary,
            clean_data_fn,
            split_data_fn,
            lambda split: fit_pipeline(config, split[0], split[1]),
        )
    return execute_pipeline


def main():
    data_root = pathlib.Path(os.getenv("DATA_ROOT", "/mnt/data/fanniemae"))

    config = PipelineConfig(
        data=DataConfig(data_root=data_root),
        model=evolve(ModelConfig(), num_boost_round=50, max_depth=12),
    )

    pipeline = create_mortgage_pipeline(config)
    result = pipeline()
    ctx = ConnectionContext.create()

    return result, config, ctx


def example_new_data_prediction_with_xgboost_udf():
    result, config, ctx = main()

    new_data_expr = load_data(evolve(config.data, filter_date="2001-01-01"), ctx).cache(
        storage=ParquetStorage(source=ctx.con)
    )

    new_predictions_quickgrove = predict_new_data_with_quickgrove(
        result, new_data_expr, config, ctx
    )

    dct = {
        "quickgrove": new_predictions_quickgrove,
    }
    # df = pd.DataFrame({k: v.execute().set_index("loan_id")["prediction"] for k, v in dct.items()})
    return dct
