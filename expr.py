import pathlib
import os
import tempfile
from typing import List, Dict, Any

import pandas as pd
import xgboost as xgb
from toolz import curry, pipe
from attrs import frozen, evolve, field

import xorq.api as xo
import xorq.expr.datatypes as dt
from xorq.expr.ml import train_test_splits
from xorq.expr.ml.pipeline_lib import FittedPipeline
from xorq.expr.ml.structer import ENCODED
from xorq.caching import ParquetStorage
from xorq_sklearn_mortgage.lib import (
    ConnectionContext,
    DataConfig,
    FeatureConfig,
    ModelConfig,
    PipelineConfig,
    OneHotHelper,
    MortgageXGBoost,
    load_data,
)


@frozen
class MLPipelineResult:
    train_expr: Any = field()
    test_expr: Any = field()
    fitted_pipeline: FittedPipeline = field()
    predictions: Any = field()
    model: Any = field()
    deferred_model: Any = field(default=None)


def xgboost_to_quickgrove(
    booster: "xgb.Booster", feature_names: list = None, cleanup_temp_file: bool = True
):
    try:
        import xgboost as xgb
        import quickgrove
    except ImportError as e:
        missing_lib = "xgboost" if "xgboost" in str(e) else "quickgrove"
        raise ImportError(
            f"Required library '{missing_lib}' is not installed. Please install it with: pip install {missing_lib}"
        ) from e

    if not isinstance(booster, xgb.Booster):
        raise TypeError(f"Expected xgb.Booster, got {type(booster)}")

    if feature_names is None:
        feature_names = getattr(booster, "feature_names", None)
        if not feature_names:
            num_features = booster.num_feature()
            feature_names = [f"feature_{i}" for i in range(num_features)]

    booster.feature_names = feature_names

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_file = f.name

    try:
        booster.save_model(temp_file)
        quickgrove_model = quickgrove.json_load(temp_file)
        return quickgrove_model
    finally:
        if cleanup_temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)


def xgboost_to_quickgrove_udf(
    booster: "xgb.Booster",
    feature_names: list = None,
    model_name: str = "xgboost_model",
    cleanup_temp_file: bool = True,
):
    quickgrove_model = xgboost_to_quickgrove(
        booster, feature_names=feature_names, cleanup_temp_file=cleanup_temp_file
    )
    return make_quickgrove_udf(quickgrove_model, model_name=model_name)


def extract_onehot_with_ibis(t, model_features: List[str]):
    base_cols = [col for col in t.columns if col != "encoded"]
    onehot_features = [f for f in model_features if f not in base_cols]

    if not onehot_features:
        return t.select(base_cols)

    t_with_id = t.mutate(row_id=xo.row_number())

    unnested = (
        t_with_id.unnest("encoded")
        .mutate(key=xo._["encoded"]["key"], value=xo._["encoded"]["value"])
        .drop("encoded")
    )

    agg_dict = {}

    for col in base_cols:
        agg_dict[col] = unnested[col].first()

    for feature in onehot_features:
        agg_dict[feature] = (
            xo.case()
            .when(unnested.key == feature, unnested.value)
            .else_(0.0)
            .end()
            .sum()
        )

    return unnested.group_by("row_id").aggregate(**agg_dict).drop("row_id")


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
            xo._.high_ltv.any().name("high_ltv_flag"),
            xo._.high_dti.any().name("high_dti_flag"),
            xo._.subprime.any().name("subprime_flag"),
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
            mutate_exprs.append(getattr(xo._, col).fill_null(False).name(col))

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
    udf = xgboost_to_quickgrove_udf(result.model)

    fitted_onehot = result.fitted_pipeline.fitted_steps[0]
    t = fitted_onehot.transform(result.test_expr).cache(
        storage=ParquetStorage(source=ctx.duck_con)
    )

    wide_table = extract_onehot_with_ibis(t, result.model.feature_names)
    test_predicted = wide_table.into_backend(ctx.con).mutate(prediction=udf.on_expr)

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

    udf = xgboost_to_quickgrove_udf(result.model)

    fitted_onehot = result.fitted_pipeline.fitted_steps[0]
    t = (
        fitted_onehot.transform(processed_new_data).cache(
            storage=ParquetStorage(source=ctx.duck_con)
        )  # potential Filter to be pushed here
    )
    # actively require the user to request the rewrite

    # FIXME: If i try to cache wide table i get a `Compilation rule for
    # `TableUnnest` operation is not defined` error

    # this is needed to avoid the MortgageXGBoost do_explode_encode which we do
    # not have access to in this udf
    wide_table = extract_onehot_with_ibis(t, result.model.feature_names)

    new_predicted = wide_table.into_backend(ctx.con).mutate(prediction=udf.on_expr)

    return new_predicted


def create_xgboost_schema_wide(t, model_features) -> Dict[str, Any]:
    base_cols = [col for col in t.columns if col != "encoded"]
    onehot_features = [f for f in model_features if f not in base_cols]

    schema_dict = {}

    exclude_cols = {"loan_id", "ever_90_delinq", "row_id"}
    for col in base_cols:
        if col not in exclude_cols:
            col_type = t.schema()[col]

            if str(col_type) == "boolean":
                schema_dict[col] = dt.boolean
            elif "float" in str(col_type).lower():
                schema_dict[col] = (
                    dt.Float64(nullable=True) if col_type.nullable else dt.float64
                )
            elif "int" in str(col_type).lower():
                schema_dict[col] = (
                    dt.Int64(nullable=True) if col_type.nullable else dt.int64
                )
            else:
                schema_dict[col] = dt.Float64(nullable=True)

    for feature in onehot_features:
        schema_dict[feature] = dt.Float64(nullable=True)

    return schema_dict


def create_xgboost_pandas_udf(
    model: xgb.Booster, t, model_features, name: str = "xgboost_scorer"
):
    schema_dict = create_xgboost_schema_wide(t, model_features)

    @xo.udf.make_pandas_udf(
        schema=xo.schema(schema_dict), return_type=dt.float64, name=name
    )
    def score_xgboost(df):
        for col in df.columns:
            if df[col].dtype == bool:
                df[col] = df[col].astype(float)

        for feature in model.feature_names:
            if feature not in df.columns:
                df[feature] = 0.0

        feature_df = df[model.feature_names].fillna(0.0)

        dmatrix = xgb.DMatrix(feature_df)
        predictions = model.predict(dmatrix)

        return pd.Series(predictions)

    return score_xgboost


def xgboost_to_pandas_udf(
    model: xgb.Booster, t, model_features, name: str = "xgboost_scorer"
):
    return create_xgboost_pandas_udf(model, t, model_features, name)


def predict_new_data_with_xgboost_udf(
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

    fitted_onehot = result.fitted_pipeline.fitted_steps[0]

    t = fitted_onehot.transform(processed_new_data).cache(
        storage=ParquetStorage(source=ctx.duck_con)
    )

    udf = xgboost_to_pandas_udf(result.model, t, result.model.feature_names)

    wide_table = extract_onehot_with_ibis(t, result.model.feature_names)

    new_predicted = wide_table.into_backend(ctx.con).mutate(prediction=udf.on_expr)

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
    final_result = create_quickgrove_predictions(result, ctx)

    return final_result, config, ctx


def scratch():
    # transformed cache
    d = xo.deferred_read_parquet(
        "/home/hussainsultan/.cache/xorq/parquet/letsql_cache-1d75b96f58cd6a54c2f942616087b375.parquet",
        con=xo.duckdb.connect(),
    )
    wide_table = extract_onehot_with_ibis(d, result.model.feature_names)
    udf = xgboost_to_quickgrove_udf(result.model)
    r = wide_table.into_backend(xo.connect()).mutate(prediction=udf.on_expr)


def example_new_data_prediction_with_xgboost_udf():
    result, config, ctx = main()

    new_data_expr = load_data(evolve(config.data, filter_date="2001-01-01"), ctx).cache(
        storage=ParquetStorage(source=ctx.con)
    )

    new_predictions_quickgrove = predict_new_data_with_quickgrove(
        result, new_data_expr, config, ctx
    )

    new_predictions_xgboost = predict_new_data_with_xgboost_udf(
        result, new_data_expr, config, ctx
    )

    dct = {
        "quickgrove": new_predictions_quickgrove,
        "xgboost_udf": new_predictions_xgboost,
    }
    # df = pd.DataFrame({k: v.execute().set_index("loan_id")["prediction"] for k, v in dct.items()})
    return dct
