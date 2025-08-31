import pickle
import pathlib
import os
import tempfile
import time
from typing import Optional, Tuple, List, Dict, Any, Callable, Union
from functools import singledispatch, partial

import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score
from toolz import curry, pipe
from attrs import frozen, evolve, field

import xorq.api as xo
from xorq.api import selectors as s
import xorq.expr.datatypes as dt
from xorq.expr.ml import train_test_splits
from xorq.expr.ml.fit_lib import transform_sklearn_feature_names_out
from xorq.expr.ml.pipeline_lib import FittedPipeline, Step
from xorq.expr.ml.structer import ENCODED
from xorq.caching import ParquetStorage
from xorq.ml import make_quickgrove_udf, rewrite_quickgrove_expr


@frozen
class DataConfig:
    data_root: pathlib.Path = field(converter=pathlib.Path)
    perf_path: str = field(default="data/perf/perf.parquet")
    acq_path: str = field(default="data/acq/acq.parquet")
    filter_date: str = field(default="2001-01-01") # data that works without ArrowInvalid: offset overflow while concatenating arrays
    
    def with_data_root(self, data_root: Union[str, pathlib.Path]) -> 'DataConfig':
        return evolve(self, data_root=pathlib.Path(data_root))


@frozen
class FeatureConfig:
    numeric_features: Tuple[str, ...] = field(default=(
        'orig_rate', 'orig_ltv', 'dti', 'credit_score', 
        'orig_upb', 'current_balance_ratio'
    ))
    categorical_features: Tuple[str, ...] = field(default=(
        'property_state', 'loan_purpose', 'property_type'
    ))
    flag_features: Tuple[str, ...] = field(default=(
        'high_ltv_flag', 'high_dti_flag', 'subprime_flag'
    ))
    target_col: str = field(default='ever_90_delinq')
    
    @property
    def all_features(self) -> Tuple[str, ...]:
        return self.numeric_features + self.categorical_features + self.flag_features


@frozen
class ModelConfig:
    num_boost_round: int = field(default=100)
    max_depth: int = field(default=6)
    eta: float = field(default=0.1)
    objective: str = field(default='binary:logistic')
    eval_metric: str = field(default='logloss')
    seed: int = field(default=42)
    prediction_threshold: float = field(default=0.5)
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            'max_depth': self.max_depth,
            'eta': self.eta,
            'objective': self.objective,
            'eval_metric': self.eval_metric,
            'seed': self.seed
        }


@frozen
class PipelineConfig:
    data: DataConfig = field(factory=DataConfig)
    features: FeatureConfig = field(factory=FeatureConfig)
    model: ModelConfig = field(factory=ModelConfig)
    test_size: float = field(default=0.5)
    random_seed: int = field(default=42)


@frozen
class ConnectionContext:
    con: Any = field()
    duck_con: Any = field()
    
    @classmethod
    def create(cls) -> 'ConnectionContext':
        return cls(
            con=xo.connect(),
            duck_con=xo.duckdb.connect()
        )


@frozen
class MLPipelineResult:
    train_expr: Any = field()
    test_expr: Any = field()
    fitted_pipeline: FittedPipeline = field()
    predictions: Any = field()
    model: Any = field()
    deferred_model: Any = field(default=None)


def xgboost_to_quickgrove(booster: "xgb.Booster", feature_names: list = None, cleanup_temp_file: bool = True):
    try:
        import xgboost as xgb
        import quickgrove
    except ImportError as e:
        missing_lib = "xgboost" if "xgboost" in str(e) else "quickgrove"
        raise ImportError(f"Required library '{missing_lib}' is not installed. Please install it with: pip install {missing_lib}") from e
    
    if not isinstance(booster, xgb.Booster):
        raise TypeError(f"Expected xgb.Booster, got {type(booster)}")
    
    if feature_names is None:
        feature_names = getattr(booster, 'feature_names', None)
        if not feature_names:
            num_features = booster.num_feature()
            feature_names = [f"feature_{i}" for i in range(num_features)]
    
    booster.feature_names = feature_names
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
    
    try:
        booster.save_model(temp_file)
        quickgrove_model = quickgrove.json_load(temp_file)
        return quickgrove_model
    finally:
        if cleanup_temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)


def xgboost_to_quickgrove_udf(booster: "xgb.Booster", feature_names: list = None, model_name: str = "xgboost_model", cleanup_temp_file: bool = True):
    quickgrove_model = xgboost_to_quickgrove(
        booster, 
        feature_names=feature_names,
        cleanup_temp_file=cleanup_temp_file
    )
    return make_quickgrove_udf(quickgrove_model, model_name=model_name)


def extract_onehot_with_ibis(t, model_features: List[str]):
    base_cols = [col for col in t.columns if col != 'encoded']
    onehot_features = [f for f in model_features if f not in base_cols]
    
    if not onehot_features:
        return t.select(base_cols)
    
    t_with_id = t.mutate(row_id=xo.row_number())
    
    unnested = (
        t_with_id
        .unnest('encoded')
        .mutate(
            key=xo._['encoded']['key'],
            value=xo._['encoded']['value']
        )
        .drop('encoded')
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
    
    return (
        unnested
        .group_by('row_id')
        .aggregate(**agg_dict)
        .drop('row_id')
    )


class OneHotStep(OneHotEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_names_out_ = None
    
    def transform(self, *args, **kwargs):
        return transform_sklearn_feature_names_out(super(), *args, **kwargs)
    
    def fit(self, X, y=None):
        result = super().fit(X, y)
        if hasattr(self, 'get_feature_names_out'):
            self.feature_names_out_ = self.get_feature_names_out()
        return result
    
    @classmethod
    def get_step_f_kwargs(cls, kwargs):
        from xorq.expr.ml.fit_lib import deferred_fit_transform_sklearn
        return (deferred_fit_transform_sklearn, kwargs | {
            "return_type": dt.Array(dt.Struct({"key": str, "value": float})),
            "target": None,
        })


class MortgageXGBoost(BaseEstimator):
    def __init__(self, num_boost_round=100, encoded_col=ENCODED, **params):
        self.num_boost_round = num_boost_round
        self.encoded_col = encoded_col
        self.params = {
            'max_depth': params.get('max_depth', 6),
            'eta': params.get('eta', 0.1), 
            'objective': params.get('objective', 'binary:logistic'),
            'eval_metric': params.get('eval_metric', 'logloss'),
            'seed': params.get('seed', 42)
        }
        self.model = None
    
    return_type = dt.float64
    
    def explode_encoded(self, X):
        return X.drop(columns=self.encoded_col).join(
            X[self.encoded_col].apply(
                lambda lst: pd.Series({d["key"]: d["value"] for d in lst})
            )
        )
    
    def fit(self, X, y):
        X_exploded = self.explode_encoded(X)
        dtrain = xgb.DMatrix(X_exploded, y)
        self.model = xgb.train(self.params, dtrain, self.num_boost_round)
        return self
    
    def predict(self, X):
        X_exploded = self.explode_encoded(X)
        return self.model.predict(xgb.DMatrix(X_exploded))


@curry
def load_data(config: DataConfig, ctx: ConnectionContext):
    perf_expr = xo.deferred_read_parquet(
        str(config.data_root / config.perf_path), 
        ctx.duck_con, 
        "perf_raw"
    )
    acq_expr = xo.deferred_read_parquet(
        str(config.data_root / config.acq_path), 
        ctx.duck_con, 
        "acq_raw"
    )
    
    return (
        acq_expr
        .join(perf_expr, acq_expr.loan_id == perf_expr.loan_id, how="left")
        .filter(xo._.monthly_reporting_period <= config.filter_date)
    )


def create_features(expr):
    return expr.mutate([
        xo._.current_loan_delinquency_status.fillna(0).name('delinq_status'),
        xo._.dti.fillna(0).name('dti'),
        xo._.borrower_credit_score.fillna(650).name('credit_score'),
        
        (xo._.orig_ltv > 80).name('high_ltv'),
        (xo._.dti > 43).name('high_dti'),
        (xo._.borrower_credit_score < 620).name('subprime'),
        (xo._.current_actual_upb / xo._.orig_upb).name('balance_ratio'),
        
        (xo._.current_loan_delinquency_status >= 3).name('target_delinquent')
    ])


def create_loan_summary(expr):
    return expr.group_by('loan_id').aggregate([
        xo._.orig_interest_rate.first().name('orig_rate'),
        xo._.orig_ltv.first().cast('float64').name('orig_ltv'), 
        xo._.dti.first().cast('float64').name('dti'),
        xo._.credit_score.first().cast('float64').name('credit_score'),
        xo._.orig_upb.first().cast('float64').name('orig_upb'),
        xo._.property_state.first().name('property_state'),
        xo._.loan_purpose.first().name('loan_purpose'),
        xo._.property_type.first().name('property_type'),
        
        xo._.delinq_status.max().cast('float64').name('max_delinquency'),
        xo._.target_delinquent.max().cast('float64').name('ever_90_delinq'),
        xo._.balance_ratio.last().name('current_balance_ratio'),
        
        xo._.high_ltv.any().name('high_ltv_flag'),
        xo._.high_dti.any().name('high_dti_flag'), 
        xo._.subprime.any().name('subprime_flag')
    ])


@curry
def clean_features(config: FeatureConfig, expr):
    mutate_exprs = []
    
    for col in config.numeric_features:
        if col in expr.columns:
            fill_value = 0 if col in ['orig_ltv', 'dti', 'credit_score', 'orig_upb'] else 0.0
            mutate_exprs.append(getattr(xo._, col).fill_null(fill_value).name(col))
    
    for col in config.categorical_features:
        if col in expr.columns:
            mutate_exprs.append(getattr(xo._, col).fill_null('Unknown').name(col))
    
    for col in config.flag_features:
        if col in expr.columns:
            mutate_exprs.append(getattr(xo._, col).fill_null(False).name(col))
    
    mutate_exprs.append(xo._[config.target_col].fill_null(0).name(config.target_col))
    
    available_features = [f for f in config.all_features if f in expr.columns]
    
    return (
        expr
        .select(['loan_id'] + available_features + [config.target_col])
        .mutate(mutate_exprs)
    )


@curry
def create_train_test_split(config: PipelineConfig, ctx: ConnectionContext, expr):
    ml_with_row = expr.mutate(row_id=xo.row_number())
    train_expr, test_expr = train_test_splits(
        ml_with_row, 'row_id', 
        test_sizes=config.test_size, 
        random_seed=config.random_seed
    )
    
    storage = ParquetStorage(source=ctx.con)
    train_clean = train_expr.drop('row_id').cache(storage=storage)
    test_clean = test_expr.drop('row_id').cache(storage=storage)
    
    return train_clean, test_clean


def create_pipeline_steps(config: PipelineConfig):
    one_hot_step = Step(
        OneHotStep,
        "one_hot_encoder", 
        params_tuple=(("handle_unknown", "ignore"), ("drop", "first"))
    )

    xgb_step = Step(
        MortgageXGBoost,
        "xgboost_model",
        params_tuple=(
            ("num_boost_round", config.model.num_boost_round),
            ("encoded_col", ENCODED),
            ("max_depth", config.model.max_depth),
            ("eta", config.model.eta),
            ("objective", config.model.objective),
            ("eval_metric", config.model.eval_metric),
            ("seed", config.model.seed)
        )
    )
    
    return one_hot_step, xgb_step


def fit_pipeline(config: PipelineConfig, train_expr, test_expr):
    one_hot_step, xgb_step = create_pipeline_steps(config)

    con = xo.connect()
    
    fitted_onehot = one_hot_step.fit(
        train_expr,
        features=config.features.categorical_features,
        dest_col=ENCODED,
        storage=ParquetStorage(source=con),
    )

    fitted_xgb = xgb_step.fit(
        expr=train_expr.mutate(fitted_onehot.mutate),
        features=list(config.features.numeric_features) + 
                list(config.features.flag_features) + [ENCODED],
        target=config.features.target_col,
        dest_col='predicted_prob',
        storage=ParquetStorage(source=con),
    )

    pipeline = FittedPipeline((fitted_onehot, fitted_xgb), train_expr)

    
    predictions = pipeline.predict(test_expr).mutate(
        predicted_class=(xo._.predicted_prob >= config.model.prediction_threshold).cast('int')
    )
    
    deferred_model = fitted_xgb.deferred_model
    
    model = pickle.loads(deferred_model.execute().iloc[0,0]).model
    
    return MLPipelineResult(
        train_expr=train_expr,
        test_expr=test_expr,
        fitted_pipeline=pipeline,
        predictions=predictions,
        model=model,
        deferred_model=fitted_xgb.model_udf,
    )


def create_quickgrove_predictions(result: MLPipelineResult, ctx: ConnectionContext):
    udf = xgboost_to_quickgrove_udf(result.model)
    
    fitted_onehot = result.fitted_pipeline.fitted_steps[0]
    t = fitted_onehot.transform(result.test_expr).cache(storage=ParquetStorage(source=ctx.duck_con))
    
    wide_table = extract_onehot_with_ibis(t, result.model.feature_names)
    test_predicted = wide_table.into_backend(ctx.con).mutate(
        prediction=udf.on_expr
    )
    
    return evolve(result, predictions=test_predicted)


def predict_new_data(result: MLPipelineResult, new_data_expr, config: PipelineConfig):
    # sklearn prediction

    processed_new_data = (
        pipe(
            new_data_expr,
            create_features,
            create_loan_summary,
            lambda expr: clean_features(config.features, expr) # spurious into_backend?
        )
        .into_backend(con=xo.connect()) 
        # if I remove it I get InvalidInputException: Invalid Input Error:
        # Python exception occurred while executing the UDF: ValueError: Caller
        # must bind computed_arg to the output of computed_kwargs_expr
    )
    
    new_predictions = (
            result
            .fitted_pipeline
            .predict(
                processed_new_data
            )
            .mutate(
                predicted_class=(
                    xo._.predicted_prob >= config.model.prediction_threshold
                    )
                   .cast('int')
            )
        )
    
    return new_predictions


def predict_new_data_with_quickgrove(result: MLPipelineResult, new_data_expr, config: PipelineConfig, ctx: ConnectionContext):
    processed_new_data = pipe(
        new_data_expr,
        create_features,
        create_loan_summary,
        lambda expr: clean_features(config.features, expr)
    )
    
    udf = xgboost_to_quickgrove_udf(result.model)
    
    fitted_onehot = result.fitted_pipeline.fitted_steps[0]
    t = (
        fitted_onehot
        .transform(
            processed_new_data
        )
        .cache(storage=ParquetStorage(source=ctx.duck_con)) # potential Filter to be pusehd here
    )
    # FIXME: If i try to cache wide table i get a `Compilation rule for
    # `TableUnnest` operation is not defined` error 

    # this is needed to avoid the MortgageXGBoost do_explode_encode which we do
    # not have access to in this udf
    wide_table = extract_onehot_with_ibis(t, result.model.feature_names)

    new_predicted = wide_table.into_backend(ctx.con).mutate(
        prediction=udf.on_expr
    )
    
    return new_predicted

def create_xgboost_schema_wide(t, model_features) -> Dict[str, Any]:
    base_cols = [col for col in t.columns if col != 'encoded']
    onehot_features = [f for f in model_features if f not in base_cols]
    
    schema_dict = {}
    
    exclude_cols = {'loan_id', 'ever_90_delinq', 'row_id'}
    for col in base_cols:
        if col not in exclude_cols:
            col_type = t.schema()[col]
            
            if str(col_type) == 'boolean':
                schema_dict[col] = dt.boolean
            elif 'float' in str(col_type).lower():
                schema_dict[col] = dt.Float64(nullable=True) if col_type.nullable else dt.float64
            elif 'int' in str(col_type).lower():
                schema_dict[col] = dt.Int64(nullable=True) if col_type.nullable else dt.int64
            else:
                schema_dict[col] = dt.Float64(nullable=True)
    
    for feature in onehot_features:
        schema_dict[feature] = dt.Float64(nullable=True)
    
    return schema_dict


def create_xgboost_pandas_udf(
    model: xgb.Booster, 
    t,
    model_features,
    name: str = "xgboost_scorer"
):
    schema_dict = create_xgboost_schema_wide(t, model_features)
    
    @xo.udf.make_pandas_udf(
        schema=xo.schema(schema_dict),
        return_type=dt.float64,
        name=name
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
    model: xgb.Booster, 
    t,
    model_features,
    name: str = "xgboost_scorer"
):
    return create_xgboost_pandas_udf(model, t, model_features, name)


def predict_new_data_with_xgboost_udf(
    result: MLPipelineResult, 
    new_data_expr, 
    config: PipelineConfig, 
    ctx: ConnectionContext
):
    processed_new_data = pipe(
        new_data_expr,
        create_features,
        create_loan_summary,
        lambda expr: clean_features(config.features, expr)
    )
    
    fitted_onehot = result.fitted_pipeline.fitted_steps[0]
    
    t = fitted_onehot.transform(processed_new_data).cache(storage=ParquetStorage(source=ctx.duck_con))
    
    udf = xgboost_to_pandas_udf(result.model, t, result.model.feature_names)
    
    wide_table = extract_onehot_with_ibis(t, result.model.feature_names)
    
    new_predicted = wide_table.into_backend(ctx.con).mutate(
        prediction=udf.on_expr
    )
    
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
            lambda split: fit_pipeline(config, split[0], split[1])
        )
    
    return execute_pipeline


def main():
    data_root = pathlib.Path(os.getenv("DATA_ROOT", "/mnt/data/fanniemae"))
    
    config = PipelineConfig(
        data=DataConfig(data_root=data_root),
        model=evolve(ModelConfig(), num_boost_round=50, max_depth=12)
    )
    
    pipeline = create_mortgage_pipeline(config)
    result = pipeline()
    
    ctx = ConnectionContext.create()
    final_result = create_quickgrove_predictions(result, ctx)
    
    return final_result, config, ctx


def scratch():
    # transformed cache
    d = xo.deferred_read_parquet("/home/hussainsultan/.cache/xorq/parquet/letsql_cache-1d75b96f58cd6a54c2f942616087b375.parquet", con=xo.duckdb.connect())
    wide_table = extract_onehot_with_ibis(d, result.model.feature_names)
    udf = xgboost_to_quickgrove_udf(result.model)
    r = wide_table.into_backend(xo.connect()).mutate(prediction=udf.on_expr)


def example_new_data_prediction_with_xgboost_udf():
    result, config, ctx = main()
    
    new_data_expr = (
        load_data(
            evolve(
                config.data, filter_date="2001-01-01"
            ), 
            ctx
        )
        .cache(
            storage=ParquetStorage(source=ctx.con)
        )
    )
    
    new_predictions_quickgrove = predict_new_data_with_quickgrove(result, new_data_expr, config, ctx)
    
    new_predictions_xgboost = predict_new_data_with_xgboost_udf(result, new_data_expr, config, ctx)
    
    return {
        'quickgrove': new_predictions_quickgrove,
        'xgboost_udf': new_predictions_xgboost
    }
