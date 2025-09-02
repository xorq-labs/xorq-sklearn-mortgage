import os
import pathlib
from typing import Tuple, Dict, Any

import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder
from attrs import frozen, field

import threading
from codetiming import Timer

import xorq.api as xo
import xorq.expr.datatypes as dt
from xorq.expr.ml.fit_lib import transform_sklearn_feature_names_out
from xorq.expr.ml.pipeline_lib import Step
from xorq.expr.ml.structer import ENCODED


@frozen
class ConnectionContext:
    con: Any = field()
    duck_con: Any = field()

    @classmethod
    def create(cls) -> "ConnectionContext":
        return cls(con=xo.connect(), duck_con=xo.duckdb.connect())


@frozen
class DataConfig:
    data_root: pathlib.Path = field(converter=pathlib.Path, default=os.getenv("DATA_ROOT", "/mnt/data/fanniemae"))
    perf_rel_path: str = field(default="data/perf/perf.parquet")
    acq_rel_path: str = field(default="data/acq/acq.parquet")
    filter_date: str = field(
        default="2001-01-01"
    )  # data that works without ArrowInvalid: offset overflow while concatenating arrays

    @property
    def perf_path(self):
        return self.data_root.joinpath(self.perf_rel_path)

    @property
    def acq_path(self):
        return self.data_root.joinpath(self.acq_rel_path)

    def load_data(self, ctx: ConnectionContext):
        perf_expr = xo.deferred_read_parquet(
            self.perf_path, ctx.duck_con, "perf_raw"
        )
        acq_expr = xo.deferred_read_parquet(
            self.acq_path, ctx.duck_con, "acq_raw"
        )
        expr = acq_expr.join(
            perf_expr, acq_expr.loan_id == perf_expr.loan_id, how="left"
        ).filter(xo._.monthly_reporting_period <= self.filter_date)
        return expr


load_data = DataConfig.load_data


@frozen
class FeatureConfig:
    numeric_features: Tuple[str, ...] = field(
        default=(
            "orig_rate",
            "orig_ltv",
            "dti",
            "credit_score",
            "orig_upb",
            "current_balance_ratio",
        )
    )
    categorical_features: Tuple[str, ...] = field(
        default=("property_state", "loan_purpose", "property_type")
    )
    flag_features: Tuple[str, ...] = field(
        default=("high_ltv_flag", "high_dti_flag", "subprime_flag")
    )
    target_col: str = field(default="ever_90_delinq")

    @property
    def all_features(self) -> Tuple[str, ...]:
        return self.numeric_features + self.categorical_features + self.flag_features


@frozen
class ModelConfig:
    num_boost_round: int = field(default=100)
    max_depth: int = field(default=6)
    eta: float = field(default=0.1)
    objective: str = field(default="binary:logistic")
    eval_metric: str = field(default="logloss")
    seed: int = field(default=42)
    prediction_threshold: float = field(default=0.5)

    @property
    def params(self) -> Dict[str, Any]:
        return {
            "max_depth": self.max_depth,
            "eta": self.eta,
            "objective": self.objective,
            "eval_metric": self.eval_metric,
            "seed": self.seed,
        }


@frozen
class SplitConfig:
    test_size: float = field(default=0.5)
    random_seed: int = field(default=42)

    @property
    def split_kwargs(self):
        return {
            "random_seed": self.random_seed,
            "test_sizes": self.test_size,
        }


@frozen
class PipelineConfig:
    data: DataConfig = field(factory=DataConfig)
    features: FeatureConfig = field(factory=FeatureConfig)
    model: ModelConfig = field(factory=ModelConfig)
    split: SplitConfig = field(factory=SplitConfig)


class OneHotHelper(OneHotEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_names_out_ = None

    def transform(self, *args, **kwargs):
        return transform_sklearn_feature_names_out(super(), *args, **kwargs)

    def fit(self, X, y=None):
        result = super().fit(X, y)
        if hasattr(self, "get_feature_names_out"):
            self.feature_names_out_ = self.get_feature_names_out()
        return result

    @classmethod
    def get_step_f_kwargs(cls, kwargs):
        from xorq.expr.ml.fit_lib import deferred_fit_transform_sklearn

        return (
            deferred_fit_transform_sklearn,
            kwargs
            | {
                "return_type": dt.Array(dt.Struct({"key": str, "value": float})),
                "target": None,
            },
        )

    @classmethod
    def get_step(cls, name="one_hot_step", params_tuple=(("handle_unknown", "ignore"), ("drop", "first"))):
        return Step(
            cls,
            name,
            params_tuple=params_tuple,
        )


class MortgageXGBoost(BaseEstimator):
    def __init__(self, num_boost_round=100, encoded_col=ENCODED, **params):
        self.num_boost_round = num_boost_round
        self.encoded_col = encoded_col
        self.params = {
            "max_depth": params.get("max_depth", 6),
            "eta": params.get("eta", 0.1),
            "objective": params.get("objective", "binary:logistic"),
            "eval_metric": params.get("eval_metric", "logloss"),
            "seed": params.get("seed", 42),
        }
        self.model = None

    return_type = dt.float64

    def explode_encoded(self, X):
        def make_df_apply(series):
            df = (
                series
                .apply(
                    lambda lst: pd.Series({d["key"]: d["value"] for d in lst})
                )
            )
            return df

        def make_df_expensive(series):
            (keys, values) = (
                [
                    tuple(dct[which] for dct in lst)
                    for lst in series
                ]
                for which in ("key", "value")
            )
            (columns, *rest) = keys
            assert all(el == columns for el in rest)
            df = pd.DataFrame(
                values,
                index=X.index,
                columns=columns,
            )
            return df

        def make_df_cheap(series):
            values = [
                [dct["value"] for dct in lst]
                for lst in series
            ]
            columns = [dct["key"] for dct in series.iloc[0]]
            df = pd.DataFrame(
                values,
                index=series.index,
                columns=columns,
            )
            return df

        f = make_df_expensive
        with Timer(f"exlode_encoded-{f.__name__}-{threading.current_thread().native_id}", logger=None):
            return X.drop(columns=self.encoded_col).join(f(X[self.encoded_col]))

    def fit(self, X, y):
        X_exploded = self.explode_encoded(X)
        with Timer(f"MortgageXGBoost.fit-{threading.current_thread().native_id}", logger=None):
            dtrain = xgb.DMatrix(X_exploded, y)
            self.model = xgb.train(self.params, dtrain, self.num_boost_round)
            return self

    def predict(self, X):
        X_exploded = self.explode_encoded(X)
        with Timer(f"MortgageXGBoost.predict-{threading.current_thread().native_id}", logger=None):
            return self.model.predict(xgb.DMatrix(X_exploded))

    @classmethod
    def get_step(cls, config, name="xgboost_model"):
        params_tuple=(
            ("num_boost_round", config.model.num_boost_round),
            ("encoded_col", ENCODED),
            ("max_depth", config.model.max_depth),
            ("eta", config.model.eta),
            ("objective", config.model.objective),
            ("eval_metric", config.model.eval_metric),
            ("seed", config.model.seed),
        )
        return Step(
            cls,
            name,
            params_tuple=params_tuple,
        )
