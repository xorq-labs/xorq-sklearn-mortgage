import functools
import os
import pathlib

import xorq.api as xo
from attrs import evolve, field, frozen
from toolz import curry, pipe
from xorq.caching import ParquetStorage
from xorq.expr.ml import train_test_splits
from xorq.expr.ml.pipeline_lib import FittedPipeline
from xorq.expr.ml.structer import ENCODED
import xorq_sklearn_mortgage.constants as C
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
from xorq_sklearn_mortgage.quickgrove_lib import (
    mortgage_xgboost_to_quickgrove_udf,
)


@frozen
class MLPipelineResult:
    config = field()
    ctx = field(init=False, factory=ConnectionContext.create)

    @property
    @functools.cache
    def storage(self):
        return ParquetStorage(source=self.ctx.con)

    @property
    @functools.cache
    def raw_expr(self):
        return self.config.data.load_data(self.ctx)

    def process_expr(self, expr):
        funcs = (
            create_features,
            create_loan_summary,
            clean_features(self.config.features),
        )
        return pipe(
            expr,
            *funcs,
        )

    @property
    @functools.cache
    def train_test_split(self):
        (train_expr, test_expr) = (
            expr.drop("row_id").cache(storage=self.storage)
            for expr in train_test_splits(
                self.process_expr(self.raw_expr).mutate(row_id=xo.row_number()),
                "row_id",
                **self.config.split.split_kwargs,
            )
        )
        return train_expr, test_expr

    @property
    def train_expr(self):
        return self.train_test_split[0]

    @property
    def test_expr(self):
        return self.train_test_split[1]

    @property
    def fitted_pipeline(self):
        fitted_onehot = OneHotHelper.get_step().fit(
            self.train_expr,
            features=self.config.features.categorical_features,
            dest_col=ENCODED,
            storage=self.storage,
        )

        fitted_xgb = MortgageXGBoost.get_step(self.config).fit(
            expr=self.train_expr.mutate(fitted_onehot.mutate),
            features=list(self.config.features.numeric_features)
            + list(self.config.features.flag_features)
            + [ENCODED],
            target=self.config.features.target_col,
            dest_col="predicted_prob",
            storage=self.storage,
        )
        fitted_pipeline = FittedPipeline((fitted_onehot, fitted_xgb), self.train_expr)
        return fitted_pipeline

    @property
    def predictions(self):
        predictions = self.fitted_pipeline.predict(self.test_expr).mutate(
            predicted_class=(xo._.predicted_prob >= self.config.model.prediction_threshold).cast(
                "int"
            )
        )
        return predictions

    @property
    def model(self):
        return self.fitted_pipeline.predict_step.model

    @property
    def deferred_model(self):
        raise NotImplementedError


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


def main():
    config = PipelineConfig(
        data=DataConfig(data_root=pathlib.Path(os.getenv("DATA_ROOT", "/mnt/data/fanniemae"))),
        model=evolve(ModelConfig(), num_boost_round=50, max_depth=12),
    )
    result = MLPipelineResult(config)
    ctx = result.ctx
    return result, config, ctx


def example_new_data_prediction_with_xgboost_udf():
    result, config, ctx = main()

    new_data_expr = load_data(evolve(config.data, filter_date=C.filter_date), ctx).cache(
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
