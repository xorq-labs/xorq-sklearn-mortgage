import pathlib
import os
import time
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score

import xorq.api as xo
from xorq.api import selectors as s
import xorq.expr.datatypes as dt
from xorq.expr.ml import train_test_splits
from xorq.expr.ml.fit_lib import transform_sklearn_feature_names_out
from xorq.expr.ml.pipeline_lib import FittedPipeline, Step
from xorq.expr.ml.structer import ENCODED
from xorq.caching import ParquetStorage

# Setup paths and connection
data_root = pathlib.Path(os.getenv("DATA_ROOT", "/mnt/data/fanniemae"))
con = xo.connect()

duck_con = xo.duckdb.connect()

# Load and join data
perf_expr = xo.deferred_read_parquet(str(data_root / "data" / "perf/perf.parquet"), duck_con, "perf_raw")
acq_expr = xo.deferred_read_parquet(str(data_root / "data" / "acq/acq.parquet"), duck_con, "acq_raw")
joined_expr = acq_expr.join(perf_expr,acq_expr.loan_id==perf_expr.loan_id, how="left").filter(xo._.monthly_reporting_period<="2001-01-01")

dup_cols = [c for c in joined_expr.columns if c != 'loan_id' and c.startswith('loan_id')]
if dup_cols:
    joined_expr = joined_expr.drop(*dup_cols)


cache_storage = ParquetStorage(source=con)



class OneHotStep(OneHotEncoder):
    def transform(self, *args, **kwargs):
        return transform_sklearn_feature_names_out(super(), *args, **kwargs)
    
    @classmethod
    def get_step_f_kwargs(cls, kwargs):
        from xorq.expr.ml.fit_lib import deferred_fit_transform_sklearn
        return (deferred_fit_transform_sklearn, kwargs | {
            "return_type": dt.Array(dt.Struct({"key": str, "value": float})),
            "target": None,
        })


class MortgageXGBoost(BaseEstimator):
    def __init__(self, num_boost_round=100, encoded_col=ENCODED):
        self.encoded_col = encoded_col
        self.num_boost_round = num_boost_round
        self.params = {
            'max_depth': 6,
            'eta': 0.1, 
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'seed': 42
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


def create_features(expr):
    """Simplified feature engineering"""
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
    """Create loan-level summary with key features"""
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


featured_data = create_features(joined_expr)

loan_summary = create_loan_summary(featured_data)

target_col = 'ever_90_delinq'

numeric_features = ['orig_rate', 'orig_ltv', 'dti', 'credit_score', 'orig_upb', 
                   'current_balance_ratio']
categorical_features = ['property_state', 'loan_purpose', 'property_type']
flag_features = ['high_ltv_flag', 'high_dti_flag', 'subprime_flag']
available_numeric = [f for f in numeric_features if f in loan_summary.columns]
available_categorical = [f for f in categorical_features if f in loan_summary.columns]
available_flags = [f for f in flag_features if f in loan_summary.columns]

all_features = available_numeric + available_categorical + available_flags

mutate_exprs = []

for col in available_numeric:
    if col in ['orig_ltv', 'dti', 'credit_score', 'orig_upb']:
        mutate_exprs.append(getattr(xo._, col).fill_null(0).name(col))
    else:
        mutate_exprs.append(getattr(xo._, col).fill_null(0.0).name(col))

for col in available_categorical:
    mutate_exprs.append(getattr(xo._, col).fill_null('Unknown').name(col))

for col in available_flags:
    mutate_exprs.append(getattr(xo._, col).fill_null(False).name(col))

mutate_exprs.append(xo._[target_col].fill_null(0).name(target_col))

ml_data = loan_summary.select(['loan_id'] + all_features + [target_col]).mutate(mutate_exprs)

ml_with_row = ml_data.mutate(row_id=xo.row_number())
train_expr, test_expr = train_test_splits(
    ml_with_row, 'row_id', test_sizes=0.5, random_seed=42
)
train_clean = train_expr.drop('row_id').cache(storage=ParquetStorage(source=con))
test_clean = (test_expr.drop('row_id')).cache(storage=ParquetStorage(source=con))

train_clean = (
        train_clean
            .mutate(
                loan_purpose=xo._.loan_purpose.cast(dt.LargeString), 
                property_type=xo._.property_type.cast(dt.LargeString),
                property_state=xo._.property_state.cast(dt.LargeString),
            )
        )

test_clean = (
        test_clean 
            .mutate(
                loan_purpose=xo._.loan_purpose.cast(dt.LargeString), 
                property_type=xo._.property_type.cast(dt.LargeString),
                property_state=xo._.property_state.cast(dt.LargeString),
            )
        )

one_hot_step = Step(
    OneHotStep,
    "one_hot_encoder", 
    params_tuple=(("handle_unknown", "ignore"), ("drop", "first"))
)

xgb_step = Step(
    MortgageXGBoost,
    "xgboost_model",
    params_tuple=(("encoded_col", ENCODED),)
)

# Fit pipeline
fitted_onehot = one_hot_step.fit(
    train_clean,
    features=available_categorical,
    dest_col=ENCODED
)

fitted_xgb = xgb_step.fit(
    expr=train_clean.mutate(fitted_onehot.mutate),
    features=available_numeric + available_flags + [ENCODED],
    target=target_col,
    dest_col='predicted_prob'
)

pipeline = FittedPipeline((fitted_onehot, fitted_xgb), train_clean)

# Make predictions
predictions = pipeline.predict(test_clean).mutate(
    predicted_class=(xo._.predicted_prob >= 0.5).cast('int')
)


loan_count = loan_summary.count().execute()
pred_df = predictions.execute()
y_true = pred_df[target_col]
y_pred = pred_df['predicted_class']
y_prob = pred_df['predicted_prob']

auc = roc_auc_score(y_true, y_prob)
print(f"   AUC Score: {auc:.4f}")

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
print(f"   Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
    
