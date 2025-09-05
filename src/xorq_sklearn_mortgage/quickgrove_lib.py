import os
import tempfile
from pathlib import Path
from typing import List, Union

import dask
import pandas as pd
import xorq.expr.datatypes as dt
from quickgrove import PyGradientBoostedDecisionTrees
from xorq.expr.ml.quickgrove_lib import UDFWrapper
from xorq.expr.ml.structer import ENCODED
from xorq.vendor.ibis.common.annotations import Argument
from xorq.vendor.ibis.expr.rules import ValueOf


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


@dask.base.normalize_token.register(PyGradientBoostedDecisionTrees)
def normalize_token_PyGradientBoostedDecisionTrees(obj):
    import itertools
    import toolz

    f = toolz.excepts(Exception, obj.tree_info)
    gen = itertools.takewhile(bool, (f(i) for i in itertools.count()))
    return dask.base.tokenize(tuple(str(el) for el in gen))
