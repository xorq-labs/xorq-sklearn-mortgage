import warnings
from typing import Any, Callable, Dict, List

import pandas as pd
import xorq.expr.datatypes as dt
import xorq.vendor.ibis.expr.types as ir
from xorq.expr.ml.quickgrove_lib import (
    SUPPORTED_TYPES,
    UDFWrapper,
    _all_predicates_are_features,
    _create_udf_function,
    _create_udf_node,
    _validate_model_features,
    collect_predicates,
)
from xorq.expr.ml.structer import ENCODED
from xorq.vendor.ibis.common.annotations import Argument
from xorq.vendor.ibis.common.patterns import pattern, replace
from xorq.vendor.ibis.expr import operations as ops
from xorq.vendor.ibis.expr.rules import ValueOf
from xorq.vendor.ibis.util import Namespace

from xorq_sklearn_mortgage.quickgrove_lib import make_df_cheap

p = Namespace(pattern, module=ops)


def _validate_model_features(
    model: "PyGradientBoostedDecisionTrees", supported_types: Dict
) -> None:
    schema = [model.feature_names[i] for i in sorted(model.required_features)]
    feature_types = [model.feature_types[i] for i in sorted(model.required_features)]
    unsupported = [
        f"{name}: {type_}"
        for name, type_ in zip(schema, feature_types)
        if type_ not in supported_types
    ]
    if unsupported:
        raise ValueError(f"Unsupported feature types: {', '.join(unsupported)}")


def make_pruned_udf(
    original_udf,
    predicates: List[dict],
) -> UDFWrapper:
    from quickgrove import Feature

    model = original_udf.model
    pred_feature_names = {pred["column"] for pred in predicates}
    model_feature_names = set(model.feature_names)

    if not pred_feature_names.issubset(model_feature_names):
        warnings.warn(
            "Feature not found in predicates, skipping pruning...", UserWarning
        )
        return original_udf

    pruned_model = model.prune(
        [
            Feature(pred["column"]) < pred["value"]
            if pred["op"] == "Less"
            else Feature(pred["column"]) >= pred["value"]
            if pred["op"] == "Greater"
            else Feature(pred["column"]) >= pred["value"]
            for pred in predicates
        ]
    )

    _validate_model_features(pruned_model, SUPPORTED_TYPES)
    required_features = sorted(pruned_model.required_features)
    schema = [pruned_model.feature_names[i] for i in required_features]

    # Handle ENCODED column structure
    fields = {}

    # Get required features from the UDF function's metadata
    # This should be set when the UDF is created
    if hasattr(original_udf, "_required_features"):
        required_features_list = original_udf._required_features
    else:
        # Fallback: assume all base features + ENCODED
        # This is a temporary fallback until we modify UDF creation
        base_features = [
            name
            for name in model.feature_names
            if name != ENCODED
            and not any(
                name.startswith(prefix + "_")
                for prefix in ["property_state", "loan_purpose", "property_type"]
            )
        ]
        required_features_list = base_features + [ENCODED]

    for name in required_features_list:
        if name != ENCODED:
            fields[name] = Argument(
                pattern=ValueOf(dt.float64),
                typehint=dt.float64,
            )

    fields[ENCODED] = Argument(
        pattern=ValueOf(dt.Array(dt.Struct({"key": dt.string, "value": dt.float64}))),
        typehint=dt.Array(dt.Struct({"key": dt.string, "value": dt.float64})),
    )

    def fn_from_arrays(*arrays):
        if len(arrays) != len(fields):
            raise ValueError(f"Expected {len(fields)} arrays, got {len(arrays)}")

        encoded_array = arrays[-1]
        base_arrays = arrays[:-1]

        base_feature_names = [name for name in fields.keys() if name != ENCODED]
        base_df = pd.DataFrame(
            {feature: array for feature, array in zip(base_feature_names, base_arrays)}
        )

        encoded_series = pd.Series(encoded_array)
        onehot_df = make_df_cheap(encoded_series)

        combined_df = pd.concat([base_df, onehot_df], axis=1)

        feature_matrix = pd.DataFrame()
        for feature in schema:
            if feature in combined_df.columns:
                feature_matrix[feature] = combined_df[feature].fillna(0.0)
            else:
                feature_matrix[feature] = 0.0

        # Use pruned model for prediction
        feature_arrays = [feature_matrix[feature].values for feature in schema]
        return pruned_model.predict_arrays(feature_arrays)

    udf_func = _create_udf_function(
        _create_udf_node(
            model=pruned_model,
            fn_from_arrays=fn_from_arrays,
            fields=fields,
            udf_name=getattr(original_udf.name, "__name__", "udf") + "_pruned",
        )
    )

    return UDFWrapper(
        udf_func,
        pruned_model,
        getattr(original_udf, "model_path", "temp"),
        required_features=list(fields.keys()),
    )


@replace(p.Filter(p.Project))
def prune_quickgrove_model(_, **kwargs):
    parent_op = _.parent
    predicates = collect_predicates(_)
    if not predicates:
        return _

    new_values = {}
    pruned_udf_wrapper = None

    for name, value in parent_op.values.items():
        if isinstance(value, ops.ScalarUDF) and hasattr(value, "model"):
            if not _all_predicates_are_features(_, value.model):
                return _

            pruned_udf_wrapper = make_pruned_udf(value, predicates)
            required_features = pruned_udf_wrapper.required_features

            udf_kwargs = {
                feat_name: parent_op.values[feat_name]
                for feat_name in required_features
                if feat_name in parent_op.values
            }

            if callable(pruned_udf_wrapper):
                new_values[name] = pruned_udf_wrapper(**udf_kwargs)
            else:
                new_values[name] = value
        else:
            new_values[name] = value

    if not pruned_udf_wrapper:
        return _

    new_project = ops.Project(parent_op.parent, new_values)

    subs = {
        ops.Field(parent_op, k): ops.Field(new_project, k) for k in parent_op.values
    }
    new_predicates = tuple(p.replace(subs, filter=ops.Value) for p in _.predicates)
    return ops.Filter(parent=new_project, predicates=new_predicates)


def rewrite_quickgrove_expr(expr) -> ir.Table:
    op = expr.op()
    new_op = op.replace(prune_quickgrove_model)
    return new_op.to_expr()
