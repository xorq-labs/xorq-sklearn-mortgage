import toolz
from attr import evolve
from codetiming import Timer

import xorq.api as xo
from expr import (
    ConnectionContext,
    DataConfig,
    ModelConfig,
    PipelineConfig,
    clean_features,
    create_features,
    create_loan_summary,
    create_mortgage_pipeline,
)
from xorq.caching import (
    ParquetStorage,
    SourceStorage,
)


config = PipelineConfig(
    data=DataConfig(),
    model=evolve(ModelConfig(), num_boost_round=50, max_depth=12),
)
ctx = ConnectionContext.create()
new_data_expr = (
    config.data.load_data(ctx)
    .cache(
        storage=ParquetStorage(source=ctx.con)
    )
)
processed_new_data = (
    toolz.pipe(
        new_data_expr,
        create_features,
        create_loan_summary,
        clean_features(config.features)
    )
    # .cache(SourceStorage(source=xo.connect()))
    .cache(ParquetStorage(source=ctx.con))
)
result = create_mortgage_pipeline(config)()
new_predictions = (
    processed_new_data
    # .limit(100_000)
    .pipe(result.fitted_pipeline.predict)
    .mutate(
        predicted_class=(xo._.predicted_prob >= config.model.prediction_threshold)
        .cast("int")
    )
)


with Timer("outer loop", logger=None):
    x = new_predictions.execute()
dct = dict(Timer.timers)
sums = toolz.valmap(lambda tpl: sum(v for k, v in tpl), toolz.groupby(lambda kv: kv[0].rsplit("-", 1)[0], dct.items()))
print(sums)
