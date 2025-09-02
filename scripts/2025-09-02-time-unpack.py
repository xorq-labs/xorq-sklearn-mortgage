from codetiming import Timer

from expr import *
from xorq.caching import SourceStorage


config = PipelineConfig(
    data=DataConfig(data_root=pathlib.Path(os.getenv("DATA_ROOT", "/mnt/data/fanniemae"))),
    model=evolve(ModelConfig(), num_boost_round=50, max_depth=12),
)
ctx = ConnectionContext.create()
new_data_expr = (
    load_data(evolve(config.data, filter_date="2001-01-01"), ctx)
    .cache(
        storage=ParquetStorage(source=ctx.con)
    )
)
processed_new_data = (
    pipe(
        new_data_expr,
        create_features,
        create_loan_summary,
        clean_features(config.features)
    )
    .cache(SourceStorage(source=xo.connect()))
)
result = create_mortgage_pipeline(config)()
new_predictions = (
    result.fitted_pipeline.predict(processed_new_data.limit(10_000))
    .mutate(
        predicted_class=(xo._.predicted_prob >= config.model.prediction_threshold)
        .cast("int")
    )
)
with Timer("outer loop", logger=None):
    x = new_predictions.execute()
print(Timer.timers)
