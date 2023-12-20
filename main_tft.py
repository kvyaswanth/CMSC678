from pathlib import Path
import pickle
import warnings

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner
import numpy as np
import pandas as pd
from pandas.core.common import SettingWithCopyWarning
import torch

from pytorch_forecasting import GroupNormalizer, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.examples import get_stallion_data
from pytorch_forecasting.metrics import MAE, RMSE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting.utils import profile

warnings.simplefilter("error", category=SettingWithCopyWarning)


data = df.read_csv('all_inst.csv')

data["time_idx"] = np.arange(0,df.shape[0])
max_encoder_length = 15

training = TimeSeriesDataSet(
    data,
    time_idx="time_idx",
    target=['x','y','z'],
    group_ids=[],
    min_encoder_length=max_encoder_length // 2,  # allow encoder lengths from 0 to max_prediction_length
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=1,
    # variable_groups={"special_days": special_days},  # group of categorical variables can be treated as one variable
    time_varying_known_reals=["time_stamp"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=["x","y","z"],
    target_normalizer=GroupNormalizer(
        transformation="softplus", center=False
    ),  
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)


validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)
batch_size = 64
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)


# save datasets
training.save("t raining.pkl")
validation.save("validation.pkl")

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
lr_logger = LearningRateMonitor()
logger = TensorBoardLogger(log_graph=True)

trainer = pl.Trainer(
    max_epochs=100,
    accelerator="auto",
    gradient_clip_val=0.1,
    limit_train_batches=30,
    # val_check_interval=20,
    # limit_val_batches=1,
    # fast_dev_run=True,
    logger=logger,
    # profiler=True,
    callbacks=[lr_logger, early_stop_callback],
)


tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,
    loss=QuantileLoss(),
    log_interval=10,
    log_val_interval=1,
    reduce_on_plateau_patience=3,
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# # find optimal learning rate
# # remove logging and artificial epoch size
# tft.hparams.log_interval = -1
# tft.hparams.log_val_interval = -1
# trainer.limit_train_batches = 1.0
# # run learning rate finder
# res = Tuner(trainer).lr_find(
#     tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, min_lr=1e-5, max_lr=1e2
# )
# print(f"suggested learning rate: {res.suggestion()}")
# fig = res.plot(show=True, suggest=True)
# fig.show()
# tft.hparams.learning_rate = res.suggestion()

# trainer.fit(
#     tft,
#     train_dataloaders=train_dataloader,
#     val_dataloaders=val_dataloader,
# )

# # make a prediction on entire validation set
# preds, index = tft.predict(val_dataloader, return_index=True, fast_dev_run=True)


# tune
study = optimize_hyperparameters(
    train_dataloader,
    val_dataloader,
    model_path="optuna_test",
    n_trials=200,
    max_epochs=50,
    gradient_clip_val_range=(0.01, 1.0),
    hidden_size_range=(8, 128),
    hidden_continuous_size_range=(8, 128),
    attention_head_size_range=(1, 4),
    learning_rate_range=(0.001, 0.1),
    dropout_range=(0.1, 0.3),
    trainer_kwargs=dict(limit_train_batches=30),
    reduce_on_plateau_patience=4,
    use_learning_rate_finder=False,
)
with open("test_study.pkl", "wb") as fout:
    pickle.dump(study, fout)


# profile speed
# profile(
#     trainer.fit,
#     profile_fname="profile.prof",
#     model=tft,
#     period=0.001,
#     filter="pytorch_forecasting",
#     train_dataloaders=train_dataloader,
#     val_dataloaders=val_dataloader,
# )
