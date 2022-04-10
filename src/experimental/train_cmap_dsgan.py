import argparse
import os
import pathlib

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.experimental.datamodule import CmapDatamodule
from src.gan.gan import PhotobashingDSGAN

os.environ["WANDB_START_METHOD"] = "thread"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=str, default=420)
    parser.add_argument("--batch_size", type=int, default=28)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--val_samples", type=int, default=10)
    parser.add_argument("--subsample", type=int, default=-1)

    parser.add_argument("--epochs", type=int, default=175)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--lamb", type=float, default=10.0)
    parser.add_argument("--lamb_ds", type=float, default=15.0)
    args = parser.parse_args()

    # construct datamodule
    datamodule = CmapDatamodule(
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_samples=args.val_samples,
        subsample=args.subsample
    )
    args.data_size = len(datamodule.train)

    # construct model
    pl.seed_everything(seed=args.seed)
    gan = PhotobashingDSGAN(epochs=args.epochs, lr=args.lr, lamb=args.lamb, lamb_ds=args.lamb_ds)

    # logging
    results_dir = pathlib.Path(__file__).parents[2] / "results"
    logger = WandbLogger(project="cmap_dsgan", log_model="all", save_dir=str(results_dir))
    logger.experiment.config.update(vars(args))

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # checkpointing
    checkpointing = ModelCheckpoint(monitor="epoch", save_top_k=3, every_n_epochs=3, mode="max")

    # training
    trainer = pl.Trainer(
        callbacks=[checkpointing, lr_monitor],
        enable_progress_bar=False,
        gpus=(1 if torch.cuda.is_available() else 0),
        logger=logger,
        log_every_n_steps=50,
        min_epochs=args.epochs,
        max_epochs=args.epochs,
        check_val_every_n_epoch=1,
        track_grad_norm=2
    )

    pl.seed_everything(seed=args.seed)
    trainer.fit(model=gan, datamodule=datamodule)


if __name__ == "__main__":
    main()
