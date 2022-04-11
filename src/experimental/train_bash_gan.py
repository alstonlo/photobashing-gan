import argparse
import os
import pathlib

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.experimental.datamodule import ColorMapDatamodule
from src.gan.gan import PhotobashGAN

os.environ["WANDB_START_METHOD"] = "thread"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=str, default=420)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_samples", type=int, default=14)
    parser.add_argument("--subsample", type=int, default=-1)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--lamb", type=float, default=10.0)
    parser.add_argument("--lamb_ds", type=float, default=0.0)
    args = parser.parse_args()

    # construct datamodule
    datamodule = ColorMapDatamodule(
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_samples=args.val_samples,
        subsample=args.subsample
    )
    args.data_size = len(datamodule.train)

    # construct model
    pl.seed_everything(seed=args.seed)
    gan = PhotobashGAN(epochs=args.epochs, lr=args.lr, lamb=args.lamb, lamb_ds=args.lamb_ds)

    # logging
    results_dir = pathlib.Path(__file__).parents[2] / "results"
    logger = WandbLogger(project="photobash_gan", log_model=True, save_dir=str(results_dir))
    logger.experiment.config.update(vars(args))

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # checkpointing
    checkpointing = ModelCheckpoint(monitor="epoch", save_top_k=1, every_n_epochs=1, mode="max")

    # training
    trainer = pl.Trainer(
        callbacks=[checkpointing, lr_monitor],
        enable_progress_bar=False,
        gpus=(1 if torch.cuda.is_available() else 0),
        logger=logger,
        log_every_n_steps=25,
        min_epochs=args.epochs,
        max_epochs=args.epochs,
        check_val_every_n_epoch=1,
        track_grad_norm=2
    )

    pl.seed_everything(seed=args.seed)
    trainer.fit(model=gan, datamodule=datamodule)


if __name__ == "__main__":
    main()
