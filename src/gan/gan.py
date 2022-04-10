import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim.lr_scheduler import ConstantLR, LinearLR, SequentialLR
from torchvision.utils import make_grid

from src.gan.discriminator import MultiScaleDiscriminator
from src.gan.generator import Generator
from src.gan.vgg import VGG19

VGG_MODEL = VGG19()  # so that it is not saved to checkpoint
VGG_MODEL.to("cuda" if torch.cuda.is_available() else "cpu")  # kind of a hack


class PhotobashGAN(pl.LightningModule):

    def __init__(
            self,
            base_channels=32, latent_dim=256,
            epochs=200, lr=0.0002, betas=(0.0, 0.999), lamb=10, lamb_ds=0.0
    ):
        super().__init__()
        self.save_hyperparameters()

        self.gen = Generator(latent_dim, base_channels)
        self.dis = MultiScaleDiscriminator(base_channels)

        self.augment_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Resize(size=286, interpolation=T.InterpolationMode.NEAREST),
            T.RandomCrop(size=256)
        ])

        self.train_samples = torch.zeros((3, 1, 1))  # dummy

    def training_step(self, batch, batch_idx, optimizer_idx):

        # random augmentation
        combined = torch.concat(batch, dim=0)
        combined = self.augment_transform(combined)
        i = combined.shape[0] // 2
        real_images, cmaps = combined[:i, ...], combined[i:, ...]

        # train generator
        if optimizer_idx == 0:
            fake_images, loss = self._gen_train_step(real_images, cmaps)
            if batch_idx == 0:
                image_stack = list(zip(real_images, cmaps, fake_images))
                image_stack = image_stack[:10]
                image_stack = torch.stack(list(sum(image_stack, ())), dim=0)
                self.train_samples = make_grid(image_stack, 6, normalize=True, value_range=(-1, 1))

        # train discriminator
        else:
            loss = self._dis_train_step(real_images, cmaps)

        return loss

    def validation_step(self, batch, batch_idx):
        real_images, cmaps = batch
        fake_images = self._sample_images(cmaps)
        image_stack = torch.concat([real_images, cmaps, fake_images], dim=0)
        return image_stack

    def validation_epoch_end(self, outputs):
        image_gallery = torch.concat(outputs, dim=0)
        grid = make_grid(image_gallery, 6, normalize=True, value_range=(-1, 1))
        self.logger.log_image(key="val_samples", images=[grid], step=self.current_epoch)

        grid = self.train_samples
        self.logger.log_image(key="train_samples", images=[grid], step=self.current_epoch)

    def configure_optimizers(self):
        epochs = self.hparams.epochs
        lr_g = self.hparams.lr / 2
        lr_d = self.hparams.lr * 2
        betas = self.hparams.betas

        opt_g = torch.optim.Adam(self.gen.parameters(), lr=lr_g, betas=betas)
        opt_d = torch.optim.Adam(self.dis.parameters(), lr=lr_d, betas=betas)
        opts = [opt_g, opt_d]

        schedules = []
        for opt in opts:
            const = ConstantLR(opt, factor=1.0, total_iters=(epochs // 2 + 1))
            decay = LinearLR(opt, start_factor=1.0, end_factor=0.0, total_iters=(epochs // 2))
            chained = SequentialLR(opt, schedulers=[const, decay], milestones=[epochs // 2])
            schedules.append(chained)

        return opts, schedules

    def _sample_images(self, cmaps, return_noise=False):
        batch_size = cmaps.shape[0]
        noise = torch.randn((batch_size, self.hparams.latent_dim), device=cmaps.device)
        fake_images = self.gen(noise, cmaps)
        return (fake_images, noise) if return_noise else fake_images

    def _gen_train_step(self, real_images, cmaps):
        fake_images, noise = self._sample_images(cmaps, return_noise=True)

        multi_real_feats = self.dis(real_images, cmaps)
        multi_fake_feats = self.dis(fake_images, cmaps)

        gen_loss = torch.tensor(0.0, device=cmaps.device)
        feat_loss = torch.tensor(0.0, device=cmaps.device)
        vgg_loss = torch.tensor(0.0, device=cmaps.device)

        for real_feats, fake_feats in zip(multi_real_feats, multi_fake_feats):
            fake_logits = fake_feats[-1]
            gen_loss += -torch.mean(fake_logits)

            for f_real, f_fake in zip(real_feats[:-1], fake_feats[:-1]):
                feat_loss += F.l1_loss(f_fake, f_real.detach())

        vgg_real_feats = VGG_MODEL(real_images)
        vgg_fake_feats = VGG_MODEL(fake_images)
        weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        for f_real, f_fake, w in zip(vgg_real_feats, vgg_fake_feats, weights):
            vgg_loss += w * F.l1_loss(f_fake, f_real.detach())

        if self.hparams.lamb_ds > 0:
            with torch.no_grad():
                fake_images_ds, noise_ds = self._sample_images(cmaps, return_noise=True)
            fake_image_diff = torch.abs(fake_images - fake_images_ds.detach()).mean(dim=(1, 2, 3))
            noise_diff = torch.abs(noise.detach() - noise_ds.detach()).mean(dim=1)
            ds_loss = -torch.mean(fake_image_diff / (noise_diff + 1e-5))
        else:
            ds_loss = 0.0

        gen_loss = gen_loss / self.dis.n_subunits
        feat_loss = feat_loss / self.dis.n_subunits
        loss = gen_loss + self.hparams.lamb * (feat_loss + vgg_loss) + self.hparams.lamb_ds * ds_loss

        batch_size = real_images.shape[0]
        self.log("gen_loss", gen_loss, batch_size=batch_size)
        self.log("feat_loss", feat_loss, batch_size=batch_size)
        self.log("vgg_loss", vgg_loss, batch_size=batch_size)
        self.log("ds_loss", ds_loss, batch_size=batch_size)

        return fake_images, loss

    def _dis_train_step(self, real_images, cmaps):
        with torch.no_grad():
            fake_images = self._sample_images(cmaps)

        multi_real_feats = self.dis(real_images, cmaps)
        multi_fake_feats = self.dis(fake_images, cmaps)

        real_loss = torch.tensor(0.0, device=cmaps.device)
        fake_loss = torch.tensor(0.0, device=cmaps.device)

        for real_feats, fake_feats in zip(multi_real_feats, multi_fake_feats):
            real_logits = real_feats[-1]
            fake_logits = fake_feats[-1]

            zero = torch.zeros_like(real_logits)
            real_loss += -torch.mean(torch.minimum(real_logits - 1, zero))
            fake_loss += -torch.mean(torch.minimum(-fake_logits - 1, zero))

        real_loss = real_loss / self.dis.n_subunits
        fake_loss = fake_loss / self.dis.n_subunits
        loss = real_loss + fake_loss

        batch_size = real_images.shape[0]
        self.log("real_loss", real_loss, batch_size=batch_size)
        self.log("fake_loss", fake_loss, batch_size=batch_size)
        self.log("dis_loss", loss, batch_size=batch_size)

        return loss
