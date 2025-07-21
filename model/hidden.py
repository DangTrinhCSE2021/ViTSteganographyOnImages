import numpy as np
import torch
import torch.nn as nn

from options import HiDDenConfiguration
from model.encoder_decoder import EncoderDecoder
from vgg_loss import VGGLoss
from noise_layers.noiser import Noiser
from model.discriminator import Discriminator


class Hidden:
    def __init__(self, configuration: HiDDenConfiguration, device: torch.device, noiser: Noiser, tb_logger):
        super(Hidden, self).__init__()

        self.encoder_decoder = EncoderDecoder(configuration, noiser).to(device)
        self.optimizer_enc_dec = torch.optim.Adam(self.encoder_decoder.parameters())

        # Add discriminator and its optimizer
        self.discriminator = Discriminator(configuration).to(device)
        self.optimizer_discrim = torch.optim.Adam(self.discriminator.parameters())

        if configuration.use_vgg:
            self.vgg_loss = VGGLoss(3, 1, False)
            self.vgg_loss.to(device)
        else:
            self.vgg_loss = None

        self.config = configuration
        self.device = device

        self.bce_loss = nn.BCELoss().to(device)
        self.mse_loss = nn.MSELoss().to(device)

        self.tb_logger = tb_logger
        if tb_logger is not None:
            from tensorboard_logger import TensorBoardLogger
            encoder_final = self.encoder_decoder.encoder._modules['final_layer']
            encoder_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/encoder_out'))
            decoder_final = self.encoder_decoder.decoder.message_linear
            decoder_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/decoder_out'))

    def train_on_batch(self, batch: list):
        images, messages = batch
        batch_size = images.shape[0]
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)

        # === 1. Forward pass through encoder-decoder ===
        self.encoder_decoder.train()
        with torch.enable_grad():
            encoded_images, noised_images, decoded_messages, recovered_images = self.encoder_decoder(images, messages)

            # === 2. Train Discriminator ===
            self.discriminator.train()
            self.optimizer_discrim.zero_grad()
            real_outputs = self.discriminator(images)
            fake_outputs = self.discriminator(encoded_images.detach())
            d_loss_real = self.bce_loss(real_outputs, real_labels)
            d_loss_fake = self.bce_loss(fake_outputs, fake_labels)
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            self.optimizer_discrim.step()

            # === 3. Train Encoder-Decoder (Generator) ===
            self.optimizer_enc_dec.zero_grad()
            # Perceptual (VGG) loss
            if self.vgg_loss is not None:
                vgg_on_cov = self.vgg_loss(images)
                vgg_on_enc = self.vgg_loss(encoded_images)
                vgg_on_rec = self.vgg_loss(recovered_images)
                perceptual_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc)
                perceptual_loss_rec = self.mse_loss(vgg_on_cov, vgg_on_rec)
            else:
                perceptual_loss_enc = 0.0
                perceptual_loss_rec = 0.0
            # L2 regularization between original and encoded images
            l2_reg = torch.mean((encoded_images - images) ** 2)
            # Standard losses
            g_loss_enc = self.mse_loss(encoded_images, images)
            g_loss_dec = self.mse_loss(decoded_messages, messages)
            g_loss_img = self.mse_loss(recovered_images, images)
            # Adversarial loss: generator wants discriminator to think encoded images are real
            fake_outputs_for_g = self.discriminator(encoded_images)
            g_adv_loss = self.bce_loss(fake_outputs_for_g, real_labels)
            # Total generator loss
            g_loss = (
                self.config.encoder_loss * g_loss_enc +
                self.config.decoder_loss * g_loss_dec +
                5.0 * g_loss_img +  # image loss weight increased to 5.0
                self.config.adversarial_loss * g_adv_loss +
                0.1 * perceptual_loss_enc +  # weight for perceptual loss (tunable)
                0.1 * perceptual_loss_rec +  # weight for perceptual loss on recovered image (tunable)
                0.01 * l2_reg  # weight for L2 regularization (tunable)
            )
            g_loss.backward()
            self.optimizer_enc_dec.step()

        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1])

        losses = {
            'loss           ': g_loss.item(),
            'encoder_mse    ': g_loss_enc.item(),
            'dec_mse        ': g_loss_dec.item(),
            'img_recovery   ': g_loss_img.item(),
            'bitwise-error  ': bitwise_avg_err,
            'd_loss         ': d_loss.item(),
            'g_adv_loss     ': g_adv_loss.item(),
            'perceptual_enc ': perceptual_loss_enc.item() if self.vgg_loss is not None else 0.0,
            'perceptual_rec ': perceptual_loss_rec.item() if self.vgg_loss is not None else 0.0,
            'l2_reg         ': l2_reg.item(),
        }
        return losses, (encoded_images, noised_images, decoded_messages, recovered_images)

    def validate_on_batch(self, batch: list):
        if self.tb_logger is not None:
            encoder_final = self.encoder_decoder.encoder._modules['final_layer']
            self.tb_logger.add_tensor('weights/encoder_out', encoder_final.weight)
            decoder_final = self.encoder_decoder.decoder.message_linear
            self.tb_logger.add_tensor('weights/decoder_out', decoder_final.weight)

        images, messages = batch
        batch_size = images.shape[0]
        self.encoder_decoder.eval()
        with torch.no_grad():
            encoded_images, noised_images, decoded_messages, recovered_images = self.encoder_decoder(images, messages)
            # Perceptual (VGG) loss
            if self.vgg_loss is not None:
                vgg_on_cov = self.vgg_loss(images)
                vgg_on_enc = self.vgg_loss(encoded_images)
                vgg_on_rec = self.vgg_loss(recovered_images)
                perceptual_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc)
                perceptual_loss_rec = self.mse_loss(vgg_on_cov, vgg_on_rec)
            else:
                perceptual_loss_enc = 0.0
                perceptual_loss_rec = 0.0
            # L2 regularization between original and encoded images
            l2_reg = torch.mean((encoded_images - images) ** 2)
            # Standard losses
            g_loss_enc = self.mse_loss(encoded_images, images)
            g_loss_dec = self.mse_loss(decoded_messages, messages)
            g_loss_img = self.mse_loss(recovered_images, images)
            g_loss = (
                self.config.encoder_loss * g_loss_enc +
                self.config.decoder_loss * g_loss_dec +
                5.0 * g_loss_img +  # image loss weight increased to 5.0
                0.1 * perceptual_loss_enc +
                0.1 * perceptual_loss_rec +
                0.01 * l2_reg
            )

        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1])

        losses = {
            'loss           ': g_loss.item(),
            'encoder_mse    ': g_loss_enc.item(),
            'dec_mse        ': g_loss_dec.item(),
            'img_recovery   ': g_loss_img.item(),
            'bitwise-error  ': bitwise_avg_err,
            'perceptual_enc ': perceptual_loss_enc.item() if self.vgg_loss is not None else 0.0,
            'perceptual_rec ': perceptual_loss_rec.item() if self.vgg_loss is not None else 0.0,
            'l2_reg         ': l2_reg.item(),
        }
        return losses, (encoded_images, noised_images, decoded_messages, recovered_images)

    def to_string(self):
        return '{}'.format(str(self.encoder_decoder))
