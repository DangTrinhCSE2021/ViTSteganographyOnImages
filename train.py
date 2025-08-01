# import os
# import time
# import torch
# import numpy as np
# import utils
# import logging
# from collections import defaultdict
# import math
# from skimage.metrics import structural_similarity as ssim

# from options import *
# from model.hidden import Hidden
# from average_meter import AverageMeter


# def compute_psnr(img1, img2):
#     mse = torch.mean((img1 - img2) ** 2)
#     if mse == 0:
#         return float('inf')
#     PIXEL_MAX = 2.0  # since images are in range [-1, 1]
#     return 20 * math.log10(PIXEL_MAX / math.sqrt(mse.item()))


# def compute_ssim(img1, img2):
#     # img1, img2: torch tensors, shape (B, C, H, W), range [-1, 1]
#     img1 = img1.detach().cpu().numpy()
#     img2 = img2.detach().cpu().numpy()
#     ssim_vals = []
#     for i in range(img1.shape[0]):
#         # Convert to (H, W, C) and rescale to [0, 1]
#         x = ((img1[i].transpose(1, 2, 0) + 1) / 2).clip(0, 1)
#         y = ((img2[i].transpose(1, 2, 0) + 1) / 2).clip(0, 1)
#         ssim_val = ssim(x, y, data_range=1.0, channel_axis=2)
#         ssim_vals.append(ssim_val)
#     return sum(ssim_vals) / len(ssim_vals) if ssim_vals else 0


# def train(model: Hidden,
#           device: torch.device,
#           hidden_config: HiDDenConfiguration,
#           train_options: TrainingOptions,
#           this_run_folder: str,
#           tb_logger):
#     """
#     Trains the HiDDeN model
#     :param model: The model
#     :param device: torch.device object, usually this is GPU (if avaliable), otherwise CPU.
#     :param hidden_config: The network configuration
#     :param train_options: The training settings
#     :param this_run_folder: The parent folder for the current training run to store training artifacts/results/logs.
#     :param tb_logger: TensorBoardLogger object which is a thin wrapper for TensorboardX logger.
#                 Pass None to disable TensorboardX logging
#     :return:
#     """

#     train_data, val_data = utils.get_data_loaders(hidden_config, train_options)
#     file_count = len(train_data.dataset)
#     if file_count % train_options.batch_size == 0:
#         steps_in_epoch = file_count // train_options.batch_size
#     else:
#         steps_in_epoch = file_count // train_options.batch_size + 1

#     print_each = 10
#     images_to_save = 8
#     saved_images_size = (512, 512)

#     best_val_error = float('inf')
#     best_epoch = -1

#     for epoch in range(train_options.start_epoch, train_options.number_of_epochs + 1):
#         logging.info('\nStarting epoch {}/{}'.format(epoch, train_options.number_of_epochs))
#         logging.info('Batch size = {}\nSteps in epoch = {}'.format(train_options.batch_size, steps_in_epoch))
#         training_losses = defaultdict(AverageMeter)
#         epoch_start = time.time()
#         step = 1
#         for image, _ in train_data:
#             image = image.to(device)
#             message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], hidden_config.message_length))).to(device)
#             losses, _ = model.train_on_batch([image, message])

#             for name, loss in losses.items():
#                 training_losses[name].update(loss)
#             if step % print_each == 0 or step == steps_in_epoch:
#                 logging.info(
#                     'Epoch: {}/{} Step: {}/{}'.format(epoch, train_options.number_of_epochs, step, steps_in_epoch))
#                 utils.log_progress(training_losses)
#                 logging.info('-' * 40)
#             step += 1

#         train_duration = time.time() - epoch_start
#         logging.info('Epoch {} training duration {:.2f} sec'.format(epoch, train_duration))
#         logging.info('-' * 40)
#         utils.write_losses(os.path.join(this_run_folder, 'train.csv'), training_losses, epoch, train_duration)
#         if tb_logger is not None:
#             tb_logger.save_losses(training_losses, epoch)
#             tb_logger.save_grads(epoch)
#             tb_logger.save_tensors(epoch)

#         first_iteration = True
#         validation_losses = defaultdict(AverageMeter)
#         psnr_values = []
#         ssim_watermarked_values = []
#         ssim_recovered_values = []
#         logging.info('Running validation for epoch {}/{}'.format(epoch, train_options.number_of_epochs))
#         for image, _ in val_data:
#             image = image.to(device)
#             message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], hidden_config.message_length))).to(device)
#             losses, (encoded_images, noised_images, decoded_messages, recovered_images) = model.validate_on_batch([image, message])
#             for name, loss in losses.items():
#                 validation_losses[name].update(loss)
#             # Compute PSNR for each batch
#             psnr = compute_psnr(image, encoded_images)
#             psnr_values.append(psnr)
#             # Compute SSIM for each batch
#             ssim_watermarked = compute_ssim(image, encoded_images)
#             ssim_recovered = compute_ssim(image, recovered_images)
#             ssim_watermarked_values.append(ssim_watermarked)
#             ssim_recovered_values.append(ssim_recovered)
#             if first_iteration:
#                 if hidden_config.enable_fp16:
#                     image = image.float()
#                     encoded_images = encoded_images.float()
#                     recovered_images = recovered_images.float()
#                 utils.save_images(image.cpu()[:images_to_save, :, :, :],
#                                   encoded_images[:images_to_save, :, :, :].cpu(),
#                                   epoch,
#                                   os.path.join(this_run_folder, 'images'), resize_to=saved_images_size)
#                 # Save recovered images
#                 utils.save_images(image.cpu()[:images_to_save, :, :, :],
#                                   recovered_images[:images_to_save, :, :, :].cpu(),
#                                   epoch,
#                                   os.path.join(this_run_folder, 'recovered_images'), resize_to=saved_images_size)
#                 first_iteration = False

#         avg_psnr = sum(psnr_values) / len(psnr_values) if psnr_values else 0
#         avg_ssim_watermarked = sum(ssim_watermarked_values) / len(ssim_watermarked_values) if ssim_watermarked_values else 0
#         avg_ssim_recovered = sum(ssim_recovered_values) / len(ssim_recovered_values) if ssim_recovered_values else 0
#         logging.info(f'Average PSNR for epoch {epoch}: {avg_psnr:.2f} dB')
#         logging.info(f'Average SSIM (watermarked) for epoch {epoch}: {avg_ssim_watermarked:.4f}')
#         logging.info(f'Average SSIM (recovered) for epoch {epoch}: {avg_ssim_recovered:.4f}')
#         utils.log_progress(validation_losses)
#         logging.info('-' * 40)
#         utils.write_losses(os.path.join(this_run_folder, 'validation.csv'), validation_losses, epoch,
#                            time.time() - epoch_start)

#         # Only save the best model (lowest validation bitwise-error)
#         val_bitwise_error = validation_losses['bitwise-error  '].avg
#         if val_bitwise_error < best_val_error:
#             best_val_error = val_bitwise_error
#             best_epoch = epoch
#             logging.info(f'New best model found at epoch {epoch} with bitwise-error {val_bitwise_error:.4f}. Saving checkpoint.')
#             utils.save_checkpoint(model, train_options.experiment_name, epoch, os.path.join(this_run_folder, 'checkpoints'))
#         else:
#             logging.info(f'No improvement in bitwise-error ({val_bitwise_error:.4f}) at epoch {epoch}. Best so far: {best_val_error:.4f} at epoch {best_epoch}.')
