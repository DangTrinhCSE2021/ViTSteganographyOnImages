import torch
import torch.nn as nn
import torch.nn.functional as F
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu


class Decoder(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts both the watermark and recovers the original image.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """
    def __init__(self, config: HiDDenConfiguration):

        super(Decoder, self).__init__()
        self.channels = config.decoder_channels
        self.H = config.H
        self.W = config.W

        # Message extraction branch (existing)
        layers_message = [ConvBNRelu(3, self.channels)]
        for _ in range(config.decoder_blocks - 1):
            layers_message.append(ConvBNRelu(self.channels, self.channels))
        layers_message.append(ConvBNRelu(self.channels, config.message_length))
        layers_message.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.message_layers = nn.Sequential(*layers_message)
        self.message_linear = nn.Linear(config.message_length, config.message_length)

        # Image recovery branch (new) - ensure same output size as input
        layers_image = [ConvBNRelu(3, self.channels)]  # padding=1 is default in ConvBNRelu
        for _ in range(config.decoder_blocks - 1):
            layers_image.append(ConvBNRelu(self.channels, self.channels))
        layers_image.append(nn.Conv2d(self.channels, 3, kernel_size=3, padding=1))  # Final conv with padding=1
        self.image_layers = nn.Sequential(*layers_image)

    def forward(self, image_with_wm):
        # Message extraction
        message_features = self.message_layers(image_with_wm)
        message_features = message_features.squeeze(3).squeeze(2)
        extracted_message = self.message_linear(message_features)

        # Image recovery
        recovered_image = self.image_layers(image_with_wm)
        recovered_image = torch.tanh(recovered_image)  # Output in range [-1, 1]
        # Ensure output matches input size
        recovered_image = F.interpolate(recovered_image, size=(self.H, self.W), mode='bilinear', align_corners=False)

        return extracted_message, recovered_image
