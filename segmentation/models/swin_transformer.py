import torch
import torch.nn as nn
import timm

class SwinV2FeatureExtractor(nn.Module):
    def __init__(self, model_name="swinv2_tiny_window8_256", pretrained=True):
        super(SwinV2FeatureExtractor, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)

    def forward(self, x):
        features = self.backbone(x)
        output = {
            'fm1': features[0],  # (1, 64, 64, 96)
            'fm2': features[1],  # (1, 32, 32, 192)
            'fm3': features[2],  # (1, 16, 16, 384)
            'fm4': features[3],  # (1, 8, 8, 768)
        }
        return output


class SwinV2Decoder_out128_ker3_out256(nn.Module):
    def __init__(self, out_channels=3):
        super(SwinV2Decoder_out128_ker3_out256, self).__init__()
        self.n_class = out_channels

        self.up4 = self._upsample_block(768, 256)  # (8x8 → 16x16)
        self.conv4 = nn.Conv2d(640, 256, kernel_size=1)  # (256 + 384 → 256)

        self.up3 = self._upsample_block(256, 128)  # (16x16 → 32x32)
        self.conv3 = nn.Conv2d(320, 128, kernel_size=1)  # (128 + 192 → 128)

        self.up2 = self._upsample_block(128, 64)  # (32x32 → 64x64)
        self.conv2 = nn.Conv2d(160, 64, kernel_size=1)  # (64 + 96 → 64)

        self.up1 = self._upsample_block(64, 32)  # (64x64 → 128x128)
        self.up_final = self._upsample_block(32, 16)  # (128x128 → 256x256)

        self.classifier = self._classifier(16)

    def _upsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _classifier(self, inplanes):
        return nn.Sequential(
            nn.Conv2d(inplanes, inplanes // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inplanes // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes // 2, self.n_class, kernel_size=1)
        )

    def forward(self, features):
        fm1, fm2, fm3, fm4 = features['fm1'], features['fm2'], features['fm3'], features['fm4']
        fm1, fm2, fm3, fm4 = [f.permute(0, 3, 1, 2) for f in [fm1, fm2, fm3, fm4]]

        x = self.up4(fm4)  # (8x8 → 16x16)
        x = torch.cat([x, fm3], dim=1)
        x = self.conv4(x)

        x = self.up3(x)  # (16x16 → 32x32)
        x = torch.cat([x, fm2], dim=1)
        x = self.conv3(x)

        x = self.up2(x)  # (32x32 → 64x64)
        x = torch.cat([x, fm1], dim=1)
        x = self.conv2(x)

        x = self.up1(x)  # (64x64 → 128x128)
        x = self.up_final(x)  # (128x128 → 256x256)

        output = self.classifier(x)
        return output


class SwinV2Decoder_out128_ker3(nn.Module):
    def __init__(self, out_channels=3):
        super(SwinV2Decoder_out128_ker3, self).__init__()
        self.n_class = out_channels

        self.up4 = self._upsample_block(768, 256)  # (8x8 → 16x16)
        self.conv4 = nn.Conv2d(640, 256, kernel_size=1)

        self.up3 = self._upsample_block(256, 128)  # (16x16 → 32x32)
        self.conv3 = nn.Conv2d(320, 128, kernel_size=1)

        self.up2 = self._upsample_block(128, 64)  # (32x32 → 64x64)
        self.conv2 = nn.Conv2d(160, 64, kernel_size=1)

        self.up1 = self._upsample_block(64, 32)  # (64x64 → 128x128)
        self.classifier = self._classifier(32)
        # self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1) 
        # ---------------------------------------------------------------
        # self.up1 = self._upsample_block(64, 32)  # (64x64 → 128x128)
        # self.up_final = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)  # (128x128 → 256x256)

        # self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)  # Output 3 classes

    def _upsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _classifier(self, inplanes):
        return nn.Sequential(
            nn.Conv2d(inplanes, inplanes // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inplanes // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes // 2, self.n_class, kernel_size=1)
        )

    def forward(self, features):
        fm1, fm2, fm3, fm4 = features['fm1'], features['fm2'], features['fm3'], features['fm4']
        fm1, fm2, fm3, fm4 = [f.permute(0, 3, 1, 2) for f in [fm1, fm2, fm3, fm4]]

        x = self.up4(fm4)  # (8x8 → 16x16)
        x = torch.cat([x, fm3], dim=1)
        x = self.conv4(x)

        x = self.up3(x)  # (16x16 → 32x32)
        x = torch.cat([x, fm2], dim=1)
        x = self.conv3(x)

        x = self.up2(x)  # (32x32 → 64x64)
        x = torch.cat([x, fm1], dim=1)
        x = self.conv2(x)

        x = self.up1(x)  # (64x64 → 128x128)
        output = self.classifier(x)  # Classifier với BatchNorm
        return output
    
        # output = self.final_conv(x)  # (B, 3, 128, 128)
        # return output
    
        # x = self.up_final(x)  # (128x128 → 256x256)

        # output = self.final_conv(x)
        # return output

class SwinV2Decoder(nn.Module):
    def __init__(self, out_channels=3):
        super(SwinV2Decoder, self).__init__()
        self.n_class = out_channels

        self.up4 = self._upsample_block(768, 256)  # (8x8 → 16x16)
        self.conv4 = nn.Conv2d(640, 256, kernel_size=1)  # (256 + 384 → 256)

        self.up3 = self._upsample_block(256, 128)  # (16x16 → 32x32)
        self.conv3 = nn.Conv2d(320, 128, kernel_size=1)  # (128 + 192 → 128)

        self.up2 = self._upsample_block(128, 64)  # (32x32 → 64x64)
        self.conv2 = nn.Conv2d(160, 64, kernel_size=1)  # (64 + 96 → 64)

        self.up1 = self._upsample_block(64, 32)  # (64x64 → 128x128)
        
        # Classifier với kernel_size=5 thay vì 3
        self.classifier = self._classifier(32)

    def _upsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1),  # Changed kernel_size to 5
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _classifier(self, inplanes):
        return nn.Sequential(
            nn.Conv2d(inplanes, inplanes // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inplanes // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes // 2, self.n_class, kernel_size=1)
        )

    def forward(self, features):
        fm1, fm2, fm3, fm4 = features['fm1'], features['fm2'], features['fm3'], features['fm4']
        fm1, fm2, fm3, fm4 = [f.permute(0, 3, 1, 2) for f in [fm1, fm2, fm3, fm4]]

        x = self.up4(fm4)  # (8x8 → 16x16)
        x = torch.cat([x, fm3], dim=1)
        x = self.conv4(x)

        x = self.up3(x)  # (16x16 → 32x32)
        x = torch.cat([x, fm2], dim=1)
        x = self.conv3(x)

        x = self.up2(x)  # (32x32 → 64x64)
        x = torch.cat([x, fm1], dim=1)
        x = self.conv2(x)

        x = self.up1(x)  # (64x64 → 128x128)
        x = self.classifier(x)  # Classifier với BatchNorm
        
        return x
