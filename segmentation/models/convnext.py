import torch
import torch.nn as nn
# from fcn import Upsample, Fusion
import timm
import torch.nn.functional as F
from collections import namedtuple


class Upsample(nn.Module):
    def __init__(self, inplanes, planes):
        super(Upsample, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x, size):
        x = F.interpolate(x, size=size, mode="bilinear")
        x = self.conv1(x)
        x = self.bn(x)
        return x


class Fusion(nn.Module):
    def __init__(self, inplanes):
        super(Fusion, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=1)
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(.1)

    def forward(self, x1, x2):
        out = self.bn(self.conv(x1)) + x2
        out = self.relu(out)

        return out

class ConvNeXtV2FeatureExtractor(nn.Module):
    def __init__(self, model_name="convnext_small", pretrained=True):
        super(ConvNeXtV2FeatureExtractor, self).__init__()
        
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)

    # def forward(self, x):
    #     features = self.backbone(x)
    #     return features[-1]

    def forward(self, x):
        features = self.backbone(x)
        # print("Feature shape:", [f.shape for f in features])  # (batch_size, 1024, 14, 14)
        # Feature shape: [torch.Size([1, 96, 56, 56]), torch.Size([1, 192, 28, 28]), torch.Size([1, 384, 14, 14]), torch.Size([1, 768, 7, 7])]

        output = {
            'conv_x': features[0],  # Max Feature map (56x56)
            'pool_x': features[1],  # (28x28)
            'fm2': features[2],     # (14x14)
            'fm3': features[3],     # (7x7)
            'img_size': x.size()[2:]
        }
        return output

# class ConvNeXtV2Decoder(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super(ConvNeXtV2Decoder, self).__init__()
        
#         self.decoder = nn.Sequential(
#             nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(128, num_classes, kernel_size=1)
#         )

#     def forward(self, x):
#         return self.decoder(x)


class ConvNeXtV2Decoder(nn.Module):
    def __init__(self, n_class):
        super(ConvNeXtV2Decoder, self).__init__()

        self.n_class = n_class

        self.upsample1 = Upsample(768, 384)
        self.upsample2 = Upsample(384, 192)
        self.upsample3 = Upsample(192, 96)
        self.upsample4 = Upsample(96, 64)

        self.fs1 = Fusion(384)
        self.fs2 = Fusion(192)
        self.fs3 = Fusion(96)
        self.fs4 = Fusion(64)

        self.out4 = self._classifier(64)

    def _classifier(self, inplanes):
        return nn.Sequential(
            nn.Conv2d(inplanes, inplanes // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inplanes // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes // 2, self.n_class, kernel_size=1)
        )

    def forward(self, features):
        fsfm1 = self.fs1(features['fm2'], self.upsample1(features['fm3'], features['fm2'].size()[2:]))
        fsfm2 = self.fs2(features['pool_x'], self.upsample2(fsfm1, features['pool_x'].size()[2:]))
        fsfm3 = self.fs3(features['conv_x'], self.upsample3(fsfm2, features['conv_x'].size()[2:]))
        fsfm4 = self.upsample4(fsfm3, features['img_size'])

        out = self.out4(fsfm4)
        return out



# feature_extractor = ConvNeXtV2FeatureExtractor("convnext_small", pretrained=True)
# decoder = ConvNeXtV2Decoder(in_channels=768, num_classes=3)

# input_tensor = torch.randn(1, 3, 224, 224)
# features = feature_extractor(input_tensor)
# output = decoder(features)

# print("Feature shape:", features.shape)  # (batch_size, 1024, 14, 14)
# print("Segmentation output shape:", output.shape)  # (batch_size, 21, 28, 28)

feature_extractor = ConvNeXtV2FeatureExtractor("convnext_small", pretrained=True)
decoder = ConvNeXtV2Decoder(n_class=3)

input_tensor = torch.randn(1, 3, 224, 224)
features = feature_extractor(input_tensor)

for key, value in features.items():
    if key != 'img_size':
        print(f"{key}: {value.shape}")
output = decoder(features)
print("Output shape:", output.shape)  # Expected: torch.Size([1, 3, 224, 224])
