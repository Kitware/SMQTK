import logging
from smqtk.algorithms.descriptor_generator.pytorch_model import PyTorchModelElement

try:
    import torch
except ImportError as ex:
    logging.getLogger(__name__).warning("Failed to import pytorch module: %s",
                                        str(ex))
else:
    from torch import nn

try:
    import torchvision
except ImportError as ex:
    logging.getLogger(__name__).warning("Failed to import pytorch module: %s",
                                        str(ex))
else:
    from torchvision import models
    from torchvision import transforms

__author__ = 'bo.dong@kitware.com'

__all__ = ["ImageNet_ResNet50",]

# The model structure class
class ImageNet_ResNet50_def(nn.Module):
    def __init__(self):
        super(ImageNet_ResNet50_def, self).__init__()
        self.resnet = models.resnet50(pretrained=True)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        features = x.view(x.size(0), -1)
        out = self.resnet.fc(features)

        return (features, out)

    def load(self, model_file):
        snapshot = torch.load(model_file)
        self.load_state_dict(snapshot['state_dict'])


class ImageNet_ResNet50(PyTorchModelElement):

    @classmethod
    def is_usable(cls):
        valid = torchvision is not None
        if not valid:
            cls.get_logger().debug("Pytorch or torchvision (or both) python \
                module cannot be imported")
        return valid

    def model_def(self):
        return ImageNet_ResNet50_def()

    def transforms(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

        return transform