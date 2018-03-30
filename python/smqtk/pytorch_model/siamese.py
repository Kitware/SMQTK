import logging
from smqtk.pytorch_model import PyTorchModelElement

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

__all__ = [ "Siamese", ]

# The model structure class
class Siamese_def(nn.Module):
    def __init__(self):
        super(Siamese_def, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        self.num_fcin = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.num_fcin, 500)
        self.pdist = nn.PairwiseDistance(1)

    def forward(self, x):
        x = self.resnet(x)
        return x

# Transforms for pre-processing
transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class Siamese(PyTorchModelElement):
    @classmethod
    def is_usable(cls):
        valid = torch is not None and torchvision is not None
        if not valid:
            cls.get_logger().debug("Pytorch or torchvision (or both) python \
                module cannot be imported")
        return valid

    def model_def(self):
        return Siamese_def()

    def transforms(self):
        return transform