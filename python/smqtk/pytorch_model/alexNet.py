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


# The model structure class
class AlexNet_def(nn.Module):
    def __init__(self):
        super(AlexNet_def, self).__init__()
        self.alexnet = models.alexnet(pretrained=True)
        # remove the last FC layer
        self.classifier = nn.Sequential(*list(self.alexnet.classifier.children())[:-1])

    def forward(self, x):
        x = self.alexnet.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)

        return x


class AlexNet(PyTorchModelElement):
    @classmethod
    def is_usable(cls):
        valid = torchvision is not None
        if not valid:
            cls.get_logger().debug("Pytorch or torchvision (or both) python \
                module cannot be imported")
        return valid

    def model_def(self):
        return AlexNet_def()

    def transforms(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize
        ])

        return transform