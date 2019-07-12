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

__author__ = 'deepak.chittajallu@kitware.com'

__all__ = ["Cholec80_SelfSupervised_ResNet50",]

# The model structure class
class Cholec80_SelfSupervised_ResNet50_def(nn.Module):


    num_features = 4096
    """Number of output neurons, i.e., dimension of the learned feature space."""

    def __init__(self):
        super(Cholec80_SelfSupervised_ResNet50_def, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.feature = nn.Linear(
            12288, Cholec80_SelfSupervised_ResNet50_def.num_features)
        self.sig = nn.Sigmoid()

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
        x = x.view(x.size(0), -1)
        x = self.feature(x)

        return (self.sig(x),)

    def load(self, model_file):
        self.load_state_dict(torch.load(model_file))

    def save(self, model_file):
        torch.save(self.state_dict(), model_file)


class Cholec80_SelfSupervised_ResNet50(PyTorchModelElement):

    @classmethod
    def is_usable(cls):
        valid = torchvision is not None
        if not valid:
            cls.get_logger().debug("Pytorch or torchvision (or both) python \
                module cannot be imported")
        return valid

    def model_def(self):
        return Cholec80_SelfSupervised_ResNet50_def()

    def transforms(self):

        transform = transforms.Compose([
            transforms.Resize((216, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        return transform