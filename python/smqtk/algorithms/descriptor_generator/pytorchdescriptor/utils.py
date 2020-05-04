from torch.utils.data import Dataset
from PIL import Image
import io


class PytorchImagedataset(Dataset):
    """
    A Pytorch dataset class that loads images for feature extraction,
    while maintaining a corresponde between their feature vectors
    and uuids.     
    """

    def __init__(self, data_elements, uuid4proc, transforms):
        """
        Create a Pytorch dataset for feature extraction using CNN.
        :param data_elements: A dictionary of uuids to corresponding
               smqtk.representation.DataElement 
        :type data_elements: dict[uuid, smqtk.representation.DataElement]
        :param uuid4proc: A queue of descriptor element uuids.
        :type uuid4proc: list[uuid]
        :param transforms: Augmentations and transforms applied to each
               image.
        :type tranforms: torchvision.transforms

        :return: A tuple containing the transformed image and corresponding
                 uuid.
        :rtype: tuple(torch.tensor, str)
        """
        self.transform = transforms
        self._uuid4proc = uuid4proc
        self.data_ele = data_elements

    def __len__(self):
        """
        Returns the length of dataset
        """
        return len(self.data_ele)

    def __getitem__(self, idx):
        """
        Returns both the transformed image tensor and its corresponding uuids
        at a random position inside the dataset.
        :param idx: id of a dataset elements to be fetched in current batch
               of feature extraction.
        :type idx: int or [int]
   
        :return res: A tuple of the image tensor and its uuid.
        :rtype res: tuple(torch.tensor, str) 
        """
        img = Image.open(io.BytesIO(self.data_ele[self._uuid4proc[idx]].get_bytes()))
        img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        res = (img, self._uuid4proc[idx])
        return res
