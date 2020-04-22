from torch.utils.data import Dataset
from PIL import Image
import io


class PytorchImagedataset(Dataset):
    def __init__(self, img_paths, uuid4proc, transforms):
        self.transform = transforms
        self._uuid4proc = uuid4proc
        self.image_path_list = img_paths
        if not self.image_path_list:
            self._log.info("Given file path contains no images of specified format {}".format(img_paths[0].split('.')[-1]))

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        img = Image.open(io.BytesIO(self.image_path_list[self._uuid4proc[idx]].get_bytes()))
        img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        res = (img, self._uuid4proc[idx])
        return res
