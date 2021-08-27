from torch.utils.data import Dataset
from PIL import Image
import os


class Mydata(Dataset):

    def __init__(self, root_dir, train, transform, download):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, '')
        self.img_path = os.listdir(self.path)
        self.train = train
        self.transform = transform
        self.download = download

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, img_name)
        img = Image.open(img_item_path)
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_path)


root_dir = "dataset"
label_dir = "radar_image"
