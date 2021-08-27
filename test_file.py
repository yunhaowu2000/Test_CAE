from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

image_path = "dataset/_2009-01-01-01-17-54_0_38.png"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)
