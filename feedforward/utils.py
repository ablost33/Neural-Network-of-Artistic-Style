import torchvision.transforms as transforms
from torch.autograd import Variable

from PIL import Image
import scipy.misc
import imageio
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import itertools


imsize = 256

loader = transforms.Compose([
    transforms.Scale(imsize),
    transforms.CenterCrop(imsize),
    transforms.ToTensor()])

unloader = transforms.ToPILImage()

unloader = transforms.ToPILImage()

# Opens image at a path and loads it as PyTorch variable of size imsize
def image_loader(image_name):
    image = Image.open(image_name)
    image = Variable(loader(image))
    image = image.unsqueeze(0)
    return image
 
# Ret 
def save_image(input, path):
    image = input.data.clone().cpu()
    image = image.view(3, imsize, imsize)
    image = unloader(image)
    imageio.imwrite(path, image)


def save_images(input, paths):
    N = input.size()[0]
    images = input.data.clone().cpu()
    for n in range(N):
        image = images[n]
        image = image.view(3, imsize, imsize)
        image = unloader(image)
        imageio.imwrite(paths[n], image)