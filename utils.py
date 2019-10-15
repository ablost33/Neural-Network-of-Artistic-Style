import torchvision.transforms as transforms
from torch.autograd import Variable

from PIL import Image
import scipy.misc
import imageio


imsize = 256

loader = transforms.Compose([
             transforms.Resize(imsize),
             transforms.ToTensor()
         ])

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


