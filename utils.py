import  torch
from torchvision import  transforms
from PIL import image
import io

CIFAR10_Classes=[
    'airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'
]

CIFAR10_mean=(0.4914,0.4822,0.4461)
CIFAR10_std=(0.2470,0.2435,0.2616)

train_transform=transforms.Compose([
    transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_mean,CIFAR10_std)
])

val_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_mean,CIFAR10_std)
])

def preprocess_image_bytes(image_bytes):
    img=Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img=img.resize(32,32)
    tensor=val_transform(img).unsqueeze(0)
    return tensor