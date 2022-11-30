import torchvision.transforms as transforms

custom_transform = transforms.Compose([
    transforms.Resize((448,448)),
    transforms.ToTensor(),
])

def create_transform():
    return custom_transform