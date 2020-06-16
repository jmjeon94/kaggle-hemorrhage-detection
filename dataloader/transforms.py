import torchvision.transforms as T

def build_transform():
    transforms = T.Compose([T.Resize([256,256]),
                        T.ToTensor()])
    return transforms