import torchvision.transforms as T
transforms = T.Compose([T.Resize([256,256]),
                        T.ToTensor()])