import torch
import time

resneXt = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resneXt', pretrained=False)

device = torch.device("cuda:0")
net = torch.nn.DataParallel(resneXt, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
net.to(device)
torch.backends.cudnn.enabled = False

print("Start Running...")
while True:
    imgs = torch.rand((80, 3, 512, 512)).contiguous()
    imgs = imgs.to(device)
    outputs = net(imgs)