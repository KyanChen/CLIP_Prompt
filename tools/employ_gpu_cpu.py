import torch
import time
import tqdm

resneXt = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resneXt', pretrained=False)

# device = torch.device("cpu")
net = torch.nn.DataParallel(resneXt, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])
# net = resneXt
torch.backends.cudnn.enabled = True

print("Start Running...")
while True:
    device = torch.device("cuda:0")
    imgs = torch.rand((80, 3, 512, 512)).contiguous()
    imgs = imgs.to(device)
    net.to(device)
    outputs = net(imgs)

    device = 'cpu'
    imgs = torch.rand((80, 3, 512, 512)).contiguous()
    imgs = imgs.to(device)
    net.to(device)
    outputs = net(imgs)

# t0 = time.time()
# imgs = torch.rand((1, 3, 1024, 1024))
# for _ in tqdm.tqdm(range(4)):
#     imgs = imgs.to(device)
#     outputs = net(imgs)
# print((time.time() - t0)/4)
#
# t0 = time.time()
# imgs = torch.rand((4, 3, 512, 512))
# for _ in tqdm.tqdm(range(4)):
#     imgs = imgs.to(device)
#     outputs = net(imgs)
# print((time.time() - t0)/4)

