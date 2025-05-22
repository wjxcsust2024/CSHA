import torch
from thop import profile
from Net2 import Net

# Model
print('==> Building model..')
# model = torchvision.models.alexnet(pretrained=False)
model = Net()

dummy_input = torch.randn(2, 3, 512, 512)
flops, params = profile(model, (dummy_input,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))