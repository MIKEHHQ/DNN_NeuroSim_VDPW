import torch

# model_path = './log/2_bit_Resnet20_model_best.pth.tar'
# model_path = './log/Resnet20_model_best.pth.tar'
model_path = './pretrained_models/mbv1_pretrained_439.pth'
model_path = './pretrained_models/mbv1_200epoch.pth'
model = torch.load(model_path)
# print(model)
for i in model['net']:
    print(i)
# print(model)
