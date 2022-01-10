from time import perf_counter
import numpy as np
import time
import torchvision
import torch
from torch import  nn

class ModelConversion():
    def __init__(self,weights_path,torch_pretrained_model):
        self.torch_pretrained_model = torch_pretrained_model
        map_location = torch.device('cpu')
        weights = torch.load(weights_path, map_location)
        self.torch_pretrained_model.load_state_dict(weights.state_dict(), map_location)
        self.torch_pretrained_model.eval()

    def toJit_traced(self,save_model_path):
        example = torch.rand(1, 3, 224, 224)
        traced_script_module = torch.jit.trace(self.torch_pretrained_model, example)
        traced_script_module.save(save_model_path)
        print("Model has been sucessfully Converted!")

        jit_model = torch.jit.load(save_model_path)
        start = time.time()
        out = jit_model(example)
        print("Time Taken to predict ==>",(time.time()-start)*1000)

def timer(f, *args):
    start = perf_counter()
    f(*args)
    return (1000 * (perf_counter() - start))

weights_path = "/Models_Visualisation/my_res18_best_ever.pth"  #my_MobileNetV3.pth
map_location = torch.device('cpu')
example = torch.rand(1, 3, 224, 224)

''' For Inception '''

# model = torchvision.models.mobilenet_v2(pretrained=False)
# model.classifier[1] = nn.Linear(1280, 3)

''' For ResNet '''
model = torchvision.models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3) # For Resnet


m = ModelConversion(weights_path,model)
m.toJit_traced("/Users/prathameshsardeshmukh/PycharmProjects/Motor_AI_Test/Models_Visualisation/Traced_MobileNetV3.pt")


