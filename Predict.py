import torchvision.models
from PIL import Image
from torch.autograd import Variable
import argparse
import time
from torchvision.transforms import transforms
import torch

class Predict():
    '''
    Utility to predict On single Image
    Init: loads the model
    predict_img:
            inputs: img_path = Path to image
            outputs: Returns Predicted Class
    '''
    def __init__(self,model_path):
        try:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        except RuntimeError:
            self.device = torch.device('cpu')
        self.model = torch.load(model_path,self.device)

    def predict_img(self,img_path):
        image = Image.open(img_path)
        test_transforms = transforms.Compose([transforms.Resize(224),
                                              transforms.ToTensor(),
                                              ])
        image_tensor = test_transforms(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input = Variable(image_tensor)
        print("Input tensor Shape ==>",image_tensor.shape)
        input = input.to(self.device)
        start = time.time()
        output = self.model(input)
        print("Time taken = ", time.time() - start)
        index = output.data.cpu().numpy().argmax()
        return index


# if __name__ == '__main__':
#
#     #model_path_mobile_net = "/Users/prathameshsardeshmukh/PycharmProjects/Motor_AI_Test/Models_Visualisation/my_MobileNetV3.pth"
#     parser = argparse.ArgumentParser(description='Predict on Single Image')
#     parser.add_argument("--model_path", type=str, help='Path for image')
#     parser.add_argument("--img_path", type=str,help='Path for image')
#
#     args = parser.parse_args()
#     P = Predict(args.model_path)
#     output = P.predict_img(args.img_path)
#     print("Predicted Class ==>",output)

print(torchvision.models.resnet18(pretrained=True))