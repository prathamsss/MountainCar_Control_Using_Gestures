# Driver Code
import gym
import torch
from PIL import Image
from torchvision import  transforms as T
from torch.autograd import Variable
import cv2
import numpy as np
import argparse
''' 
    0      Accelerate to the Left
    1      Don't accelerate
    2      Accelerate to the Right
'''
class MountainCarControl(object):
    ''' Class to Control Mountain Car Using Hand Gesture'''
    def __init__(self,model_path):
        ''' We Initialise - 1) MountainCarV0 Environment
                            2) Setup Model for   Evaluation
                            3) Setup Web Cam OpenCV Parameters
        '''
        self.env = gym.make('MountainCar-v0')
        self.env.reset()
        self.model_path = model_path
        map_location = torch.device('cpu')
        # self.model = torch.load(self.model_path, map_location)
        self.model= torch.jit.load(self.model_path)
        # self.model.eval()
        self.cap = cv2.VideoCapture(0)


    def Predict_Action(self,frame):
        ''' Predicts the Hand Gesture on Given Frame
            inputs: frame - Input Frame on which actions need to be predicted. '''
        transforms = T.Compose([T.Resize((224,224)),T.ToTensor()])
        image = transforms(Image.fromarray(frame.astype(np.uint8))).float()
        image = image.unsqueeze_(0)
        input = Variable(image)
        output = self.model(input)
        output = output.data.numpy().argmax()
        return  output

    def CarControl(self):
        ''' Runs Gym Environment & And takes action according Gesture Predicted '''
        while True:
            ret, frame = self.cap.read()
            frame = cv2.flip(frame, 1)
            if not ret: break
            x1=800
            y1=20
            x2 = x1+400
            y2= y1+500
            cv2.rectangle(frame, (x1 , y1), (x2,y2), (255, 0, 0), 1)
            roi = frame[y1:y2, x1:x2]
            self.env.render()
            action = self.Predict_Action(roi)
            self.env.step(action)
            cv2.putText(frame,"0: Accelerate to the Left",(0, 40+50),cv2.FONT_HERSHEY_PLAIN,2, (0, 255, 0), 3)
            cv2.putText(frame, "1: Don't accelerate",  (0, 40+100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
            cv2.putText(frame, "2: Accelerate to the Right", (0, 40 +150), cv2.FONT_HERSHEY_PLAIN, 2,
                        (0, 255, 0), 3)
            cv2.putText(frame, "Predicted Action: ", (0, 40 +200), cv2.FONT_HERSHEY_PLAIN, 2,
                        (250,216,135), 3)
            cv2.putText(frame, str(action), (0+70, 40+350), cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 255), 6)
            cv2.imshow("Action Window", roi)
            cv2.imshow("Action Window2", frame)
            self.interrupt = cv2.waitKey(10) & 0xFF
            if self.interrupt == 27:
                cv2.destroyAllWindows()
                self.env.close()
                break



if __name__ == '__main__':
    # model_path = "/Users/prathameshsardeshmukh/PycharmProjects/Motor_AI_Test/Models/my_res18_best_ever.pth"
    parser = argparse.ArgumentParser(description='Hi...')
    parser.add_argument("model_path", type=str, default='/Models/my_res18_best_ever.pth',
                        help='path to Model weights')

    args = parser.parse_args()
    M = MountainCarControl(args.model_path)
    M.CarControl()
