import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
import time
from torch import nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from  matplotlib import  pyplot as plt
import argparse

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    max_k = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)  # returns k largest elements from the tensor
    pred = pred.t()  # get the transpose
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def test_or_validate(data_loader, model_path, loss_func, verbose_display_iter):
    """
    Run evaluation
    """
    map_location = torch.device('cpu')
    model =  torch.load(model_path,map_location)
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    y_pred_list = []
    y_true_list = []

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(data_loader):

            y_true_list.extend(target.data.numpy()) #Store Actual label list

            model_output = model(input)
            loss = loss_func(model_output, target)
            y_pred_list.extend((torch.max(torch.exp(model_output), 1)[1]).data.numpy()) # Store predicted list
            output = model_output.float()
            loss = loss.float()

            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            # if i % verbose_display_iter == 0:
            #     print('Test/Validation: [{0}/{1}]\t'
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            #         i, len(data_loader), batch_time=batch_time, loss=losses, top1=top1))


    print(' * Accuracy @1 {top1.avg:.3f}'.format(top1=top1))
    print('Top1 error rate -> {}\n'.format(100 - top1.avg))
    classes = ('Zero','one','two')

    confu_mat = confusion_matrix(y_true_list,y_pred_list)
    print(confu_mat)

    display_confusion_mat = ConfusionMatrixDisplay(confusion_matrix=confu_mat,
                            display_labels = classes)

    display_confusion_mat.plot()
    plt.savefig('Confusion_Matrix.png')

    return top1.avg

def main(args):
    real_time_set = ImageFolder(args.dataset_path, transform=Compose([Resize((224,224)),ToTensor()]))
    real_time_set_loader = DataLoader(real_time_set,batch_size=32,shuffle=False)
    test_or_validate(real_time_set_loader,args.model_path,nn.CrossEntropyLoss(),10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Evaluation Code')

    parser.add_argument("--model_path", type = str,  default='Models/my_res18_best_ever.pth',
                        help ='path to Model weights')

    parser.add_argument("--dataset_path", type=str, default='New_dataset/real_time_test',
                        help='path to dataset for evaluation')

    # model_path_resnet = "/Users/prathameshsardeshmukh/PycharmProjects/Motor_AI_Test/Models/my_res18_best_ever.pth"
    # model_path_mobile_net = "/Users/prathameshsardeshmukh/PycharmProjects/Motor_AI_Test/Models/my_MobileNetV3.pth"
    args = parser.parse_args()
    main(args)