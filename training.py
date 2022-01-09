from __future__ import print_function, division
from torch.optim import lr_scheduler
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
import os
from torchvision import datasets, transforms
from torch import nn, optim
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import DataLoader
from config_reader import read_config
import argparse

def GestureData(main_dir_path, batch_size):
  train_transform = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.ToTensor(),
                                      # transforms.CenterCrop(224),
                                      transforms.RandomRotation(20),
                                      # transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.3,hue=0),
                                      # transforms.Pad(2),
                                      # transforms.RandomPerspective(0.3,p=0.6),
                                      transforms.GaussianBlur(3)
                                      # transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                      # transforms.RandomErasing(p=0.2),
                                      # transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                      #                      std=[0.5, 0.5, 0.5]),

                                      ])

  train_data = ImageFolder(main_dir_path+'/train', transform=train_transform)
  test_data = ImageFolder(main_dir_path+'/test', transform=Compose([Resize((224,224)),ToTensor()]))

  train_loader = DataLoader(train_data,
                      batch_size=32,shuffle=True)
  test_loader = DataLoader(test_data,
                      batch_size=32,shuffle=False)

  print("Data Stats ==>",train_loader.dataset, test_loader.dataset)

  return train_loader, test_loader

def Visualise(data_loader):
  img,label = next(iter(data_loader))
  img = torchvision.utils.make_grid(img)
  npimg = img.numpy()
  print(' '.join('%5s' % (data_loader.dataset.classes)[label[j]] for j in range(4)))
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()

def Train(model, train_loader, test_loader, device, criterion, optimizer, scheduler, epochs, save_mode):
    steps = 0
    running_loss = 0
    print_every = 7
    train_losses, test_losses = [], []

    for epoch in range(epochs):

        for inputs, labels in train_loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            ''' Calculate Test Set Every on thresshold- print_every'''
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                train_losses.append(running_loss / len(train_loader))
                test_losses.append(test_loss / len(test_loader))
                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Test loss: {test_loss / len(test_loader):.3f}.. "
                      f"Test accuracy: {accuracy / len(test_loader):.3f}")
                running_loss = 0
                model.train()

    torch.save(model, os.path.join(save_mode))
    return model, train_losses, test_losses

def main(yml_file_path):
    process_configs = read_config(
        file_path=yml_file_path)

    train_loader, test_loader = GestureData(process_configs["Data_dir"], process_configs["batch_size"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    '''  Initialise Model Parameters'''
    model_conv = torchvision.models.mobilenet_v2(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    model_conv.classifier[1] = nn.Linear(1280, process_configs["no_classes"])
    # model_conv.fc = nn.Linear(num_ftrs, no_classes) # For Resnet

    model_conv = model_conv.to(device)
    criterion = nn.CrossEntropyLoss()

    # optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=learning_rate, momentum=momentum)
    optimizer_conv = optim.Adam(model_conv.parameters(), lr=process_configs["learning_rate"])

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, process_configs["step_size"],
                                           process_configs["gamma"])  # Decay LR by a factor of 0.1 every 7 epochs

    model_conv, train_losses, test_losses = Train(model_conv,train_loader,test_loader,device, criterion, optimizer_conv,
                                                  exp_lr_scheduler, process_configs['epochs'],
                                                  process_configs["save_model_path"]
                                                  )

    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.show()
    print("Training Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Evaluation Code')
    parser.add_argument("yml_file_path", type=str, default='model_config.yml',
                        help='Path where model_config.yml is present')
    args = parser.parse_args()

    main(args.yml_file_path)