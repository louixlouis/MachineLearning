import random
import os

import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.init

import warnings
warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 랜덤 시드 고정
torch.manual_seed(777)

# GPU 사용 가능일 경우 랜덤 시드 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

learning_rate = 0.001
training_epochs = 15
batch_size = 100
checkpoints_path = './checkpoints'
if not os.path.isdir(checkpoints_path):
    os.makedirs(checkpoints_path)

mnist_test = datasets.MNIST(
    root='MNIST_data/', # 다운로드 경로 지정
    train=False, # False를 지정하면 테스트 데이터로 다운로드
    transform=transforms.ToTensor(), # 텐서로 변환
    download=True)

def load_image(path):
    image = Image.open(path)
    image = image.convert('L')

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    image_tensor = transform(image)
    return image_tensor

def save_checkpoint(model, opt, epoch, path):
    torch.save({
        'model':model.state_dict(),
        'optimizer': opt.state_dict(),
        'epoch': epoch
    }, os.path.join(path, f'model_{epoch}.tar'))

def load_checkpoint(path):
    pass

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.keep_prob = 0.5
        # L1 ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # L2 ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        # L3 ImgIn shape=(?, 7, 7, 64)
        #    Conv      ->(?, 7, 7, 128)
        #    Pool      ->(?, 4, 4, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        # L4 FC 4x4x128 inputs -> 625 outputs
        self.fc1 = torch.nn.Linear(4 * 4 * 128, 625, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1 - self.keep_prob))
        # L5 Final FC 625 inputs -> 10 outputs
        self.fc2 = torch.nn.Linear(625, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        out = self.layer4(out)
        out = self.fc2(out)
        return out

# CNN 모델 정의
model = CNN().to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)    # 비용 함수에 소프트맥스 함수 포함되어져 있음.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#total_batch = len(data_loader)
#print('총 배치의 수 : {}'.format(total_batch))

checkpoint = torch.load('./checkpoints/model_14.tar')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
epoch = checkpoint['epoch']

input_tensor = load_image('./test_image.png')

with torch.no_grad():
    # MNIST 테스트 데이터에서 무작위로 하나를 뽑아서 예측을 해본다
    r = random.randint(0, len(mnist_test) - 1)
    X_single_data = input_tensor.view(1, 1, 28, 28).float().to(device)
    #X_single_data = mnist_test.test_data[r:r + 1].view(1, 1, 28, 28).float().to(device)
    #Y_single_data = mnist_test.test_labels[r:r + 1].to(device)

    #print('Label: ', Y_single_data.item())
    single_prediction = model(X_single_data)
    print('Prediction: ', torch.argmax(single_prediction, 1).item())

    plt.imshow(input_tensor.view(28, 28), cmap='Greys', interpolation='nearest')
    #plt.imshow(mnist_test.test_data[r:r + 1].view(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()