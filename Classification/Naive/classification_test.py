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

        # L1 784 inputs -> 10 outputs
        self.fc1 = torch.nn.Linear(784, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        out = self.fc1(x)
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
    X_single_data = input_tensor.view(-1, 28 * 28).float().to(device)
    #X_single_data = mnist_test.test_data[r:r + 1].view(1, 1, 28, 28).float().to(device)
    #Y_single_data = mnist_test.test_labels[r:r + 1].to(device)

    #print('Label: ', Y_single_data.item())
    single_prediction = model(X_single_data)
    print('Prediction: ', torch.argmax(single_prediction, 1).item())

    plt.imshow(input_tensor.view(28, 28), cmap='Greys', interpolation='nearest')
    #plt.imshow(mnist_test.test_data[r:r + 1].view(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()