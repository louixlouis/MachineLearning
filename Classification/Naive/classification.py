import random
import os

import matplotlib.pyplot as plt

import torch
import torchvision.datasets as dsets
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

mnist_train = dsets.MNIST(
    root='MNIST_data/', # 다운로드 경로 지정
    train=True, # True를 지정하면 훈련 데이터로 다운로드
    transform=transforms.ToTensor(), # 텐서로 변환
    download=True)

mnist_test = dsets.MNIST(
    root='MNIST_data/', # 다운로드 경로 지정
    train=False, # False를 지정하면 테스트 데이터로 다운로드
    transform=transforms.ToTensor(), # 텐서로 변환
    download=True)

data_loader = torch.utils.data.DataLoader(
    dataset=mnist_train,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True)

def save_checkpoint(model, opt, epoch, path):
    torch.save({
        'model':model.state_dict(),
        'optimizer': opt.state_dict(),
        'epoch': epoch
    }, os.path.join(path, f'model_{epoch}.tar'))

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
total_batch = len(data_loader)
print('총 배치의 수 : {}'.format(total_batch))

for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y느 ㄴ레이블.
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        X = X.view(-1, 28 * 28).to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    save_checkpoint(model, optimizer, epoch, checkpoints_path)
    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

with torch.no_grad():
    X_test = mnist_test.test_data.view(-1, 28 * 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())

    # MNIST 테스트 데이터에서 무작위로 하나를 뽑아서 예측을 해본다
    r = random.randint(0, len(mnist_test) - 1)
    X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
    Y_single_data = mnist_test.test_labels[r:r + 1].to(device)

    print('Label: ', Y_single_data.item())
    single_prediction = model(X_single_data)
    print('Prediction: ', torch.argmax(single_prediction, 1).item())

    plt.imshow(mnist_test.test_data[r:r + 1].view(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()