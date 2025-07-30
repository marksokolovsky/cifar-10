import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
import torch.optim.lr_scheduler as lr_scheduler

# текущая директория проекта
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# директория, в которой храним модели
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_NAME = "cnn_4"
MODEL_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}.pth")

epochs = 1
total_iters = 100 # lr_scheduler


"""
1.4 Аппаратные ресурсы
Для обеспечения справедливости и равных возможностей для всех, а также для того, чтобы не
оценивать, у кого больше вычислительных мощностей, ваша сеть должна обучаться в течение
максимум 4 часов на CPU (это может быть любая модель). Обучение на графических процессорах не
допускается. Если у вас нет подходящего компьютера, вы можете запросить доступ к факультетскому
вычислительному серверу. Обратите внимание, что со дня запроса до получения доступа может
пройти несколько дней, так что планируйте это заранее.
"""
# поэтому мы явно указываем использование cpu, при необходимости, первую строку можно расскомментировать и использовать GPU (при установке соответствующей версии torch)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(f"device: {device}")


"""
Вы можете использовать методы дополнения данных при условии, что изображения
дополняются изобучающего набора CIFAR-10, и не используются никакие другие внешние
изображения или знания
"""
# аугментация, обогащение обучающей выборки. Используем только самые быстрые преобразования
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(), 
    transforms.RandomRotation(15), 
    transforms.ColorJitter(), 
    transforms.RandomGrayscale(), 
    transforms.ToTensor(), 
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

# валидационную и тестовые выборки аугментировать, конечно, не надо
valid_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])


"""
1.2 Набор данных
Набор данных, который мы будем использовать, - это набор данных CIFAR-10. Подробности и
загрузка данных приведены на его сайте: https://www.cs.toronto.edu/~kriz/cifar.html. Однако вместо
загрузки данных вручную мы настоятельно рекомендуем использовать загрузчики наборов
данных в PyTorch (https://pytorch.org/vision/main/generated/torchvision.datasets.CIFAR10.html) для
прямой загрузки наборов данных в вашу среду, включая метки и разбиения на
тренировки/оценки/тесты.
Вам не разрешается использовать изображения из других наборов данных для обучения сети. Для
экспериментов необходимо использовать официальное разделение на обучающий,
валидационный и тестовый наборы, используемые в оригинальном наборе данных CIFAR-10.
Обучающий набор используется для обучения модели, валидационный набор - для
оценки производительности модели в процессе обучения и для настройки гиперпараметров, а
тестовый набор будет использоваться только для окончательной оценки.

"""
# датасет скачивается в папку data текущего проекта
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=valid_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

# посмотрим классы датасета
print(train_dataset.classes)
# ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

"""
1.3 Архитектура нейронной сети
Вам предстоит построить нейросетевой классификатор, который будет принимать на вход
изображение, а на выходе предсказывать определенный класс. Вам нужно будет использовать
соответствующие слои и функцию потерь, а также обучить сеть с помощью оптимизатора на
наборе данных.
Вы можете черпать вдохновение в существующих сетевых архитектурах, но вы должны
разработать/написать архитектуру самостоятельно, в своем собственном коде, используя
PyTorch. Использование существующих моделей или существующего кода моделей из онлайнисточников (например, GitHub) не допускается. Если вы черпаете вдохновение из существующего
кода, вы должны достаточно четко указать, что это ваш собственный код, а не скопированный и
вставленный из онлайн-источника.
Вы должны обучить свою собственную сеть с нуля, т.е. использование предварительно обученной
сети или предварительно обученных весов не допускается.
"""

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(32)          
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)         
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)         
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(8192, 256)
        self.bn4 = nn.BatchNorm1d(256)         

        self.fc2 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)         

        self.dropout = nn.Dropout(0.3)        
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  
        x = self.pool2(F.relu(self.bn2(self.conv2(x)))) 
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        # print(x.shape)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)                             
        x = self.fc3(x)
        return x

model = ConvNet().to(device)


# непосредственное обучение нашей модели
# Loss, Optimizer
min_valid_loss = np.inf

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=total_iters)

# Training
n_total_steps = len(train_loader)
for epoch in range(epochs):
    train_acc = 0.0
    train_loss = 0.0
    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate Loss
        train_loss += loss.item()        
        # Calculate Accuracy
        acc = ((outputs.argmax(dim=1) == labels).float().mean())
        train_acc += acc


    train_acc = train_acc / len(train_loader) * 100
    prev_train_loss = train_loss
    train_loss = train_loss / len(train_loader)  

    # Валидация
    valid_acc = 0.0
    valid_loss = 0.0
    model.eval()
    for images, labels in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Предсказание
        target = model(images)
        # Потери
        loss = criterion(target,labels)
        # Calculate Loss
        valid_loss += loss.item()
        # Calculate Accuracy
        acc = ((target.argmax(dim=1) == labels).float().mean())
        valid_acc += acc

    valid_acc = valid_acc / len(test_loader) * 100
    prev_valid_loss = valid_loss
    valid_loss = valid_loss / len(test_loader)

    before_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    after_lr = optimizer.param_groups[0]["lr"]

    print(f'Epoch {epoch+1} | Train Acc: {train_acc:.2f}% | Train Loss: {train_loss:.6f} | Valid Acc: {valid_acc:.2f}% | Valid Loss: {valid_loss:.6f}')
    print(f"before_lr: {before_lr:.5f} -> {after_lr:.5f}")

    if min_valid_loss > valid_loss:
        print(f'Потери валидации уменьшились: ({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Сохраняем модель')
        min_valid_loss = valid_loss
        
        # Saving State Dict
        torch.save(model.state_dict(), MODEL_PATH)

    print()
print('Тренировка завершена!')


# Тестирование, что получилось
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for _ in range(10)]
    n_class_samples = [0 for _ in range(10)]

    
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(labels.size(0)):
            label = labels[i]
            pred = predicted[i]
            if label == pred:
                n_class_correct[label] += 1


            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Суммарная точность: {acc:.2f}%')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Точность класса {train_dataset.classes[i]}: {acc:.2f}%')

"""
Суммарная точность: 66.86%
Точность класса airplane: 65.40%
Точность класса automobile: 84.70%
Точность класса bird: 50.10%
Точность класса cat: 48.20%
Точность класса deer: 66.00%
Точность класса dog: 55.70%
Точность класса frog: 77.00%
Точность класса horse: 70.30%
Точность класса ship: 80.10%
Точность класса truck: 71.10%

"""