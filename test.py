import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split

# текущая директория проекта
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# директория, в которой храним модели
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_NAME = "cnn_4_1"
MODEL_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}.pth")


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


# валидационную и тестовые выборки аугментировать, конечно, не надо
valid_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=valid_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)


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
#CNN
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

other_classes = {}
for i in range(10):
    other_classes[i] = {}


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
            else:
                other_classes_stat = other_classes[int(label)]
                try:
                    other_classes_stat[int(pred)] += 1
                except:
                    other_classes_stat[int(pred)] = 1
                other_classes[int(label)] = other_classes_stat


            n_class_samples[label] += 1

    
    acc = 100.0 * n_correct / n_samples
    print(f'Суммарная точность: {acc:.2f}%')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Точность класса {test_dataset.classes[i]}: {acc:.2f}%')
        other_classes_stat = other_classes[i]
        test_acc = 0
        for j in range(10):
            try:
                class_acc = 100.0 * other_classes_stat[j] / n_class_samples[i]
                test_acc += class_acc
                print(f'- {test_dataset.classes[j]}: {class_acc:.2f}')
            except:
                pass
        print(f"Неправильно распознано: {test_acc:.2f}%")
        print()


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
