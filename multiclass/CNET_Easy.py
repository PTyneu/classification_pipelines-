import torch
import torchvision
from torchvision.transforms import transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

BATCH_SIZE = 16
IMG_WIDTH = 640
IMG_HEIGHT = 640
CLASSES = ('через запятую пишешь названия классов каждое в кавычках')
NUM_CLASSES = len(CLASSES)
EPOCHS = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms = ([
    transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ДЗ найти как загружать кастомные датасеты в пайторч через даталоадер, не получится скину
#на выходе должны быть 
train_loader = "чему же я равен, Никита?"
test_loader = "чему же я равен, Никита?"

class ShnayterNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
# c архитектурой количества сверток в начале поиграй, эту сетку юзал для цифаровского датасета, 
# там пикчи 28х28, пасхалку еще оставил, почитай что такое receptive field и как его уменьшить
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# сама сетка проще некуда - свертка пулинг свертка больше пулинг и четыре полносвзяных слоя, 
# активации везде релу, с ними градиент меньше тухнет, тоже можешь поиграться посмотреть
# также здесь можешь поиграться со слоями: попробовать пулинг по среднему, попробовать 
# батч-номализацию, дропаут, разные свертки, residual-блоки и тд, почитать почему не надо
# юзать софтмакс когда лосс кросс-энтропия

model = ShnayterNet()
# опять же, посмотри какие лоссы для каких задач юзаются, поиграй с оптимайзерами 
# Adam, RMSProp etc., почитай про регуляризацию, узнать че за моментум и поч есть сгд а есть сгд с моментумом
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

#   это все обучение, изучи как все работает
for epoch in range(EPOCHS):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        #кидаешь на гпу
        data = data.to(device)
        targets = targets.to(device)

        # вперед
        scores = model(data)
        loss = criterion(scores, targets)

        # назад
        optimizer.zero_grad()
        loss.backward()

        # чето с оптимизатором происходит непонятное
        optimizer.step()

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
#вот тут вообще суета, разберись
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)


    model.train()
    return num_correct/NUM_CLASSES

print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")