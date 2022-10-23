import torch 
from torch import nn 
import torchvision
import torchvision.datasets as datasets 
import torchvision.transforms as transforms 
from torchvision import models
from torch.utils.data import DataLoader 
from torch import optim 
import random
from tqdm import tqdm 
import numpy as np
import os 
import shutil 


# для итеративности эксперимента 
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

# кинуть на гпу если доступно 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# константы 
NUM_CLASSES = 6
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
EPOCHS = 50
IMG_CHANNES = 3 

#здесь укажи где все пикчи
data_root = 'уууууууууу сука'
#создаст пути до папок (И МАМОК АХАХАХАХАХАХАХА) обучающей и валидационной 
train_dir = 'train'
val_dir = 'val'
test_dir = 'test'

#здесь перечислишь имена классов, пикчи должны быть названы соответственно
CLASS_NAMES = ['street_noise', 'room_blur' 'etc']
# смысл в чем: из папки с пикчами для обучения каждая восьмая 
# картинка будет улетать в валидационную выборку (смысл знаешь) 
for dir_name in [train_dir, val_dir]:
    for class_name in class_names:
        os.makedirs(os.path.join(dir_name, class_name), exist_ok=True)

for class_name in CLASS_NAMES:
    source_dir = os.path.join(data_root, 'train', class_name)
    for i, file_name in enumerate(tqdm(os.listdir(source_dir))):
        if i % 8 != 0:
            dest_dir = os.path.join(train_dir, class_name) 
        else:
            dest_dir = os.path.join(val_dir, class_name)
        shutil.copy(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))
#для теста берешь из отдельной папки и затираешь лейблы
shutil.copytree(os.path.join(data_root, 'test'), os.path.join(test_dir, 'unknown'))
# преобразования картинок в тензор, здесь ещеможно прописать аугментации
# типы аугментаций из торча можешь погуглить, там в оф. доках все
train_transforms = transforms.Compose([
    #transforms.RandomResizedCrop(512),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
#формируем датасет преобразованный валидация и обучалка 
train_dataset = torchvision.datasets.ImageFolder(train_dir, train_transforms)
val_dataset = torchvision.datasets.ImageFolder(val_dir, val_transforms)

#даталоадеры, они в сетку и будут пихать всякие вещи
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False)

#функция для обучения тут блять никитос ну хуй знает че сказать, просто будешь спрашивать

def train_model(model, loss, optimizer, scheduler, num_epochs):
    for epoch in range(num_epochs):
        print('Epoch {}/{}:'.format(epoch, num_epochs - 1), flush=True)

        # каждая эпоха по сути состоит из 2 частей: учится и предиктит 
        for phase in ['train', 'val']:
            if phase == 'train':
                dataloader = train_dataloader
                scheduler.step()
                model.train()  #  тут учится 
            else:
                dataloader = val_dataloader
                model.eval()   # тут evaluate (хз как на русском (: 

            running_loss = 0.
            running_acc = 0.

            
            for inputs, labels in tqdm(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                
                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(inputs)
                    loss_value = loss(preds, labels)
                    preds_class = preds.argmax(dim=1)

                   
                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()

                # тут еще припизднул фишку для тебя раннинг лосс: каждые 10 эпох у тебя лосс становится меньше чтобы снизить вероятность проехать ноль
                running_loss += loss_value.item()
                running_acc += (preds_class == labels.data).float().mean()

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = running_acc / len(dataloader)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), flush=True)

    return model


#Посмотри как ща из торча выгружать готовые модели, там импорт должен быть 
#ЭТОТ КУСОК ДЛЯ ГОТОВОЙ СЕТИ
model = models.resnet50(pretrained=True)


for param in model.parameters():
    param.requires_grad = False

model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
#КОНЕЦ КУСКА ДЛЯ ГОТОВОЙ СЕТИ
model = model.to(device)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

train_model(model, loss, optimizer, scheduler, num_epochs=EPOCHS)

#выебистая хуйня чтобы предиктить на пикчах тестовых
# ее сам не очень понимаю, просто она есть 
class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
#укажи путь до теста 
test_dataset = ImageFolderWithPaths('путь до теста укажи', val_transforms)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False)

#модель в эвальюэйт и предиктим, градиенты не обновляем 
model.eval()

test_predictions = []
test_img_paths = []
for inputs, labels, paths in tqdm(test_dataloader):
    inputs = inputs.to(device)
    labels = labels.to(device)
    with torch.set_grad_enabled(False):
        preds = model(inputs)
    test_predictions.append(
        torch.nn.functional.softmax(preds, dim=1)[:,1].data.cpu().numpy())
    test_img_paths.extend(paths)
#вектор предиктов вероятностей, если будет непонятно скинь что выдаст
test_predictions = np.concatenate(test_predictions)

