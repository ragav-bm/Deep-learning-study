import torch as t
import torch.nn as nn


from sklearn.semi_supervised.tests.test_self_training import y_train
from torch.utils.data.dataloader import DataLoader
from sklearn.utils.class_weight import compute_class_weight

from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torchvision.models as models

data = pd.read_csv("data.csv", sep=";")
print(data.head())
train_data,val_data = train_test_split(data,test_size=0.2,random_state=42)

y_train1 = train_data.iloc[:, 1:].values.flatten()

class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_train1)
class_weights = t.tensor(class_weights, dtype=t.float32)


print("Class Weights:", class_weights)

train_dataset = ChallengeDataset(data = train_data, mode="train")
val_dataset = ChallengeDataset(data = val_data, mode="val")
train_DataLoader = t.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_DataLoader = t.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

resnet = models.resnet18(pretrained=True)
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, 2)
loss_criterion  = t.nn.BCEWithLogitsLoss()

optimizer = t.optim.Adam(resnet.parameters(), lr=0.001)
trainer = Trainer(model=resnet, crit=loss_criterion, optim=optimizer, train_dl=train_DataLoader,val_test_dl=val_DataLoader,cuda=t.cuda.is_available(),early_stopping_patience=10)
res = trainer.fit(100)

plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')

