import torch
import torch.nn as nn

# BCELoss
z = torch.FloatTensor([0., 1.])
y = torch.FloatTensor([1, 0])
criterion = nn.BCELoss()
cost = criterion(z, y)
print(cost.shape, cost)

# BCELoss
z = torch.FloatTensor([0., 1.])
y = torch.FloatTensor([1, 0])
criterion = nn.BCELoss()
cost = criterion(z, y)
print(cost.shape, cost)
