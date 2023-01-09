import traceback
from typing import Callable

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class Trainer:
    model: nn.Module
    criterion: nn.Module
    optimizer: Optimizer
    juror: Callable

    def __init__(self, model, criterion, optimizer, scheduler=None, juror=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.juror = juror

    def run(self, epochs: int, train_dataloader: DataLoader, test_dataloader: DataLoader = None):
        self.train(train_dataloader, epochs)
        self.eval(test_dataloader)

    def train(self, dataloader: DataLoader, epochs: int = 1, log_epoch_steps: int = 10, log_batch_steps: int = 10):
        try:
            self.model.train()
            accuracy = 0
            count = 0
            epoch_interval = round(epochs / log_epoch_steps) or 1
            batch_interval = round(len(dataloader) / log_batch_steps) or 1

            print("Train ...")
            print('=' * 60)
            for e in range(epochs):
                for i, (x, y) in enumerate(dataloader):
                    self.optimizer.zero_grad()
                    z = self.model(x)
                    cost = self.criterion(z, y.squeeze(dim=1))
                    cost.backward()
                    self.optimizer.step()
                    self.scheduler and self.scheduler.step()

                    if self.juror:
                        accuracy += self.juror(z, y).item()
                        count += len(y)

                    if e and e % epoch_interval == 0 and i and i % batch_interval == 0:
                        print(f"{e:>4} | {i:>4} | {round(cost.item(), 6):>10} | {round(100 * accuracy / (count or 1), 2):>6}%")
                if e and e % epoch_interval == 0:
                    print('-' * 60)
                    print(f"{e:>4} | {i:>4} | {round(cost.item(), 6):>10} | {round(100 * accuracy / (count or 1), 2):>6}%")
                    print('=' * 60)
        except Exception as ex:
            traceback.print_exc()
            print('-' * 60)
            print(f"epoch = {e:>4}: batch = {i:>4}")
            print("input =", x)
            print("output =", z)
            print("target =", y)
            print('-' * 60)

    def eval(self, dataloader: DataLoader):
        self.model.eval()
        accuracies = 0
        approxies = 0
        count = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader):
                z = self.model(x).squeeze()
                zi = z.argmax()
                y = y.squeeze()
                accuracies += int(zi == y)
                approxies += int(abs(zi - y) < 2)
                count += 1
                # print(x, z, zi, y, acc, apx)
            print(
                f"Accuracy = {accuracies:>4}/{count} = {round(100 * accuracies / (count or 1), 2):>6}% |",
                f"Approximation = {approxies:>4}/{count} = {round(100 * approxies / (count or 1), 2):>6}%",
            )
