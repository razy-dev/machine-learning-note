from torch.utils.data import DataLoader

from engine.dataset import DatasetBuilder

model = None
criterion = None
optimizer = None
scheduler = None

if __name__ == "__main__":
    engine = PropEngine(model, criterion, optimizer, scheduler)

    train_dataset, test_dataset = DatasetBuilder(

    ).build(DataBuffer(size=10))
    engine.train(dataloader=DataLoader(train_dataset, batch_size=32, shuffle=False), ephchs=10)
    engine.eval(dataloader=DataLoader(test_dataset))
