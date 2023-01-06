from torch import nn, optim
from torch.utils.data import DataLoader

from engine.dataformats import TimestepFormat
from engine.dataset import DatasetBuilder, DataBuffer
from engine.models import RnnModel
from engine.normalizers import GradScaleNormalizer, GradNormalizer
from engine.trainer import Trainer
from engine.transforms import TensorTransform, TernaryIndexTransform

input_features = [
    GradScaleNormalizer(),
    GradScaleNormalizer(),
    GradScaleNormalizer(),
    GradScaleNormalizer(),
    GradScaleNormalizer(),
]
target_features = GradNormalizer()

time_steps = 20
input_size = len(input_features)  # input features
hidden_size = time_steps * 4
output_size = 3  # prediction features
target_size = 1  # target features

model = RnnModel(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size,
    batch_first=True,
    num_layers=1
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), 0.01)
scheduler = None


def juror(z, y):
    j = (z.argmax(dim=1) == y.squeeze(dim=1))
    return j.sum()


if __name__ == "__main__":
    train_dataset, test_dataset = DatasetBuilder().build(
        data=DataBuffer(size=1000).read(
            input_features=input_features,
            target_features=target_features,
            train_rate=0.7,
        ),
        format=TimestepFormat(
            time_steps=time_steps,
            target_size=target_size,
        ),
        input_transform=TensorTransform(),
        target_transform=TernaryIndexTransform()
    )

    engine = Trainer(model, criterion, optimizer, scheduler, juror=juror)
    engine.train(dataloader=DataLoader(train_dataset, batch_size=8), epochs=1000, log_batch_steps=5)
    engine.eval(dataloader=DataLoader(test_dataset))
