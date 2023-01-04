# https://glanceyes.tistory.com/entry/PyTorch%EB%A1%9C-RNN-%EB%AA%A8%EB%8D%B8-%EA%B5%AC%ED%98%84%ED%95%B4%EB%B3%B4%EA%B8%B0-AG-NEWS-%EB%89%B4%EC%8A%A4-%EA%B8%B0%EC%82%AC-%EC%A3%BC%EC%A0%9C-%EB%B6%84%EB%A5%98#toc-link-0

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchtext
from torch.utils.data import Dataset, DataLoader, random_split
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from tqdm.notebook import tqdm_notebook

# 불러온 데이터를 저장할 위치를 지정해주세요.
data_dir = './data'  # TODO
dataset = torchtext.datasets.AG_NEWS(root=data_dir, split='train')
dataset = torchtext.datasets.AG_NEWS(root=data_dir, split='test')
tokenizer = get_tokenizer('basic_english')


def yield_tokens(data):
    for _, text in data:
        yield tokenizer(text)


vocab = build_vocab_from_iterator(yield_tokens(dataset), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])


def text_preprocess(x):
    return vocab(tokenizer(x))


def label_preprocess(x):
    return int(x) - 1


BASE_AG_NEWS_PATH = data_dir + '/datasets/AG_NEWS'
TRAIN_AG_NEWS_PATH = os.path.join(BASE_AG_NEWS_PATH, 'train.csv')
TEST_AG_NEWS_PATH = os.path.join(BASE_AG_NEWS_PATH, 'test.csv')


# CustomDataset을 정의할 때 torch.utils.data에 정의된 Dataset을 상속받습니다.
class CustomDataset(Dataset):
    # CustomDataset의 생성자를 지정합니다.
    # CustomDataset 클래스의 인스턴스를 만들 때 자동으로 호출됩니다.
    def __init__(self, path: str = f'{data_dir}/datasets/AG_NEWS/train.csv', train=True):
        tqdm_notebook.pandas(desc="PROGRESS>>")

        # 인스턴스로 자주 사용할 변수들을 정의합니다.

        # `.csv` 파일을 읽어올 때 '클래스(레이블)', '제목', '설명' 컬럼 데이터로 읽어와 data 멤버 변수에 저장합니다.
        self.data = pd.read_csv(path, sep=',', header=None, names=['class', 'title', 'description'])
        # 현재 Dataset이 학습용인지 테스트용인지를 저장합니다.
        self.train = train
        # 불러올 `.csv` 파일의 경로를 저장합니다.
        self.path = path

        # '제목'과 '설명' 컬럼을 합쳐서 학습할 데이터로 지정합니다.
        data = self.data['title'] + ' ' + self.data['description']
        # 모델에 학습할 데이터를 X로 지정하여 저장합니다.
        self.X = list()

        # '제목' + '설명' 데이터를 한 줄씩 읽으면서 데이터 X에 넣어줍니다.
        for line in data:
            self.X.append(line)
        # 데이터의 레이블을 y에 저장합니다.
        self.y = self.data['class']

        # dataset.classes를 출력하면 현재 데이터의 분류 레이블의 의미가 무엇인지 알 수 있도록 합니다.
        self.classes = ['World', 'Sports', 'Business', 'Sci/Tech']

    # len(dataset)을 호출하면 데이터 셋의 크기(길이)를 반환해줍니다.
    def __len__(self):
        len_dataset = None
        len_dataset = len(self.X)
        return len_dataset

    # dataset[idx]처럼 인덱스로 dataset에 접근했을 때 해당 인덱스(idx)에 있는 데이터를 반환할 수 있도록 합니다.
    def __getitem__(self, idx):
        X, y = None, None
        X = self.X[idx]
        if self.train is True:
            y = self.y[idx]
        # idx번째 있는 데이터를 (레이블, 텍스트)로 반환합니다.
        return y, X
        # 이런 형태 어디서 많이 보지 않으셨나요?
        # 위에서 next(iter(dataset))을 출력했을 때 반환되는 형태와 똑같습니다.
        # 결국 우리가 하고자 하는 건 위에서 사용했던 dataset을 따라서 직접 구현해보는 것과 같습니다.

    # 학습 데이터와 검증 데이터를 분리할 때 사용합니다. val_ratio는 검증 데이터를 분리할 비율 값을 의미합니다.
    def split_dataset(self, val_ratio=0.2):
        data_size = len(self)
        val_set_size = int(data_size * val_ratio)
        train_set_size = data_size - val_set_size
        # torch.utils.data의 random_split 메소드를 사용하여 데이터를 원하는 크기로 분리할 수 있습니다.
        train_set, val_set = random_split(self, [train_set_size, val_set_size])
        # 앞의 건 학습 데이터, 뒤의 건 검증 데이터로 반환합니다.
        return train_set, val_set


dataset = CustomDataset(TRAIN_AG_NEWS_PATH, train=True)
train_dataset, val_dataset = dataset.split_dataset(0.2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8


def collate_batch(batch):
    labels, texts, offsets = [], [], [0]
    for (label, text) in batch:
        # 레이블의 값을 1만큼 줄여서 모델의 입력 데이터로 넘깁니다.
        #  .
        labels.append(label_preprocess(label))
        # 텍스트를 단어장으로 번역된 수의 리스트로 바꾸고, 이를 다시 tensor 자료형으로 바꿉니다.
        # PyTorch에서는 기본적으로 tensor 자료형으로 모델을 학습시키거든요.
        processed_text = torch.tensor(text_preprocess(text), dtype=torch.int64)
        texts.append(processed_text)
        # Batch 크기를 일정하게 맞췄지만, 이렇게 될 경우 각 문장의 시작 위치가 어떤지는 알 수 없습니다.
        # 그래서 offsets에 각 문장의 시작 위치를 저장할 수 있도록 합니다.
        offsets.append(processed_text.size(0))

    # 레이블과 offsets 모두 모델이 소화할 수 있도록 tensor로 바꿔줍니다.
    labels = torch.tensor(labels, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    # tensor로 변환된 텍스트를 하나로 합칩니다.
    texts = torch.cat(texts)
    # 레이블, 변환된 텍스트, 오프셋 시작 위치 3가지를 반환하도록 합니다.
    return labels.to(device), texts.to(device), offsets.to(device)


# 학습 데이터의 Data Loader와 검증 데이터의 Data Loader를 각각 설정합니다.
# collate_fn 파라미터에 위에서 정의한 batch 적용 함수를 넘겨줍니다.
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)


class TextClassifier(nn.Module):
    # TextClassifier 클래스의 인스턴스를 생성할 때 자동으로 호출되는 함수입니다.
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        # 단어장의 크기를 저장합니다.
        self.vocab_size = vocab_size
        # Embedding을 거치고 난 후의 차원을 저장합니다.
        self.embed_dim = embed_dim
        # Hidden State의 차원을 저장합니다.
        self.hidden_dim = hidden_dim
        # RNN을 몇 단으로 stack처럼 쌓을지를 저장합니다.
        self.num_layers = num_layers
        # 최종적으로 분류해야 할 레이블의 수를 저장합니다.
        self.num_classes = num_classes

        # 커스텀 모델 클래스 내의 멤버변수에 원하는 Neural Network를 레이어로 정해서 커스텀 모델의 레이어를 깊게 쌓을 수 있습니다.
        # torch.nn에서 불러와서 원하는 neural network layer를 쌓을 수 있는 것이지요.
        # 이 실습에서는 EmbeddingBag, RNN, Linear 세 가지 레이어를 순차적으로 쌓아서 만들게요.

        # EmbeddingBag는 각 텍스트 문장의 tensor를 가방(bag)으로 묶어서 평균을 계산할 수 있습니다.
        # RNN에 일정한 차원 크기의 입력으로 데이터가 들어갈 수 있도록 미리 데이터를 embedding 시키는 것입니다.
        # 즉, RNN 레이어 모델이 잘 소화할 수 있도록 차원을 변환해주는 역할을 합니다.
        self.embedding = nn.EmbeddingBag(self.vocab_size, self.embed_dim, sparse=True)

        # 드디어 대망의 RNN을 사용하게 되네요!
        # RNN에서는 hidden state가 존재한다는 거 기억하시나요?
        # RNN 레이어는 기본적으로 텍스트의 임베딩 차원, hidden state 차원, 쌓을 RNN의 수 등을 파라미터로 넘깁니다.
        # 그런데 여기서 중요하게 봐야할 부분은 RNN의 입력을 어떠한 차원으로 줘야 하는가입니다. (RNN의 입력 차원과 텍스트의 임베딩 차원은 서로 다른 얘기입니다.)
        # 이 입력 차원을 설정할 때 batch_first 옵션에 True를 주면 RNN의 입력 차원에서 batch 크기가 맨 앞으로 이동하게 됩니다.
        self.rnn = nn.RNN(self.embed_dim, self.hidden_dim, self.num_layers, batch_first=True)
        # 4개의 라벨 중 하나로 예측해야 하므로 선형 변환하는 레이어를 설정합니다.
        self.linear = nn.Linear(self.hidden_dim, self.num_classes)

    # 모델에 데이터를 파라미터로 넘겨서 실행하면 자동적으로 모델의 `forward` 함수가 호출됩니다.
    def forward(self, text, offsets):
        # 위에서 설정한 EmbeddingBag Layer에 데이터를 넣습니다.
        # view를 사용하는 이유는, RNN의 입력 차원을 [batch_size, RNN에서 시퀀스로 판단되는 길이, 텍스트의 임베딩 차원]으로 바꾸기 위해서입니다.
        embedded = self.embedding(text, offsets).view(batch_size, -1, self.embed_dim)
        # 처음 RNN에 들어갈 hidden state를 0으로 초기화하는 작업입니다.
        # 참고로 RNN의 hidden state의 차원은 [num_layers, batch_size, hidden_dim]입니다.
        hidden = torch.zeros(
            self.num_layers, embedded.size(0), self.hidden_dim
        ).to(device)
        # RNN 레이어에 학습시키면, 마지막 cell에서의 hidden state와 RNN 레이어를 통과한 최종 결과인 각 batch별 cell별 output(hidden state)이 나옵니다.
        # RNN의 각 batch별 cell별 output 차원은 [batch_size, RNN에서 시퀀스로 판단되는 길이, hidden_dim]입니다.
        rnn_out, hidden = self.rnn(embedded, hidden)
        # RNN의 최종 결과에서 각 batch_size별로 마지막 cell에서 나온 hidden state의 결과를 가지고 선형 변환을 하여 레이블 수만큼의 차원으로 변환합니다.
        out = self.linear(rnn_out[:, -1:]).view([-1, self.num_classes])
        # 최종적으로 모델이 예측한 레이블 값이 반환될 것입니다.
        return out


# 단어장의 크기를 저장합니다.
vocab_size = len(vocab)
# 임베딩 차원의 크기를 64로 저장합니다.
embed_dim = 64
# RNN의 hidden state의 차원 크기를 32로 저장합니다.
hidden_dim = 32
# RNN을 1단만 쌓기로 합니다.
num_layers = 1
# 분류해야 할 레이블의 개수를 저장합니다.
num_classes = 4

model = TextClassifier(vocab_size, embed_dim, hidden_dim, num_layers, num_classes)
learning_rate = 0.01
epochs = 20

# Cross Entropy를 적용합니다.
criterion = torch.nn.CrossEntropyLoss()

# Optimizer를 설정하는 것인데, 여기서 주목해야할 점은 `model.parameters()`로 모델의 모든 파라미터를 optimizer 생성 인자로 넘긴다는 것입니다.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Scheduler는 학습 epoch을 늘려가는 과정에서 learning rate를 어떻게 조정할지를 정합니다.
# 일반적으로 앞의 epoch에서는 큰 learning rate로 진행하다가 뒤로 갈수록 작은 learning rate로 바꿔줘야 loss 함수의 극소로 도달하는 데 유리합니다.
# StepLR은 step size마다 gamma의 비율로 learning rate를 감소시키는 방법입니다.
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.5)


def train(dataloader, epoch):
    # 모델을 학습시키기 위해 모델을 학습 모드로 바꿔줍니다.
    # 모델이 학습 모드일 때와 검증 모드일 때 메모리 사용과 연산의 효율성이 다르다고 합니다.
    # 또한 검증 모드일 때는 Batch Normalization과 Dropout 등이 적용이 안되어서 학습 또는 검증 전에 원하는 모드로 모델을 바꿔주는 게 필요합니다.
    model.train()

    train_acc = 0
    train_count = 0

    # 얼마 만큼의 batch 간격마다 현재 모델의 학습 데이터에 관한 정확도를 출력할지 정합니다.
    # 2000 batch 간격마다 출력하도록 할게요.
    log_interval = 2000

    # 앞에서 정의한 DataLodaer로 데이터를 batch 별로 하나씩 불러옵니다.
    for idx, (labels, texts, offsets) in enumerate(dataloader):
        # Optimizer의 `zero_grad()` 함수로 optimizer에 있는 파라미터를 모두 0으로 초기화해줍니다.
        optimizer.zero_grad()

        # 앞에서 제작한 커스텀 모델에 데이터의 입력을 넣어줘서 나오는 출력을 받습니다.
        # offset을 같이 넣는 이유는, 앞서 model에서 정의한 것처럼 Embeddingbag 레이어에서 offset이 필요하기 때문이죠.
        outs = model(texts, offsets)

        # 모델이 예측한 레이블을 알아내기 위해 가장 값이 큰 요소의 인덱스를 받아옵니다.
        predicts = torch.argmax(outs, dim=-1)

        # Loss 함수로 실제 레이블과 예측 값의 차이를 구합니다.
        loss = criterion(outs, labels)

        # Loss 함수의 `backward()` 함수로 back propagation을 진행하여 각 파라미터의 gradient를 구합니다.
        loss.backward()

        # Optimizer의 `step()` 함수로 4번에서 구한 gradient를 가지고 모델의 각 파라미터를 업데이트해 줍니다.
        optimizer.step()

        # 이번 batch에서 예측 레이블과 실제 레이블이 같은 것의 개수를 더해줍니다.
        train_acc += (predicts == labels).sum().item()

        # 데이터의 개수만큼 더합니다.
        train_count += labels.size(0)

        # 2000만큼의 간격마다 accuracy를 계산합니다.
        if idx % log_interval == 0 and idx > 0:
            # 모델이 정확히 레이블을 예측한 것의 개수를 전체 데이터 수로 나눠서 모델의 학습 데이터에 대한 accuracy를 구합니다.
            print('| epoch {:3d} | {:5d}/{:5d} batches | accuracy {:8.3f}'.format(epoch, idx, len(dataloader), train_acc / train_count))

    scheduler.step()


def evaluate(dataloader):
    model.eval()
    val_acc = 0
    val_count = 0
    val_acc_items = []

    # 자동 미분 기능을 끔으로써 연산과 메모리 사용의 효율성을 높여줍니다.
    with torch.no_grad():
        # 검증 DataLoaer에서 batch 별로 하나씩 모델에 학습시킵니다.
        for idx, (labels, texts, offsets) in enumerate(dataloader):
            # 앞에서 제작한 커스텀 모델에 데이터의 입력을 넣어줘서 나오는 출력을 받습니다.
            outs = model(texts, offsets)

            # 모델이 예측한 레이블을 알아내기 위해 가장 값이 큰 요소의 인덱스를 받아옵니다.
            predicts = torch.argmax(outs, dim=-1)

            # 이번 batch에서 예측 레이블과 실제 레이블이 같은 것의 개수를 더해줍니다.
            acc_item = (labels == predicts).sum().item()
            val_acc_items.append(acc_item)

            val_count += labels.size(0)
            # 모델이 정확히 레이블을 예측한 것의 개수를 전체 데이터 수로 나눠서 모델의 검증 데이터에 대한 accuracy를 구합니다.
            val_acc = np.sum(val_acc_items) / val_count
    return val_acc


total_acc = 0
for epoch in range(1, epochs + 1):
    # 학습 DataLoader로 학습을 진행합니다.
    train(train_dataloader, epoch)
    # 검증 DataLoader로 검증을 합니다.
    acc_val = evaluate(val_dataloader)

    # 이번 epoch에서의 검증 결과(정확도)가 처음부터 이제까지의 검증 결과(정확도)보다 좋으면 가장 좋은 검증 결과고 업데이트합니다.
    if total_acc < acc_val:
        total_acc = acc_val
        # 만약에 가장 좋은 accuracy일 때의 모델 또는 모델의 파라미터를 저장하면 이 logic을 여기서 처리해주면 됩니다.
        # 실습에서는 하지 않고 과제에서 등장할 예정입니다.

    print('-' * 60)
    print('| end of epoch {:3d} | valid accuracy {:8.3f} '.format(epoch, total_acc))
    print('-' * 60)

test_dataset = CustomDataset(TEST_AG_NEWS_PATH, train=False)
test_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_batch)
acc_val = evaluate(test_dataloader)
print('-' * 59)
print('test accuracy {:8.3f} '.format(acc_val))
print('-' * 59)
