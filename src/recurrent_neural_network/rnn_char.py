import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Dataset
input_str = 'apple'
label_str = 'pple!'
char_vocab = sorted(list(set(input_str + label_str)))
vocab_size = len(char_vocab)
print(f"문자 집합의 크기 = {vocab_size} :", char_vocab)

char_to_index = dict((c, i) for i, c in enumerate(char_vocab))  # 문자에 고유한 정수 인덱스 부여
print("char_to_index =", char_to_index)

index_to_char = {}
for key, value in char_to_index.items():
    index_to_char[value] = key
print("index_to_char =", index_to_char)

x_data = [char_to_index[c] for c in input_str]
print(f"{input_str} =", x_data)

x_one_hot = [np.eye(vocab_size)[x] for x in x_data]
print(f"'{input_str}' x_one_hot =", x_one_hot)

X = torch.Tensor(np.array(x_one_hot))
print(f"'{input_str}' tensor =", X.shape, X)

y_data = [char_to_index[c] for c in label_str]
print(f"{label_str} =", y_data)

Y = torch.Tensor(y_data).long()
print(f"'{label_str}' tensor =", Y.shape, Y)

# Model
input_size = vocab_size  # 입력의 크기는 문자 집합의 크기
hidden_size = 5
output_size = 5
learning_rate = 0.1

rnn = nn.RNN(input_size, hidden_size)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), learning_rate)

# Training
for i in range(10):
    optimizer.zero_grad()
    outputs, _status = rnn(X)
    cost = criterion(outputs, Y)
    cost.backward()
    optimizer.step()

    result = outputs.data.numpy().argmax(axis=1)
    print(outputs, Y, cost, result)
