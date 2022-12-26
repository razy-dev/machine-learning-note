# tag::preset[]
import torch
import torch.nn as nn

torch.manual_seed(1)  # 난수 고정
torch.set_printoptions(threshold=10, linewidth=128)  # for printing
# end::preset[]

# tag::design[]
timesteps = 10  # 시점의 수. NLP에서는 보통 문장의 길이가 된다.
in_features = 4  # 입력의 차원. NLP에서는 보통 단어 벡터의 차원이 된다.
hidden_size = 1  # 은닉 상태의 크기. 메모리 셀의 용량이다.
# end::design[]

# tag::dataset[]
# (batch_size, time_steps, input_size)
inputs = torch.rand(timesteps, in_features)
print("inputs =", inputs.shape, inputs)
# end::dataset[]

# tag::parameters[]
# end::parameters[]

# tag::rnn[]
cell = nn.RNN(in_features, hidden_size)  # , batch_first=True)
outputs, _status = cell(inputs)
print("outputs", outputs.shape, outputs)
print("_status", _status.shape, _status)
# end::rnn[]
