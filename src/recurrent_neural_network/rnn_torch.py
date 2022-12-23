# tag::preset[]
import torch
import torch.nn as nn

torch.manual_seed(1)  # 난수 고정
torch.set_printoptions(threshold=10, linewidth=128)  # for printing
# end::preset[]

# tag::design[]
timesteps = 10  # 시점의 수. NLP에서는 보통 문장의 길이가 된다.
in_features = 4  # 입력의 차원. NLP에서는 보통 단어 벡터의 차원이 된다.
hidden_size = 8  # 은닉 상태의 크기. 메모리 셀의 용량이다.
# end::design[]

# tag::dataset[]
inputs = np.random.random((timesteps, in_features))  # 입력에 해당되는 2D 텐서
hidden_state = np.zeros((hidden_size,))  # 초기 은닉 상태는 0(벡터)로 초기화 은닉 상태의 크기 hidden_size로 은닉 상태를 만듬.
print("inputs =", inputs.shape, inputs)
print("hidden_state =", hidden_state.shape, hidden_state)
# end::dataset[]

# tag::parameters[]
Wx = np.random.random((hidden_size, in_features))  # (8, 4)크기의 2D 텐서 생성. 입력에 대한 가중치.
Wh = np.random.random((hidden_size, hidden_size))  # (8, 8)크기의 2D 텐서 생성. 은닉 상태에 대한 가중치.
b = np.random.random((hidden_size,))  # (8,)크기의 1D 텐서 생성. 이 값은 편향(bias).
print("Wx =", Wx.shape, Wx)
print("Wh =", Wh.shape, Wh)
print("b =", b.shape, b)
# end::parameters[]

# tag::rnn[]
cell = nn.RNN(input_size, hidden_size, batch_first=True)
# end::rnn[]
