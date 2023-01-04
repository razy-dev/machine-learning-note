import matplotlib.pyplot as plt
import torch

from bck.engine.dataset import dataset, load

torch.manual_seed(1)  # 난수 고정
torch.set_printoptions(threshold=5, linewidth=128)  # for printing

# =================================================================#

a, b, c, d, s = dataset(load(size=100), window_size=3, output_size=1, normalizer='gradient2')
# for i in range(len(a)):
#     print(a[i], b[i])

plt.plot(torch.tanh(b.squeeze()))
plt.show()

if __name__ == "__main__":
    pass
