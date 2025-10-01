import time

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
# pytorch модель
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from neuralnet.Features import CCELogits, Relu
from neuralnet.Layers import Conv2D, Dense
from neuralnet.Loaders import AsyncCupyDataLoader
from neuralnet.Optimizers import Adam
# моя модель
from neuralnet.core_gpu import NeuralNetwork


class TorchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Второй сверточный блок
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Полносвязные слои
        self.fc1 = nn.Linear(16 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.permute(0, 2, 3, 1)  # → BHWC
        x = x.reshape(x.shape[0], -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def check_vram():
    # Всего памяти GPU
    total_mem = cp.cuda.Device().mem_info[1] / 1024 ** 3  # в гигабайтах
    # Свободная память GPU
    free_mem = cp.cuda.Device().mem_info[0] / 1024 ** 3
    # Использованная память GPU
    used_mem = total_mem - free_mem

    return used_mem


def train_torch(torch_loader):
    losses_torch = []
    for epoch in range(10):
        for xb, yb in torch_loader:
            xb, yb = xb.cuda(), yb.cuda()
            optimizer.zero_grad()
            out = torch_model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

        torch_model.eval()
        with torch.no_grad():
            # predict на всех данных
            all_out = torch_model(torch.from_numpy(x_train.reshape(-1, 1, 28, 28)).cuda())
            full_loss = criterion(all_out, torch.from_numpy(y_train).cuda())
            losses_torch.append(full_loss.item())
        torch_model.train()
    return losses_torch


torch_model = TorchCNN().cuda()
# ---------- Подготовка данных ----------
x_train = (np.load("images.npy") / 255).astype(np.float32)
y_train = np.load("labels.npy")
# мой DataLoader
train_loader = AsyncCupyDataLoader(x_train, y_train, batch_size=1000)

# pytorch DataLoader
torch_loader = DataLoader(TensorDataset(
    torch.from_numpy(x_train.reshape(-1, 1, 28, 28)),
    torch.from_numpy(y_train)
), batch_size=1000, shuffle=True)

# ---------- NeuralNetwork (моя реализация) ----------
w1 = torch_model.conv1.weight.detach().cpu().numpy()
w1 = w1.reshape(w1.shape[0], -1)
b1 = torch_model.conv1.bias.detach().cpu().numpy()

w2 = torch_model.conv2.weight.detach().cpu().numpy()
w2 = w2.reshape(w2.shape[0], -1)
b2 = torch_model.conv2.bias.detach().cpu().numpy()

w3 = torch_model.fc1.weight.detach().cpu().numpy().T
b3 = torch_model.fc1.bias.detach().cpu().numpy()

w4 = torch_model.fc2.weight.detach().cpu().numpy().T
b4 = torch_model.fc2.bias.detach().cpu().numpy()

configs = [
    {"input_dim": (28, 28, 1), "out_channels": 8, "layer": Conv2D, "act": Relu, "pooling_func": "max", "lr": 0.001,
     "kernel_size": (3, 3), "padding": "full", "stride": 1, "pooling_shape": (2, 2), "pooling_stride": 2, "w": w1,
     "bias": b1},
    {"out_channels": 16, "layer": Conv2D, "act": Relu, "pooling_func": "max", "lr": 0.001,
     "kernel_size": (3, 3), "padding": "full", "stride": 1, "pooling_shape": (2, 2), "pooling_stride": 2, "w": w2,
     "bias": b2},
    {"neurons": 128, "layer": Dense, "act": Relu, "lr": 0.001, "w": w3, "bias": b3},
    {"neurons": 10, "layer": Dense, "lr": 0.001, "w": w4, "bias": b4}
]

model = NeuralNetwork(configs, loss_func=CCELogits(), optimizer=Adam())

used_mem = check_vram()

t0 = time.perf_counter()
model.train(train_loader, epochs=10, early_stop=10, x_test=x_train, y_test=y_train, logs=True)
t_mine = time.perf_counter() - t0

used_mem_net = check_vram()

losses_mine = model.losses
losses_mine = [loss.get() for loss in losses_mine]
print(f"Использовано Cupy NN vram: {used_mem_net - used_mem:.2f} GB")
# ---------- PyTorch ----------

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(torch_model.parameters(), lr=0.001)

t0 = time.perf_counter()
used_mem = check_vram()
losses_torch = train_torch(torch_loader)
used_mem_net = check_vram()

print(f"Использовано Torch vram: {used_mem_net - used_mem:.2f} GB")
t_torch = time.perf_counter() - t0


# ---------- График ----------
plt.plot(losses_mine, label="CuPy NN")
plt.plot(losses_torch, label="PyTorch")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Сходимость на MNIST")
plt.savefig("compare_loss.png")

print(f"Время (10 эпох) - CuPy NN: {t_mine:.2f}s | PyTorch: {t_torch:.2f}s")

# моя реализация
x_batch = cp.asarray(x_train[:1], dtype=cp.float32)
t0 = time.perf_counter()
out = model.forward(x_batch, train=True)
t_forward_mine = time.perf_counter() - t0
model.backward(out)  # разогрев
t1 = time.perf_counter()
model.backward(out)
t_backward_mine = time.perf_counter() - t1
# pytorch
xb = torch.from_numpy(x_train[:1].reshape(1, 1, 28, 28)).cuda()
yb = torch.from_numpy(y_train[:1]).cuda()
t0 = time.perf_counter()
out = torch_model(xb)
t_forward_torch = time.perf_counter() - t0
t1 = time.perf_counter()
loss = criterion(out, yb)
loss.backward()
t_backward_torch = time.perf_counter() - t1
print(
    f"Forward+Backward (1 sample) - CuPy NN forward & backward: {t_forward_mine:.5f}s,  {t_backward_mine:.5f}s |"
    f" PyTorch: {t_forward_torch:.5f}s,  {t_backward_torch:.5f}s")
