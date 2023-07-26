import torch, torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import ipdb

#############################
#
# This file references code from:
# 1. https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
# 2. https://github.com/features/copilot
#
#############################


def digit_to_onehot(digit : int):
    x = torch.zeros(10)
    x[digit] = 1
    return x

def image_to_vector(image):
    x = ToTensor()(image)
    return torch.flatten(x)

#60,000 examples
mnist_data = datasets.MNIST('./data', 
                            download=True, 
                            transform=image_to_vector, 
                            target_transform=digit_to_onehot) 

train_dataset, test_dataset = torch.utils.data.random_split(mnist_data, [50000, 10000])

ipdb.set_trace()

train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                         batch_size=16,
                                         shuffle=True)

test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                         batch_size=10000,
                                         shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

class MnistClassifier(nn.Module):
    def hidden_layer(self, in_dim, hidden_dim, out_dim):
        linear1 = nn.Linear(in_dim, hidden_dim, bias=False)
        nn.init.kaiming_normal_(linear1.weight)

        linear2 = nn.Linear(hidden_dim, out_dim, bias=True)
        nn.init.kaiming_normal_(linear2.weight)

        return nn.Sequential(
            linear1,
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            linear2,
            nn.Softmax(dim=1)
        )

    def __init__(self):
        super().__init__()
        self.preprocess = nn.Flatten()
        self.output = self.hidden_layer(784, 128, 10)

    def forward(self, x):
        x = self.preprocess(x)
        return self.output(x)
    
model = MnistClassifier().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)


epochs = 2
train_size = len(train_dataloader)
train_loss = np.zeros(train_size * epochs)
for e in range(epochs):
    model.train()
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss[e * train_size + batch] = loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{train_size:>5d}]")

    test_size = len(test_dataloader.dataset)
    model.eval()
    for batch, (X, y) in enumerate(test_dataloader):
        X, y = X.to(device), y.to(device)

        break

    # Compute prediction error
    with torch.no_grad():
        pred = model(X)
        test_loss = loss_fn(pred, y).item()
        num_correct = (pred.argmax(1) == y.argmax(1)).sum().item()

    print(f"test loss: {test_loss:>7f}")
    print(f"num correct: {num_correct:>5d}/{test_size:>5d}")

ipdb.set_trace()
# Plot the training loss
N = 100
smoothed_loss = np.convolve(train_loss, np.ones(N)/N, mode='valid')
plt.plot(smoothed_loss)
plt.axhline(test_loss)
plt.show()