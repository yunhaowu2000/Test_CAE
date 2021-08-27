import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torch import nn, optim
from dataset import Mydata
from test_cae import TESTCAE
import visdom


def main():
    # [32, 1, 155, 256]
    test_train = Mydata('./dataset', True, transform=transforms.Compose([transforms.ToTensor()]),
                        download=False)
    test_train = DataLoader(test_train, batch_size=32, shuffle=True)

    test_test = Mydata('./dataset', False, transform=transforms.Compose([transforms.ToTensor()]),
                       download=False)
    test_test = DataLoader(test_test, batch_size=32, shuffle=True)

    x = iter(test_train).next()
    print("x:", x.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TESTCAE().to(device)
    criteon = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(model)

    viz = visdom.Visdom()

    for epoch in range(1000):

        for batchidx, x in enumerate(test_train):
            x = x.to(device)
            x_hat = model(x)
            loss = criteon(x_hat, x)

            # back prop
            optimizer.zero_grad()  # 梯度清零
            loss.backward()
            optimizer.step()  # 更新梯度

        print(epoch, 'loss:', loss.item())

        x = iter(test_train).next()
        with torch.no_grad():
            x_hat = model(x)
        viz.images(x, nrow=8, win='x', opts=dict(title='x'))
        viz.images(x_hat, nrow=8, win='x_hat', opts=dict(title='x_hat'))


if __name__ == '__main__':
    main()
