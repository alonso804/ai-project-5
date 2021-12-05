import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import math
from torch.utils.data import DataLoader, random_split
from CustomDataset import LungsDataset
from CNN import CNN, CNN_bn, CNN_bnd, CNN_bnd1


def show_img(imgs, name, size=3, color=True):
    color_m = 'jet'
    if color == False:
        color_m = 'gray'
    print('*******************' + name + '*********************')
    img_numbers = imgs.shape[0]
    rows = cols = math.ceil(np.sqrt(img_numbers))

    fig = plt.figure(figsize=(rows * size, cols * size))
    for i in range(0, rows * cols):
        fig.add_subplot(rows, cols, i + 1)
        if i < img_numbers:
            plt.imshow(imgs[i].detach(), cmap='gray')
        plt.show()


def train(model, optimizer, loss_fn, num_epochs, train_loader):
    loss_vals = []
    running_loss = 0.0

    total_step = len(train_loader)

    list_loss = []
    list_time = []
    j = 0

    for epoch in range(num_epochs):
        for i, (img, label) in enumerate(train_loader):
            img = img.to(device)
            label = label.to(device)

            output = model(img)
            loss = loss_fn(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            list_loss.append(loss.item())
            list_time.append(j)
            j += 1

            if (i + 1) % 100 == 0:
                print('Epoch [{} / {}], Step [{} / {}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    return loss_vals


def main():
    batch_size = 64
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    dataset = LungsDataset(
        csv_file="./data.csv",
        root_dir="./",
        transform=transforms.ToTensor())

    size = len(dataset)
    train_size = int(7 / 10 * size)
    train_set, test_set = random_split(
        dataset, [train_size, size - train_size])

    train_loader = DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=True)

    # Hyperparameters
    num_classes = 4  # Constant
    learning_rate = 0.001
    num_epochs = 20

    model = CNN(num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    img, _ = dataset[17007]
    show_img(img, 'Temp')
    print(train_loader)
    # for i, (img, label) in enumerate(train_loader):
    # print(label)

    # Train
    # loss_results = train(model, optimizer, loss_fn, num_epochs, train_loader)


if __name__ == "__main__":
    main()
