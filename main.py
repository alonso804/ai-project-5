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


def train(model, optimizer, loss_fn, num_epochs, train_loader, device):
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

            # print(img)
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


def accuracy(model, test_loader, device):
    with torch.no_grad():
        correct = 0
        total = 0

        for img, label in test_loader:
            output = model(img)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

        print("Accuracy: {}%".format(100 * correct / total))


def main():
    batch_size = 64
    device = torch.device('cpu')
    print(device)

    dataset = LungsDataset(
        csv_file="./data.csv",
        root_dir="./",
        transform=torchvision.transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ]))

    size = len(dataset)
    train_size = int(7 / 10 * size)
    train_set, test_set = random_split(
        dataset, [train_size, size - train_size])

    train_loader = DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=True)

    if torch.cuda.is_available():
        def map_location(storage, loc): return storage.cuda()
    else:
        map_location = 'cpu'

    # Hyperparameters
    num_classes = 4  # Constant
    learning_rate = 0.001
    num_epochs = 50

    model = CNN().to(device)
    # Get the model trained
    model.load_state_dict(torch.load(
        "./Results/CNN.txt", map_location=map_location))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train
    # loss_results = train(model, optimizer, loss_fn, num_epochs, train_loader, device)

    # print("Saving ...")
    # torch.save(model.state_dict(), "./Results/CNN.txt")
    # print("Saved ...")

    accuracy(model, test_loader, device)


if __name__ == "__main__":
    main()
