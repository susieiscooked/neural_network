#Define a Convolutional Neural Network CNN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(logger, trainloader) -> Net:
    net = Net()

    # Define a loss function and a optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train the CNN network
    for epoch in range(2):  # loop over the dataset multiple times using for loop
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print  the statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                logger.info(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    logger.info('Finished Training')
    return net


def run_inference(classes, testloader, trained_model, logger):
    correct = 0
    total = 0
    # prepare to count predictions for each class {classes that performed well, and the classes that did not perform well}
    correct_pred_per_label = {classname: 0 for classname in classes}
    total_pred_per_label = {classname: 0 for classname in classes}

    # since we're not training, we don't calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = trained_model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            logger.debug('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                  for j in range(4)))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred_per_label[classes[label]] += 1
                total_pred_per_label[classes[label]] += 1
    return correct,total,correct_pred_per_label, total_pred_per_label