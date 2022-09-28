import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as Transforms
import torch.utils.data
import sys
import matplotlib.pyplot as plt
from matplotlib import style

# instantiating functional variables
n_epochs = 10  # reduce if model overfits
learning_rate = 0.01
num_classes = 10
hidden_layer = 100
input_size = 28 * 28

n_train_total_examples = 50000
n_val_total_examples = 10000

# no need to normalize input data
transform = Transforms.Compose([Transforms.ToTensor()])

device = ("cuda" if torch.cuda.is_available() else "cpu")

print(f'device: {device}')


# construct model
class NeuralNet(nn.Module):
    def __init__(self, inputs_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.lin1 = nn.Linear(inputs_size, hidden_size)
        self.rel = nn.ReLU()
        self.lin2 = nn.Linear(hidden_size, output_size)

        self.flatten = nn.Flatten(start_dim=1)

        # dropout might cause validation data to have better results than training data
        # as in the validation stage dropout is disabled (therefore all neurons are available)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # x = x.view(-1, input_size)
        x = self.flatten(x)

        out = self.lin1(x)
        out = self.dropout(out)
        out = self.rel(out)
        out = self.lin2(out)
        return out


class ConvNet(nn.Module):
    def __init__(self, output_size):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, 5, padding=2)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.bn_1 = nn.BatchNorm2d(128)
        self.bn_2 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, output_size)

        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x):
        # convolutional layers
        out = F.leaky_relu_(self.bn_1(self.conv1(x)))
        out = self.pool(out)
        out = F.leaky_relu_(self.bn_2(self.conv2(out)))
        out = self.pool(out)

        # flatten
        # out = out.view(-1, 256 * 7 * 7)
        out = self.flatten(out)

        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return out  # NB softmax (for classes probability) already included in CrossEntropyLoss


def train_epoch():
    model.train()

    curr_train_correct, curr_examples, n_examples, n_train_correct, train_loss, curr_train_loss = 0, 0, 0, 0, 0, 0

    for index, (inputs, labels) in enumerate(train_loader):
        # forward pass
        inputs = inputs.to(device)
        labels = labels.to(device)

        y_predicted = model(inputs)
        loss = criterion(y_predicted, labels)

        # backward pass
        loss.backward()

        # step
        optimizer.step()
        optimizer.zero_grad()

        # test
        curr_train_loss += loss.item() * inputs.shape[0]

        _, pred = torch.max(y_predicted, 1)

        curr_train_correct += (labels == pred).sum().item()
        curr_examples += inputs.shape[0]

        if (index + 1) % 100 == 0:
            print(
                f'Epoch {epoch + 1}/{n_epochs}, index {index + 1}/{n_total_steps}, train acc: {curr_train_correct / curr_examples * 100: .4f}, train loss: {curr_train_loss / curr_examples:.4f}')
            n_train_correct += curr_train_correct
            train_loss += curr_train_loss
            n_examples += curr_examples
            curr_train_correct, curr_examples, curr_train_loss = 0, 0, 0

    return n_train_correct, train_loss, n_examples


def validation_epoch():
    model.eval()

    n_val_correct = 0
    n_examples = 0
    val_loss = 0

    for index, (inputs, labels) in enumerate(val_loader):
        # forward pass
        inputs = inputs.to(device)
        labels = labels.to(device)

        y_predicted = model(inputs)
        loss = criterion(y_predicted, labels)

        # test
        val_loss += loss.item() * inputs.shape[0]

        _, pred = torch.max(y_predicted, 1)

        n_val_correct += (labels == pred).sum().item()
        n_examples += inputs.shape[0]

    print(
        f'\nEpoch {epoch + 1}/{n_epochs}, val acc: {n_val_correct / n_examples * 100: .4f}, val loss: {val_loss / n_val_total_examples:.4f}\n')

    return n_val_correct, val_loss, n_examples


# prepare data
train_data = torchvision.datasets.MNIST("./data", train=True, transform=transform, download=True)
test_data = torchvision.datasets.MNIST("./data", train=False, transform=transform, download=True)

# divide into train and validation data
train_data, val_data = torch.utils.data.random_split(train_data, [n_train_total_examples, n_val_total_examples])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=100, shuffle=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False, num_workers=0)

if __name__ == '__main__':
    # create model
    network_type = input(
        "Enter 'n' if you want to use a feed forward neural network, 'c' for a convolutional neural network: ")

    if network_type == 'n':
        model = NeuralNet(inputs_size=input_size, hidden_size=hidden_layer, output_size=num_classes).to(device)
    elif network_type == 'c':
        model = ConvNet(num_classes).to(device)
    else:
        print("Not a valid input")
        sys.exit(0)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



    # training (and validation)
    n_total_steps = len(train_loader)
    train = []
    val = []

    # training
    for epoch in range(n_epochs):
        train.append(train_epoch())
        val.append(validation_epoch())



    # final training and validation results
    train_correct, train_loss = sum([x[0] for x in train]), sum([x[1] for x in train])
    val_correct, val_loss = sum([x[0] for x in val]), sum([x[1] for x in val])

    print(
        f'avg train acc: {train_correct / (n_train_total_examples * n_epochs) * 100 : .4f}, avg train loss: {train_loss / n_train_total_examples:.4f}')
    print(
        f'avg val acc: {val_correct / (n_val_total_examples * n_epochs) * 100: .4f}, avg val loss: {val_loss / n_val_total_examples:.4f}')



    # final test
    n_correct = 0
    n_total_examples = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        y_pred = model(inputs)

        _, prediction = torch.max(y_pred, 1)

        n_correct += (labels == prediction).sum().item()
        n_total_examples += labels.shape[0]

    print(f'\nFinal testing acc: {n_correct / n_total_examples * 100.0:.6f}')

    # plotting results

    style.use("ggplot")


    # following graph can be helpful to determine when we should stop training
    # before overfitting, that is when val_acc doesn't increase anymore/decreases
    # or its line is passed by training data accuracy


    def create_acc_loss_graph():
        plt.figure()

        ax1 = plt.subplot2grid((2, 1), (0, 0))
        ax2 = plt.subplot2grid((2, 1), (1, 0), sharex=ax1)

        time = [x + 1 for x in range(len(train))]
        accuracies = [x[0] / x[2] for x in train]
        losses = [x[1] / x[2] for x in train]

        val_accs = [x[0] / x[2] for x in val]
        val_losses = [x[1] / x[2] for x in val]

        ax1.plot(time, accuracies, label="train_acc")
        ax1.plot(time, val_accs, label="val_acc")
        ax1.legend(loc=2)
        ax2.plot(time, losses, label="train_loss")
        ax2.plot(time, val_losses, label="val_loss")
        ax2.legend(loc=2)
        plt.show()


    create_acc_loss_graph()
