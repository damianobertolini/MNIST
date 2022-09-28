import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as Transforms
import torch.utils.data
import matplotlib.pyplot as plt
import sys

# instantiating functional variables
n_epochs = 1  # increase for better results, but avoid overfitting
learning_rate = 0.01
num_classes = 10
hidden_layer = 100
input_size = 28 * 28  # MNIST images input dimension

transform = Transforms.Compose([Transforms.ToTensor()])


# construct model

class AutoEncoder(nn.Module):
    def __init__(self, inputs_size, hidden_size, output_size):
        # note that, as we're not doing classification,
        # there is actually no need to have output_size = num_classes, in this case it's just to maintain the parallelism with NNs and CNNs
        super(AutoEncoder, self).__init__()
        self.lin1 = nn.Linear(inputs_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, output_size)
        self.lin3 = nn.Linear(output_size, hidden_size)
        self.lin4 = nn.Linear(hidden_size, inputs_size)

        self.rel = nn.ReLU()

    def encode(self, x):
        out = self.lin1(x)
        out = self.rel(out)
        out = self.lin2(out)

        return out

    def decode(self, x):
        out = self.lin3(x)
        out = self.rel(out)
        out = self.lin4(out)

        return out

    def forward(self, x):
        # flatten
        x = x.reshape(-1, input_size)

        out = self.encode(x)

        out = self.decode(out)

        return out


class HalfCnnAutoEncoder(nn.Module):
    def __init__(self, inputs_size, output_size):
        super(HalfCnnAutoEncoder, self).__init__()
        # kernel size 5 was chosen over 3 due to improved results
        self.conv1 = nn.Conv2d(1, 16, 5)  # 1 is the channel size (1 for grayscale, 3 for coloured images)
        self.conv2 = nn.Conv2d(16, 32, 2)

        self.bn_1 = nn.BatchNorm2d(16)
        self.bn_2 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(32 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, output_size)
        self.fc3 = nn.Linear(output_size, 256)
        self.fc4 = nn.Linear(256, inputs_size)
        self.pool = nn.MaxPool2d(2, stride=2)

        self.flatten = nn.Flatten(start_dim=1)

    def encode(self, x):
        out = self.pool(F.leaky_relu(self.bn_1(self.conv1(x))))
        out = self.pool(F.leaky_relu(self.bn_2(self.conv2(out))))

        # flatten
        out = self.flatten(out)
        # out = out.view(-1, 32 * 5 * 5)

        out = F.leaky_relu(self.fc1(out))
        out = F.leaky_relu(self.fc2(out))

        return out

    def decode(self, x):
        out = F.leaky_relu(self.fc3(x))
        out = F.leaky_relu(self.fc4(out))
        return out

    def forward(self, x):
        out = self.encode(x)
        out = self.decode(out)

        return out


# NB it is preferable not to have a layer with too many elements as the NN wouldn't be able to extract key features, therefore struggling to learn
class CnnAutoEncoder(nn.Module):
    def __init__(self, output_size):
        super(CnnAutoEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 2, stride=2, padding=1)

        self.bn_1 = nn.BatchNorm2d(8)
        self.bn_2 = nn.BatchNorm2d(16)

        self.flatten = nn.Flatten(start_dim=1)

        self.fc1 = nn.Linear(16 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, output_size)

        self.fc3 = nn.Linear(output_size, 256)
        self.fc4 = nn.Linear(256, 16 * 7 * 7)

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(16, 7, 7))

        # useful formula for output dim of ConvTranspose2d: out = (x - 1)s - 2p + d(k - 1) + op + 1
        # where x is the input spatial dimension and out the corresponding output size, s is the stride, d the dilation, p the padding, k the kernel size, and op the output padding
        self.conv3 = nn.ConvTranspose2d(16, 8, 2, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(8, 1, 5, stride=2, output_padding=1)

    def encode(self, x):
        out = F.leaky_relu(self.bn_1(self.conv1(x)))
        out = F.leaky_relu(self.bn_2(self.conv2(out)))

        # flatten
        out = self.flatten(out)
        # out = out.view(-1, 16 * 7 * 7)

        out = F.leaky_relu(self.fc1(out))
        out = self.fc2(out)

        return out

    def decode(self, x):
        out = F.leaky_relu(self.fc3(x))
        out = F.leaky_relu(self.fc4(out))

        # unflatten
        out = self.unflatten(out)

        out = F.leaky_relu(self.bn_1(self.conv3(out)))
        out = self.conv4(out)

        return out

    def forward(self, x):
        out = self.encode(x)
        out = self.decode(out)

        return out


if __name__ == '__main__':
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    print(f'device: {device}\n')

    # prepare data
    train_data = torchvision.datasets.MNIST("./data", train=True, transform=transform, download=True)
    test_data = torchvision.datasets.MNIST("./data", train=False, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False, num_workers=0)

    # AUTOENCODER
    encoder_type = input(
        "Enter 'a' if you want to use a simple autoencoder, 'h' for an encoder with CNNs only in encoding part, 'c' for an autoencoder which uses convolutional neural networks: ")

    if encoder_type == 'a':
        model = AutoEncoder(input_size, hidden_layer, num_classes).to(device)
    elif encoder_type == 'h':
        model = HalfCnnAutoEncoder(input_size, num_classes).to(device)
    elif encoder_type == 'c':
        model = CnnAutoEncoder(num_classes).to(device)
    else:
        print("Not a valid input")
        sys.exit(0)

    criterion = nn.MSELoss()

    # NB if amsgrad=True is not used, Adam optimizer might not be stable and have a sudden increase in weights values
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)

    # training
    n_total_steps = len(train_loader)
    for epoch in range(n_epochs):
        for index, (inputs, _) in enumerate(train_loader):
            inputs = inputs.to(device)

            # forward pass
            y_predicted = model(inputs)

            if encoder_type == 'c':
                # output is not flattened, so input need to stay as original
                pass
            else:
                # flatten inputs to have the same shape as y_predicted
                inputs = inputs.view(-1, 28 * 28)

            loss = criterion(y_predicted, inputs)

            # backward pass
            loss.backward()

            # step
            optimizer.step()
            optimizer.zero_grad()

            if index % 100 == 0:
                print(f'Epoch {epoch + 1}/{n_epochs}, index {index + 100}/{n_total_steps}, loss: {loss:.4f}')

    # "test"
    examples = iter(test_loader)
    inputs, _ = examples.next()

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(inputs[i][0], cmap='gray')
    plt.show()

    inputs = inputs.to(device)

    y_pred = model(inputs)

    y_pred = y_pred.to("cpu")  # python requires data to be on the cpu in order to plot it
    y_pred = y_pred.detach()

    # rebuild 2d structure for plotting
    y_pred = y_pred.reshape(-1, 1, 28, 28)

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(y_pred[i][0], cmap='gray')
    plt.show()
