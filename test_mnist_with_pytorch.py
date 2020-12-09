import matplotlib.pyplot as plt
import datasets.mnist as mnist
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(1, 4, 3, stride=1, padding=1, bias=False)
        self.fc = nn.Linear(14*14*4, 10)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        x = self.conv1(x) + self.conv2(x*x)
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 14*14*4)
        x = F.relu(self.fc(x))
        return x

def accuracy(y_pred, y_true):
    preds = torch.argmax(y_pred, dim=1)
    return (preds == y_true).float().mean()

def test_mnist_with_cov2d():
    (train_x, train_y), (test_x, test_y) = mnist.load_data(flatten=False, one_hot_label=False)
    val_x = train_x[50000:]
    val_y = train_y[50000:]
    train_x = train_x[:50000]
    train_y = train_y[:50000]
    net = Net()
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net.to(dev)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    epochs = 20
    batch_size = 200
    batches = len(train_x)//batch_size
    history = {"loss":[], "val_loss":[], "accuracy":[], "val_accuracy":[]}
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        for batch in range(batches):
            x_true = torch.tensor(train_x[batch*batch_size: (batch+1)*batch_size], dtype=torch.float).to(dev)
            y_true = torch.tensor(train_y[batch*batch_size: (batch+1)*batch_size], dtype=torch.long).to(dev)
            y_pred = net(x_true)
            loss = F.cross_entropy(y_pred, y_true)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            epoch_acc += accuracy(y_pred, y_true)
        train_loss = epoch_loss/batches
        train_acc = epoch_acc/batches
        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        # Do validation after each epoch
        net.eval()
        with torch.no_grad():
            x_true = torch.tensor(val_x, dtype=torch.float).to(dev)
            y_true = torch.tensor(val_y, dtype=torch.long).to(dev)
            y_pred = net(x_true)
            val_loss = F.cross_entropy(y_pred, y_true)
            val_acc = accuracy(y_pred, y_true)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        print(f"Epoch: {epoch+1} loss: {train_loss} accuracy:{train_acc} val_loss: {val_loss}, val_accuracy:{val_acc}")
    plt.plot(history["loss"], 'ro', label="Training loss")
    plt.plot(history["val_loss"], 'go',label="Validating loss")
    plt.plot(history["accuracy"], 'r', label="Training accuracy")
    plt.plot(history["val_accuracy"], 'g', label="Validating accuracy")
    plt.title('Training/Validating loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.show(block=True)

if __name__ == "__main__":
    test_mnist_with_cov2d()