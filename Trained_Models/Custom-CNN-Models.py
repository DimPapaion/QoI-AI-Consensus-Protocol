import torch.nn as nn

class FCnn(nn.Module):
  def __init__(self, num_channels, num_classes):
    super(FCnn, self).__init__()

    self.Conv = nn.Sequential(
        nn.Conv2d(num_channels, 16, kernel_size=(5,5), stride = 1, padding = 2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(16,32, kernel_size=(5,5), stride = 1, padding = 2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2,2)),
    )
    self.Fully_Con = nn.Sequential(
        #nn.Linear(64 * 7 * 7, 256),
        #nn.ReLU(),
        #nn.Dropout(p=0.5),
        nn.Linear(32 * 7 * 7, num_classes),
        )



  def forward(self, x):
    x = self.Conv(x)
    x = x.reshape(x.shape[0], -1)
    x = self.Fully_Con(x)
    return x


class Cifar10CNN(nn.Module):
    def __init__(self, n_classes):
        super(Cifar10CNN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 256 x 4 x 4
            nn.BatchNorm2d(256),

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, n_classes))

    def forward(self, x):
        return self.network(x)

class FCnn_Cifar(nn.Module):
  def __init__(self, num_channels, num_classes):
    super(FCnn_Cifar, self).__init__()

    self.Conv = nn.Sequential(
        nn.Conv2d(num_channels, 16, kernel_size=(5,5), stride = 1, padding = 2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(16,32, kernel_size=(5,5), stride = 1, padding = 2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2,2)),
    )
    self.Fully_Con = nn.Sequential(
        nn.Linear(32 * 8 *8 , 256),
        nn.ReLU(),
        #nn.Dropout(p=0.5),
        nn.Linear(256, num_classes),
        )



  def forward(self, x):
    x = self.Conv(x)
    x = x.reshape(x.shape[0], -1)
    x = self.Fully_Con(x)
    return x