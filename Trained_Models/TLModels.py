import torch.nn as nn
import torchvision

class Identity(nn.Module):
  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class MyVgg16(object):
    def __init__(self, dataset_name, trained=True):
        self.trained = trained
        self.VGG16 = torchvision.models.vgg16(pretrained=self.trained)
        self.dataset_name = dataset_name

    def construct(self, n_classes):

        if self.dataset_name == "MNIST":
            self.VGG16.features[0] = nn.Conv2d(1, 64, 3, 1, 1)

            for i in range(len(self.VGG16.features)):
                if i > 25:
                    self.VGG16.features[i] = Identity()

            # VGG16.avgpool = nn.AvgPool2d(kernel_size= 2)
            self.VGG16.avgpool = Identity()
            self.VGG16.classifier = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, n_classes)
            )
            if self.trained:
                for param in self.VGG16.features[1:26].parameters():
                    param.requires_grad = False

        elif self.dataset_name == "CIFAR10" or self.dataset_name == "CIFAR100":
            input_features = self.VGG16.classifier[6].in_features
            self.VGG16.classifier[6] = nn.Linear(input_features, n_classes)
            if self.trained:
                for param in self.VGG16.features.parameters():
                    param.requires_grad = True

        return self.VGG16

class MyDesnet161(object):
  def __init__(self, dataset_name, trained=True):
    self.trained = trained
    self.des161 = torchvision.models.densenet161(pretrained=self.trained)
    self.dataset_name = dataset_name
  def construct(self, n_classes):

    if self.dataset_name == "MNIST":
      self.des161.features.conv0 = nn.Conv2d(1, 96, 7, 2, 3)
      self.des161.classifier = nn.Linear(in_features=2208, out_features=n_classes)
      if self.trained:
        for param in self.des161.parameters():
            param.requires_grad = False
        for param in self.des161.features.conv0.parameters():
            param.requires_grad = True
        for param in self.des161.classifier.parameters():
            param.requires_grad = True

    elif self.dataset_name == "CIFAR10" or self.dataset_name == "CIFAR100":
      self.des161.classifier = nn.Linear(in_features=2208, out_features=n_classes)
      if self.trained:
        for param in self.des161.parameters():
            param.requires_grad = False
        for param in self.des161.classifier.parameters():
            param.requires_grad = True

    return self.des161

class MyGoogleNet(object):
  def __init__(self, dataset_name, trained=True):
    self.trained = trained
    self.google = torchvision.models.googlenet(pretrained=False, aux_logits=False)
    self.dataset_name = dataset_name
  def construct(self, n_classes):

    if self.dataset_name == "MNIST":
      self.google.conv1.conv = nn.Conv2d(1, 64, 7, 2, 3)
      self.google.fc = nn.Linear(in_features=1024, out_features=n_classes)
      if self.trained:
        for param in self.google.parameters():
            param.requires_grad = False
        for param in self.google.conv1.conv.parameters():
            param.requires_grad = True
        for param in self.google.fc.parameters():
            param.requires_grad = True

    elif self.dataset_name == "CIFAR10" or self.dataset_name == "CIFAR100":
      self.google.fc = nn.Linear(in_features=1024, out_features=n_classes)
      if self.trained:
        for param in self.google.parameters():
            param.requires_grad = False
        for param in self.google.fc.parameters():
            param.requires_grad = True

    return self.google

class MySqueezeNet(object):
  def __init__(self, dataset_name):
    self.sq = torchvision.models.squeezenet1_1(pretrained=True)
    self.dataset_name = dataset_name
  def construct(self, n_classes):

    if self.dataset_name == "MNIST":
      self.sq.features[0] = nn.Conv2d(1, 64, 3, 2)
      self.sq.classifier[1] = nn.Conv2d(512, n_classes, 1, 1)
      for param in self.sq.parameters():
          param.requires_grad = False
      for param in self.sq.features[0:2].parameters():
          param.requires_grad = True
      for param in self.sq.classifier.parameters():
          param.requires_grad = True

    elif self.dataset_name == "CIFAR10" or self.dataset_name == "CIFAR100":
      self.sq.classifier[1] = nn.Linear(in_features=512, out_features=n_classes)
      for param in self.sq.parameters():
          param.requires_grad = False
      for param in self.sq.classifier.parameters():
          param.requires_grad = True

    return self.sq

class MyEfficientNet(object):
  def __init__(self, dataset_name, trained=True):
    self.trained = trained
    self.ef = torchvision.models.efficientnet_b7(pretrained=self.trained)
    self.dataset_name = dataset_name
  def construct(self, n_classes):

    if self.dataset_name == "MNIST":
      self.ef.features[0][0] = nn.Conv2d(1, 64, 3, 2, 1)
      self.ef.classifier[1] = nn.Linear(in_features=2560, out_features=n_classes)
      if self.trained:
        for param in self.ef.parameters():
            param.requires_grad = False
        for param in self.ef.features[0].parameters():
            param.requires_grad = True
        for param in self.ef.classifier.parameters():
            param.requires_grad = True

    elif self.dataset_name == "CIFAR10" or self.dataset_name == "CIFAR100":
      self.ef.classifier[1] = nn.Linear(in_features=2560, out_features=n_classes)
      if self.trained:
        for param in self.ef.parameters():
            param.requires_grad = False
        for param in self.ef.classifier.parameters():
            param.requires_grad = True

    return self.ef

class MyShuffleNetv2(object):
  def __init__(self, dataset_name, trained):
    self.trained = trained
    self.shuffle = torchvision.models.shufflenet_v2_x0_5(pretrained=self.trained)
    self.dataset_name = dataset_name
  def construct(self, n_classes):

    if self.dataset_name == "MNIST":
      self.shuffle.conv1[0] = nn.Conv2d(1, 24, 3, 2, 1)
      self.shuffle.fc = nn.Linear(in_features=1024, out_features=n_classes)
      if self.trained:
        for param in self.shuffle.parameters():
            param.requires_grad = False
        for param in self.shuffle.conv1[0].parameters():
            param.requires_grad = True
        for param in self.shuffle.fc.parameters():
            param.requires_grad = True

    elif self.dataset_name == "CIFAR10" or self.dataset_name == "CIFAR100":
      self.shuffle.fc = nn.Linear(in_features=1024, out_features=n_classes)
      if self.trained:
        for param in self.shuffle.parameters():
            param.requires_grad = False
        for param in self.shuffle.fc.parameters():
            param.requires_grad = True

    return self.shuffle

class MyResnet(object):
  def __init__(self, trained):
    self.trained = trained
    self.ResNet18 = torchvision.models.resnet18(pretrained= self.trained)

  def full_con(self, n_classes):
    self.ResNet18.fc = nn.Linear(in_features=512, out_features=n_classes)
    return self.ResNet18.fc



  def construct(self,n_classes, dataset_name):
    if dataset_name == 'MNIST':
      self.ResNet18.conv1 = nn.Conv2d(1, 64, 7, 2, 3)
      self.ResNet18.fc =self.full_con(n_classes=n_classes)
      if self.trained:
        for param in self.ResNet18.parameters():
          param.requires_grad = False
        for param in self.ResNet18.conv1.parameters():
          param.requires_grad = True
        for param in self.ResNet18.fc.parameters():
          param.requires_grad = True
    #VGG16.avgpool = nn.AvgPool2d(kernel_size= 2)
    #ResNet18.avgpool = Identity()
    else:
      self.ResNet18.fc =self.full_con(n_classes=n_classes)
      if self.trained:
        for param in self.ResNet18.parameters():
          param.requires_grad = True
        for param in self.ResNet18.fc.parameters():
          param.requires_grad = True


    return self.ResNet18

class MyMobileV2(object):
  def __init__(self, dataset_name='CIFAR10', trained=True):
    self.trained = trained
    self.mobilenet = torchvision.models.mobilenet_v2(pretrained=True)
    self.dataset = dataset_name

  def full_con(self, n_classes):
    classif = nn.Linear(in_features=1280, out_features=n_classes)
    return classif


  def construct(self,n_classes):
    if self.dataset == 'MNIST':
      self.mobilenet.features[0][0] = nn.Conv2d(1, 32, 3, 2, 1)
      self.mobilenet.classifier[1] =self.full_con(n_classes=n_classes)
      if self.trained:
        for param in self.mobilenet.features[1:].parameters():
          param.requires_grad = False
        for param in self.mobilenet.classifier.parameters():
          param.requires_grad = True
    #VGG16.avgpool = nn.AvgPool2d(kernel_size= 2)
    #ResNet18.avgpool = Identity()
    else:
      self.mobilenet.classifier[1] =self.full_con(n_classes = n_classes)
      if self.trained:
        for param in self.mobilenet.features.parameters():
          param.requires_grad = True


    return self.mobilenet