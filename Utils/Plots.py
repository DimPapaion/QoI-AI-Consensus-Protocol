import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np


def show_Image(args, dataset, num, preds):
    if args.dataset_name == "SVHN":
        y_true = dataset.labels[num]
        plt.imshow(dataset.data[num].T)
    else:
        y_true = dataset.targets[num]

        plt.imshow(dataset.data[num])
    if preds is None:
        if args.dataset_name == "SVHN":
            plt.title('True Label: {}'.format(y_true))
        else:
            plt.title('True Label: {}'.format(dataset.classes[y_true]))
    else:
        predicted = preds[num]
        if args.dataset_name == "SVHN":
            plt.title('True Label: {} \n Predicted Label: {}'.format(y_true,
                                                                     dataset.classes[predicted]))
        else:
            plt.title('True Label: {} \n Predicted Label: {}'.format(dataset.classes[y_true],
                                                                     dataset.classes[predicted]))
    plt.show()


def show_dataset(args, dataset):
    figure = plt.figure(figsize=(20, 10))
    cols, rows = 5, 5
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        if args.dataset_name == "SVHN":
            real_label = dataset.labels[sample_idx]  # .numpy()
        else:
            real_label = dataset.targets[sample_idx]  # .numpy()
        if args.dataset_name == "SVHN":
            plt.title("Idx: {} True Label: {}".format(sample_idx, real_label))
            plt.axis("off")
        else:

            plt.title("Idx: {} True Label: {}".format(sample_idx, dataset.classes[real_label]))
            plt.axis("off")
        if (args.dataset_name == 'F-MNIST' or  args.dataset_name =='CIFAR100'):
            plt.imshow(img.squeeze(), cmap='gray')
        elif args.dataset_name == "SVHN":
            plt.imshow(img.T)
        else:
            plt.imshow(np.transpose(img, (1, 2, 0)), interpolation="antialiased")
    plt.show()

def show_batch(dl, batch_size):

  def show_images(images, batch_size=batch_size):
      fig, ax = plt.subplots(figsize=(8, 8))
      ax.set_xticks([]); ax.set_yticks([])
      ax.imshow(make_grid((images.detach()[:batch_size]), nrow=8).permute(1, 2, 0))
      plt.show()

  for indx, (data, label) in enumerate(dl):
      show_images(data, batch_size)
      break

