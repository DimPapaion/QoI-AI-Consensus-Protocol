import torch
from sklearn.metrics import average_precision_score

def get_default_device():
  if torch.cuda.is_available():
    print("Run on Cuda")
    return torch.device("cuda")
  else:
    return torch.device("cpu")

def to_device(data, device):
  if isinstance(data, (list, tuple)):
    return [to_device(x, device) for x in data]
  return data.to(device, non_blocking= True)


def Accuracy(y_true, y_scores):
  Acc_scores = 0.0
  for i in range(y_true.shape[0]):
    Acc_scores += average_precision_score(y_true = y_true[i], y_score = y_scores[i])
  return Acc_scores


def some_infos(loader):
  dataiter = iter(loader)
  images, labels = dataiter.next()
  print("Type of DataLoader: {} \nShape of Images: {} \nShape of Labels: {}".format(
      type(images),
      images.shape,
      labels.shape,
      ))
  for idx, (data, target) in enumerate(loader):
    real_label = torch.max(target, 1)[1].data.squeeze()
    print("Batch Labels:  " , real_label)
    break