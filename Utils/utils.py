import torch
from sklearn.metrics import average_precision_score
from Train_Test.DNN_Version.BaseTraining import *
import Datasets.Datasets


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

def build_dataset(args):
  CD = Datasets.Datasets.CustomDatasets(args)
  dataset = CD._load_Dataset()
  loaders = CD.make_loaders()
  return dataset, loaders

def get_optim(args, model):
  msg = "--optimizer  must be 'ADAM' or 'SGD' but got {} instead.".format(args.optimizer)
  if args.optimizer == "ADAM":
    return torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay= 5e-4)
  elif args.optimizer == "SGD":
     return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
  else:
    raise ValueError(msg)

def build_training(args, ModelParams,loaders):
  best_models_acc = []
  device = get_default_device()
  for model in ModelParams:
    print("----Model {} is starting the Training Process-----".format(str(model)))
    optim = get_optim(args, ModelParams[model])
    fit = BaseClassifier(args, model = ModelParams[model], model_name = model, train_dl = loaders['Train_load'],
                           valid_dl = None, test_dl = loaders['Test_load'], optim=optim, device=device)
    best_acc = fit.train_model()
    best_models_acc.append(best_acc)

  print("Accuract scores: ", best_models_acc)
  return best_models_acc

def build_Centralized_Inference(args, ModelParams, loaders, dataset ):
  device = get_default_device()
  inference = VotingClassifier(args, models = ModelParams, device = device,
                               test_dl=loaders['Test_load'], dataset = dataset['Test'],
                               targets = dataset['Test'].targets)
  preds_total_C10, preds_C10, results_C10 = inference.predict()

  preds_total_C10 = np.stack(preds_total_C10)
  acc_score = results_C10["Accuracy"].tolist()
  targets = np.stack(dataset['Test'].targets)

  return preds_total_C10, preds_C10, results_C10



