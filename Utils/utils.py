import torch
from sklearn.metrics import average_precision_score
from Train_Test.DNN_Version.BaseTraining import BaseClassifier, VotingClassifier
from Train_Test.DNN_Version.Individualized_Ensembling import *
from Train_Test.DNN_Version.QoI_Consensus import *
from Train_Test.Synthetic_Version.Individualized_Ens import *
from Train_Test.Synthetic_Version.Consensus import *
from Datasets.SyntheticDataset import *
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



def build_Synthetic_dataset(args):
  data = SyntheticDataset_v3(args)
  targets_OHE, target = data.generate_labels()

  preds = data.construct_clf(args, max_percent=None)

  m_preds = data.reform_preds_v3(preds, percent_start=0.78, step=0.21, percent_stop=0.92)

  return m_preds, targets_OHE, target



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
  preds_total_, preds_, results_ = inference.predict()

  preds_total_C10 = np.stack(preds_total_)


  return preds_total_, preds_, results_

def build_Centralized_Ensemble(args, ModelParams, loaders, dataset):
  device = get_default_device()
  inference = VotingClassifier(args, models=ModelParams, device=device,
                               test_dl=loaders['Test_load'], dataset=dataset['Test'],
                               targets=dataset['Test'].targets)

  # Weight Median Aggregation Rule.
  preds_OHE_wmed, med = inference.weight_med_predict(return_OHE=True)
  preds_OHE_wmed = preds_OHE_wmed.T
  vote_OHE_wMed = np.array([np.argmax(preds_OHE_wmed[i]) for i in range(len(preds_OHE_wmed))])
  print("Weighted Median Voting Acc {:.2f}%".format(accuracy_score(dataset['Test'].targets, vote_OHE_wMed) * 100))

  #Weight Average Aggregation Rule.
  preds_OHE_wAVG = inference.weighted_avg_predict(vote="weight_avg", return_OHE=True)
  vote_OHE_wAVG = np.array([np.argmax(preds_OHE_wAVG[i]) for i in range(len(preds_OHE_wAVG))])
  print("Weighted Averege Voting Acc {:.2f}%".format(accuracy_score(dataset['Test'].targets, vote_OHE_wAVG) * 100))

  #Average Aggregation Rule.
  preds_OHE_avg = inference.weighted_avg_predict(weights=None, vote='average', return_OHE=True)
  vote_OHE_avg = np.array([np.argmax(preds_OHE_avg[i]) for i in range(len(preds_OHE_avg))])
  print("Average Voting Acc {:.2f}%".format(accuracy_score(dataset['Test'].targets, vote_OHE_avg) * 100))

  # Median Aggregation Rule.
  preds_OHE_med = inference.weighted_avg_predict(weights=None, vote='median', return_OHE=True)
  vote_OHE_med = np.array([np.argmax(preds_OHE_med[i]) for i in range(len(preds_OHE_med))])
  print("Median Voting Acc {:.2f}%".format(accuracy_score(dataset['Test'].targets, vote_OHE_med) * 100))

  # Max Aggregation Rule.
  preds_OHE_max = inference.weighted_avg_predict(weights=None, vote='max', return_OHE=True)
  vote_OHE_max = np.array([np.argmax(preds_OHE_max[i]) for i in range(len(preds_OHE_max))])
  print("Max Voting Acc {:.2f}%".format(accuracy_score(dataset['Test'].targets, vote_OHE_max) * 100))

  # Min Aggregation Rule.
  preds_OHE_min = inference.weighted_avg_predict(weights=None, vote='min', return_OHE=True)
  vote_OHE_min = np.array([np.argmax(preds_OHE_min[i]) for i in range(len(preds_OHE_min))])
  print("Min Voting Acc {:.2f}%".format(accuracy_score(dataset['Test'].targets, vote_OHE_min) * 100))

  #Majority Voting Aggregation Rule
  preds_OHE_maj = inference.majority_voting_predict()
  print("Majority Voting Acc {:.2f}%".format(accuracy_score(dataset['Test'].targets, preds_OHE_maj) * 100))
  return

def build_Individualized_Ensemble(args, ModelParams, loaders, dataset, weights):
  device = get_default_device()
  IndivEns = Individualized_EnsDNN(args, models=ModelParams, device = device,
                               test_dl = loaders['Test_load'], dataset = dataset['Test'],
                               targets = dataset['Test'].targets, weights=weights)

  preds_all_NA, preds_NA, preds_all_OHE_NA = IndivEns.distr_agreement(vote=args.indiv_vote_type, decision='Confidence',
                                                                       decision_type=args.indiv_decision_type,
                                                                       centralized_aggregation=True,
                                                                       central_vote="weight_avg", return_OHE=True)
  return preds_all_NA, preds_NA, preds_all_OHE_NA

def build_QoI_Consensus(args,ModelParams, loaders, dataset, weights ):
  device = get_default_device()
  distr_cons3 = Distr_ConsensusDNN_v3(args, models=ModelParams, device = device,
                               test_dl = loaders['Test_load'], dataset = dataset['Test'],
                               targets = dataset['Test'].targets,  weights=weights)
  preds_total, acc_ = distr_cons3.conpredict(vote=args.QOI_vote, decision=args.QOI_decision, decision_type="Normal")
  return preds_total, acc_


def build_Centralized_Inference_Synthetic(preds, targets):

  vote = VotingClass(preds=preds, targets=targets)

  vote_pred, vote_prob, acc_score = vote.probs_(outputs=preds, targets=targets, return_="All", print_acc=True)


  return vote_pred, vote_prob, acc_score


def build_Centralized_Ens_Synthetic(preds, targets, vote_pred):
  vote = VotingClass(preds=preds, targets=targets)

  vote_OHE_avg = vote.avg_rule()
  preds_OHE_avg = np.array([np.argmax(vote_OHE_avg[i]) for i in range(len(vote_OHE_avg))])
  print("Average : ", accuracy_score(targets, preds_OHE_avg))


  vote_OHE_med = vote.median_rule()
  preds_OHE_med = np.array([np.argmax(vote_OHE_med[i]) for i in range(len(vote_OHE_med))])
  print("Median : ", accuracy_score(targets, preds_OHE_med))

  vote_OHE_wMed = vote.weighted_median_rule()
  preds_OHE_wMed = np.array([np.argmax(vote_OHE_wMed[i]) for i in range(len(vote_OHE_wMed))])
  print("W_Median : ", accuracy_score(targets, preds_OHE_wMed))


  vote_OHE_max = vote.max_rule()
  preds_OHE_max = np.array([np.argmax(vote_OHE_max[i]) for i in range(len(vote_OHE_max))])
  print("Max : ", accuracy_score(targets, preds_OHE_max))


  vote_OHE_min = vote.min_rule()
  preds_OHE_min = np.array([np.argmax(vote_OHE_min[i]) for i in range(len(vote_OHE_min))])
  print("Min : ", accuracy_score(targets, preds_OHE_min))


  vote_OHE_wAVG = vote.weight_avg_rule()
  preds_OHE_wAVG = np.array([np.argmax(vote_OHE_wAVG[i]) for i in range(len(vote_OHE_wAVG))])
  print("W_Average : ", accuracy_score(targets, preds_OHE_wAVG))

  maj = np.apply_along_axis(
    lambda x: np.argmax(np.bincount(x)),
    axis=0,
    arr=vote_pred,
  )
  print("Majority Voting: ", accuracy_score(maj, targets))

  return

def build_Individualized_Ensemble_Synthetic(args,agents, preds,targets, weight ):
  vote = VotingClass(preds=preds, targets=targets)
  distr_vote = Individualized_Ens(Agents=agents)
  Distr_Conf_NA = distr_vote.distr_Confid_agreement(Predictions=preds, targets=None, vote_type=args.indiv_vote_type,
                                                    decision_type=args.indiv_decision_type, acc=weight)
  acc_scoreC = vote.probs_(outputs=Distr_Conf_NA, targets=targets, return_="Acc", print_acc=True)
  return Distr_Conf_NA, acc_scoreC

def build_QoI_Consensus_Synthetic(args,preds,targets, weights ):
  cons = QoI_Consensus_(predictions=preds, acc_score=weights)

  All_samples = cons.consensus(decision=args.QOI_decision, decision_rule="Normal", vote_type=args.QOI_vote)
  All_samples = np.stack(All_samples)
  total = np.array([np.argmax(All_samples[i]) for i in range(len(All_samples))])

  print("Distributed consensus Average Acc {:.2f}%".format(accuracy_score(targets, total) * 100))
  return total