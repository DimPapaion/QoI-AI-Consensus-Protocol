from Utils.utils import *
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, average_precision_score
import numpy as np
import gc
import pandas as pd


class BaseClassifier(object):
    def __init__(self, args, model, model_name, train_dl, valid_dl, test_dl, device, optim):
        self.model = model
        self.optimizer = optim
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=200)
        self.train_loader = train_dl
        self.valid_loader = valid_dl
        self.test_loader = test_dl
        self.num_classes = args.n_classes
        self.epochs = args.epochs
        self.device = device
        self.model_name = model_name
        self.path = args.check_point_path + str(model_name) + ".pt"

    def Accuracy(self, y_true, y_scores):
        Acc_scores = 0.0
        for i in range(y_true.shape[0]):
            Acc_scores += average_precision_score(y_true=y_true[i], y_score=y_scores[i])
        return Acc_scores

    def calculate_metrics(self, name, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        pre = precision_score(y_true=y_true, y_pred=y_pred, average="macro")
        rec = recall_score(y_true=y_true, y_pred=y_pred, average="macro")
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
        clasRep = classification_report(y_true=y_true, y_pred=y_pred)
        print(
            "Classifier: {} finished with accuracy: {:4f}% ,precision: {:4f}%, recall: {:4f}%, f1 score: {:4f}%".format(
                name,
                acc * 100,
                pre * 100,
                rec * 100,
                f1 * 100))

        return acc, pre, rec, f1, clasRep

    def training(self, name=None):
        losst = 0.0
        acc = 0.0
        acc_f = 0.0
        y_pred = np.empty((0, self.num_classes), float)
        self.model = self.model.to(self.device)
        self.model.train(True)

        for data, target in tqdm(self.train_loader):

            # target = target.float()
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.criterion(output, target)

            losst += loss.item() * data.size(0)
            if name == "caltech":
                y_pred = np.append(y_pred, torch.Tensor.cpu(output).detach().numpy(), axis=0)
            else:
                acc += self.Accuracy(torch.Tensor.cpu(target).detach().numpy(),
                                     torch.Tensor.cpu(output).detach().numpy())

            loss.backward()

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            del data, target, output
            gc.collect()
            torch.cuda.empty_cache()

        num_samples = float(len(self.train_loader.dataset))
        total_loss_ = losst / num_samples
        if name == "caltech":
            predic = np.array([np.argmax(y_pred[i]) for i in range(len(y_pred))])
            total_acc_ = accuracy_score(self.train_loader.targets, predic)
        else:
            total_acc_ = acc.item() / num_samples
        return total_loss_, total_acc_

    def evaluation(self):

        self.model.eval()
        losst = 0.0
        acc = 0.0

        with torch.no_grad():
            for data, target in tqdm(self.valid_loader):
                # target = target.float()
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                loss = self.criterion(output, target)

                losst += loss
                acc += self.Accuracy(torch.Tensor.cpu(target).detach().numpy(),
                                     torch.Tensor.cpu(output).detach().numpy())

            num_samples = float(len(self.valid_loader))
            val_loss_ = losst.item() / num_samples
            val_acc_ = acc / num_samples

        return val_loss_, val_acc_

    def train_model(self, loader=False, name=None, return_all=False):
        tr_loss, tr_acc = [], []
        val_loss, val_acc = [], []

        best_acc = 0.0
        for epoch in range(self.epochs):
            print("-------Epoch {}/{}---------".format(epoch + 1, self.epochs))

            if loader:
                tr_loss_, tr_acc_ = self.training(name=name)
                val_loss_, val_acc_ = self.evaluation()

                tr_loss.append(tr_loss_), tr_acc.append(tr_acc_)
                val_loss.append(val_loss_), val_acc.append(val_acc_)

                print(
                    "Train Loss: {:.6f}, Train Accuracy: {:.4f}%, Validation Loss: {:.4f}, Validation Accuracy: {:.4f}%".format(
                        tr_loss_, tr_acc_, val_loss_, val_acc_ * 100))


            else:
                tr_loss_, tr_acc_ = self.training()

                tr_loss.append(tr_loss_), tr_acc.append(tr_acc_)
                print("Train Loss: {:.6f}, Train Accuracy: {:.4f}%".format(tr_loss_, tr_acc_ * 100))
                if best_acc < tr_acc_:
                    best_acc = tr_acc_
                    # if epoch +1 == self.epochs:

                    torch.save(self.model.state_dict(), self.path)
                    print("-----Model with name {} in epoch {} saved for inference in path: {} -----".format(
                        str(self.model_name), int(epoch + 1), str(self.path)))
        print(
            "Model with name {} finished with Best Accuracy score {:.4f}%".format(str(self.model_name), best_acc * 100))

        if loader:

            return ([tr_loss, tr_acc], [val_loss, val_acc])
        else:
            if return_all:
                return ([tr_loss, tr_acc])
            else:
                return best_acc

    def test_model(self, return_metrics=True):
        self.model.to(self.device)
        self.model.eval()
        losst = 0.0
        acc = 0.0

        y_pred = np.empty((0, self.num_classes), float)
        y_true = np.empty((0, self.num_classes), float)

        with torch.no_grad():
            for data, target in tqdm(self.test_loader):
                # target = target.float()
                data, target = data.to(self.device), target.to(self.device)

                bs, c, h, w = data.size()
                output = self.model(data.view(-1, c, h, w))
                loss = self.criterion(output, target)

                losst += loss
                acc += self.Accuracy(torch.Tensor.cpu(target).detach().numpy(),
                                     torch.Tensor.cpu(output).detach().numpy())

                y_pred = np.append(y_pred, torch.Tensor.cpu(output).detach().numpy(), axis=0)
                y_true = np.append(y_true, torch.Tensor.cpu(target).detach().numpy(), axis=0)

        num_samples = float(len(self.test_loader.dataset))
        test_loss = losst / num_samples
        test_acc = acc.item() / num_samples

        print("test_loss: {:.6f}, test_Accuracy: {:.4f}%".format(test_loss, test_acc))
        if return_metrics:
            return test_loss, test_acc, y_pred, y_true
        else:
            return output


class VotingClassifier(BaseClassifier):
    def __init__(self, args, models, device, test_dl, dataset, targets):
        self.Models = models
        self.device = device
        self.test_load = test_dl
        self.dataset = dataset
        self.targets = targets
        self.n_classes = args.n_classes
        self.weights_is_None = args.weights_isNone
        self.batch_size = args.batch_size

        if self.weights_is_None:
            self.is_Prob = True
        else:
            self.is_Prob = False

    def _predict(self, model, model_name, return_acc=False):
        y_pred = np.empty((0, self.n_classes), float)
        with torch.no_grad():
            model.eval().to(self.device)
            for idx, (img, label) in enumerate(self.test_load):
                img = img.to(self.device)

                if self.is_Prob:
                    m = nn.Softmax(dim=1)
                    output = m(model(img))
                else:
                    output = model(img)

                y_pred = np.append(y_pred, torch.Tensor.cpu(output).detach().numpy(), axis=0)

            preds = np.array([np.argmax(y_pred[i]) for i in range(len(y_pred))])
            weights = np.array([np.max(y_pred[i]) for i in range(len(y_pred))])
        return preds, weights, y_pred

    def predict(self, dataset_name=None, target=None, return_acc=True):
        preds_total, acc_score, results = [], [], []
        print("Models is", self.Models)
        for model in self.Models.keys():
            y_pred = np.empty((0, self.n_classes), float)
            y_true = np.empty((0, self.n_classes), float)
            acc = 0.0
            print("---- {} start the testing process-----".format(str(model)))
            self.Models[model].eval().to(self.device)
            with torch.no_grad():
                for img, label in tqdm(self.test_load):
                    img = img.to(self.device)
                    output = self.Models[model](img)

                    acc += self.Accuracy(torch.Tensor.cpu(label).detach().numpy(),
                                         torch.Tensor.cpu(output).detach().numpy())
                    y_pred = np.append(y_pred, torch.Tensor.cpu(output).detach().numpy(), axis=0)
                    y_true = np.append(y_true, torch.Tensor.cpu(label).detach().numpy(), axis=0)

                targets_ = np.array([np.argmax(y_true[i]) for i in range(len(y_true))])
                preds = np.array([np.argmax(y_pred[i]) for i in range(len(y_pred))])

                if dataset_name == "Caltech":
                    test_acc = acc.item() / len(self.dataset)
                    acc_score.append(test_acc)
                    print("\n Model  {} finished with accuracy of {:.2f}%".format(str(model), test_acc * 100))
                else:
                    acc_score.append(accuracy_score(np.array(self.targets), preds))
                    print("\n Model  {} finished with accuracy of {:.2f}%".format(str(model),
                                                                                  accuracy_score(np.array(self.targets),
                                                                                                 preds) * 100))
                    # results = self.make_results(results = results,model =model, y_true = np.array(self.targets), y_pred = preds)
                    acc, pre, rec, f1, clasRep = self.calculate_metrics(y_true=np.array(self.targets), name=model,
                                                                        y_pred=preds)
                    results.append(
                        {
                            # 'Classifier': p,
                            'Model_Name': model,
                            'Accuracy': acc * 100,
                            'Precision': pre * 100,
                            'Recall': rec * 100,
                            'F1': f1 * 100,
                            'Classification_Report': clasRep})
            preds_total.append(y_pred)
        results = pd.DataFrame(results)

        return preds_total, y_true, results

    def collect_preds(self, weights_init=None):
        _preds, _weights, _preds_OHE = [], [], []
        for model in tqdm(self.Models.keys()):
            preds, weights, preds_OHE = self._predict(model=self.Models[model], model_name=model)
            _preds.append(preds), _weights.append(weights), _preds_OHE.append(preds_OHE)
        _preds, _weights, _preds_OHE = np.array(_preds), np.array(_weights), np.array(_preds_OHE)

        if self.weights_is_None:
            return _preds.T, (_weights / sum(_weights)).T, _preds_OHE.T
        else:
            return _preds.T, weights_init, _preds_OHE.T

    def majority_voting_predict(self, test_preds=None, weights=None, safe=False, return_OHE=False):
        print("---- Majority Voting is starting the testing process-----")
        if test_preds is None:
            self.predictions, self.weights_prob, self.preds_OHE = self.collect_preds(weights_init=weights)
        else:
            self.predictions = test_preds

        maj = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x)),
            axis=1,
            arr=self.predictions
        )
        return maj

    def weight_med_predict(self, weights=None, safe=False, return_OHE=False):
        print("---- Weighted Median is starting the testing process-----")
        self.predictions, self.weights_prob, self.preds_OHE = self.collect_preds(weights_init=weights)

        sorted_idx = np.argsort(self.predictions, axis=1)

        if self.weights_is_None:
            weight_cdf = np.array(
                [np.cumsum(self.weights_prob[i][sorted_idx[i]]) for i in range(len(self.weights_prob))])
        else:
            weight_cdf = np.cumsum(weights[sorted_idx], axis=1)
        median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]

        if safe:
            l = []
            for i in range(median_or_above.shape[0]):
                if (len(median_or_above[i]) % 2 == 0):
                    l.append(int(len(median_or_above[i]) * 0.5))
                else:
                    l.append(int(len(median_or_above[i]) * 0.5) + 1)

            l = np.array(l)
            median_models = sorted_idx[np.arange(self.dataset.data.shape[0]), l]

            preds = self.predictions[np.arange(self.dataset.data.shape[0]), median_models]

            return preds
        else:
            median_idx = median_or_above.argmax(axis=1)

            median_models = sorted_idx[np.arange(self.dataset.data.shape[0]), median_idx]
            if return_OHE:
                preds = self.preds_OHE[:, np.arange(self.dataset.data.shape[0]), median_models]
                return preds, median_models
            else:
                preds = self.predictions[np.arange(self.dataset.data.shape[0]), median_models]
                return preds

    def average(self, outputs, weight=None):

        if self.weights_is_None:
            for i in range(len(outputs)):
                for j in range(len(outputs[i])):
                    outputs[i][j] = outputs[i][j] * weight[i][j]

            return sum(outputs) / len(outputs)
        else:
            return sum(outputs) / len(outputs)

    def find_weights(self, outputs, is_weight_norm=False):

        weights = np.zeros((len(outputs), len(outputs[1])))
        for i in range(len(outputs)):
            model_output = np.empty((0, len(outputs[1])), float)
            for pred in outputs[i]:
                pred = torch.Tensor.cpu(pred).detach().numpy()
                model_output = np.append(model_output, pred.max())
            weights[i] = weights[i] + model_output

        if is_weight_norm:
            for indx in range(len(weights)):
                weights[indx] = weights[indx] / sum(weights[indx])
            return weights
        else:
            return weights

    def combination(self, outputs, vote, weights, is_weight_norm):
        if vote == "weight_avg":
            if self.weights_is_None:
                weights = self.find_weights(outputs=outputs, is_weight_norm=is_weight_norm)
                proba = self.average(outputs=outputs, weight=weights)
                return proba
            else:
                proba = self.average(outputs * weights)
                return proba
        elif vote == 'sum':
            return sum(outputs)
        elif vote == 'max':
            pred = np.array([torch.Tensor.cpu(t).detach().numpy() for t in outputs])
            return torch.Tensor(np.max(pred, axis=0))
        elif vote == 'min':
            pred = np.array([torch.Tensor.cpu(t).detach().numpy() for t in outputs])
            return torch.Tensor(np.min(pred, axis=0))
        elif vote == 'average':
            return sum(outputs) / len(outputs)
        elif vote == 'median':
            pred = np.array([torch.Tensor.cpu(t).detach().numpy() for t in outputs])
            return torch.Tensor(np.median(pred, axis=0))
        else:
            return outputs

    def forward(self, image, vote, weights=None, is_weight_norm=False):
        outputs = [
            F.softmax(model.eval()(image), dim=1) for model in self.Models.values()
        ]
        outputs = torch.stack(outputs)
        # outputs = self.combination(outputs=outputs, vote=vote, weights= weights, is_weight_norm = is_weight_norm)
        if vote == "weight_avg":
            if self.weights_is_None:
                weights = self.find_weights(outputs=outputs, is_weight_norm=is_weight_norm)
                proba = self.average(outputs=outputs, weight=weights)
                return proba
            else:
                proba = self.average(outputs * weights)
                return proba
        elif vote == 'sum':
            return sum(outputs)
        elif vote == 'max':
            pred = np.array([torch.Tensor.cpu(t).detach().numpy() for t in outputs])
            return torch.Tensor(np.max(pred, axis=0))
        elif vote == 'min':
            pred = np.array([torch.Tensor.cpu(t).detach().numpy() for t in outputs])
            return torch.Tensor(np.min(pred, axis=0))
        elif vote == 'average':
            return sum(outputs) / len(outputs)
        elif vote == 'median':
            pred = np.array([torch.Tensor.cpu(t).detach().numpy() for t in outputs])
            return torch.Tensor(np.median(pred, axis=0))
        else:
            return outputs

    def weighted_avg_predict(self, dataset_name=None, weights=None, vote="weight_avg", return_acc=True,
                             return_OHE=False):

        y_pred = np.empty((0, self.n_classes), float)
        acc = 0.0
        print("---- {} is starting the testing process-----".format(vote))
        with torch.no_grad():
            # model.eval()
            for img, label in tqdm(self.test_load):

                img = img.to(self.device)
                outputs = self.forward(img, vote, weights)

                if (vote == "median" or vote == "max" or vote == "min"):
                    y_pred = np.append(y_pred, torch.Tensor(outputs).detach().numpy(), axis=0)
                else:
                    y_pred = np.append(y_pred, torch.Tensor.cpu(outputs).detach().numpy(), axis=0)

                acc += self.Accuracy(torch.Tensor.cpu(label).detach().numpy(),
                                     torch.Tensor.cpu(outputs).detach().numpy())
            predic = np.array([np.argmax(y_pred[i]) for i in range(len(y_pred))])

            if return_acc:
                if dataset_name == "Caltech":
                    test_acc = acc.item() / len(self.dataset)
                    print("\n {} finished with accuracy of {:.2f}%".format(vote, test_acc * 100))
                else:
                    print("\n {} finished with accuracy of {:.2f}%".format(vote,
                                                                           accuracy_score(np.array(self.targets),
                                                                                          predic) * 100))
                # print("\n {} finished with accuracy of {:.2f}%".format(vote,
                # accuracy_score(np.array(self.dataset.targets),
                # predic)*100))
        if return_OHE:
            return y_pred
        else:
            return predic