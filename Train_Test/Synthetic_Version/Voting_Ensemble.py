import numpy as np
from sklearn.metrics import accuracy_score

class VotingClass(object):
    def __init__(self, preds, targets):
        self.outputs = preds
        self.targets = targets

    def _probs(self, outputs, targets, return_ , print_acc=False):
        n_classifiers, n_samples = outputs.shape[0], outputs.shape[1]
        preds = np.zeros([n_classifiers, n_samples])
        prob = np.zeros([n_classifiers, n_samples])
        acc_score = np.zeros([n_classifiers])
        for i in range(len(outputs)):
            pred1 = np.array([outputs[i][j].argmax() for j in range(len(outputs[i]))])
            prob1 = np.array([outputs[i][j].max() for j in range(len(outputs[i]))])
            preds[i] = preds[i] + pred1
            prob[i] = prob[i] + prob1
            acc_score[i] = accuracy_score(targets, pred1)

            if print_acc:
                print("Classifier {} has accuracy of {:.2f}".format(i, accuracy_score(targets, pred1 ) *100))

        if return_ == "Acc":
            return acc_score
        elif return_ == "All":
            return preds, prob, acc_score
        else:
            return preds, prob

    def probs_(self, outputs, targets, return_ , print_acc=False):

        pred = []
        prob = []
        n_classifiers, n_samples = outputs.shape[0], outputs.shape[1]
        acc_score = np.zeros([n_classifiers])
        for j in range(outputs.shape[1]):
            pred_classif =[]
            prob_classif =[]
            for i in range(outputs.shape[0]):
                max = np.max(outputs[i][j])
                pred_classif.append(np.argmax(outputs[i][j]))
                prob_classif.append(np.max(outputs[i][j]))


            pred.append(np.stack(pred_classif))
            prob.append(np.stack(prob_classif))

        pred = np.stack(pred).T
        prob = np.stack(prob).T

        if targets is not None:
            for i in range(len(pred)):
                acc_score[i] = accuracy_score(targets, pred[i])

                if print_acc:
                    print("Classifier {} has accuracy of {:.2f}".format(i, accuracy_score(targets, pred[i] ) *100))

        if return_ == "Acc":

            return acc_score
        elif return_ == "All":
            return pred, prob, acc_score
        else:
            return pred, prob

    def sum_rule(self):
        return sum(self.outputs)

    def avg_rule(self):
        return np.average(self.outputs, axis=0)

    def median_rule(self):

        return np.median(self.outputs, axis=0)

    def max_rule(self):

        return np.max(self.outputs, axis =0)

    def min_rule(self):

        return np.min(self.outputs, axis =0)

    def majority_rule(self):
        predictions, weights =  self.probs_(outputs = self.outputs, targets=self.targets, return_=False)
        maj = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x)),
            axis=1,
            arr=predictions.T
        )
        return maj

    def weighted_median_rule(self):
        predictions, weights_prob = self.probs_(outputs = self.outputs, targets=self.targets, return_=False)
        predictions, weights_prob = predictions.T, (weights_prob / sum(weights_prob)).T
        sorted_idx = np.argsort(predictions, axis=1)
        weight_cdf = np.array([np.cumsum(weights_prob[i][sorted_idx[i]]) for i in range(len(weights_prob))])

        median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
        median_idx = median_or_above.argmax(axis=1)

        median_models = sorted_idx[np.arange(self.targets.shape[0]), median_idx]
        reverse_outputs = self.outputs.T
        preds = reverse_outputs[: ,np.arange(self.targets.shape[0]), median_models].T
        return preds

    def weight_avg_rule(self):
        preds, weights= self.probs_(outputs=self.outputs, targets=self.targets, return_ = False)
        # weights = weights / sum(weights)
        pred = self.average(outputs1=self.outputs ,weight= weights)
        return pred

    def average(self, outputs1, weight):

        for i in range(outputs1.shape[1]):
            for j in range(outputs1.shape[0]):
                outputs1[j][i] = outputs1[j][i] * weight[j][i]

        return sum(outputs1)  # / len(outputs1)


    def find_weights(self, outputs):
        weights = np.zeros((len(outputs), len(outputs[1])))
        for i in range(len(outputs)):
            model_output = np.empty((0, len(outputs[1])), float)
            for pred in outputs[i]:

                model_output= np.append(model_output, pred.max())
            weights[i] = weights[i] + model_output

        return weights