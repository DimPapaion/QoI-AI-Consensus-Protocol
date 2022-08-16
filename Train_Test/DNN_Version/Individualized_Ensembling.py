from Train_Test.DNN_Version.BaseTraining import VotingClassifier
from Train_Test.Synthetic_Version.Voting_Ensemble import *
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import accuracy_score


class Individualized_EnsDNN(VotingClassifier, VotingClass):

    def __init__(self, args, models, device,
                               test_dl, dataset,
                               targets, weights=None):
        super().__init__(args, models, device,
                               test_dl, dataset,
                               targets)
        self.Agents = len(self.Models)

        if weights == None:
            _ ,_ ,self.acc_score = self.predict()
        else:
            self.acc_score = weights


    def matrix_shape(self, Predictions):
        Samples, Classes = Predictions.shape[1], Predictions.shape[2]
        return Samples, Classes

    def combine_rule(self ,probs, targets, vote_type, axis = None):
        msg = "Vote type must be 'Average', 'Median', 'W_Average or 'W_Median' but got {} instead.".format(vote_type)

        if vote_type == "Average":
            prob = np.average(probs, axis = axis)

        elif vote_type == "Median":
            prob = np.median(probs, axis= axis)

        elif vote_type == "W_Average":
            prob = self.weight_avg_rule(outputs= probs, targets=targets)

        elif vote_type == "W_Median":
            prob = self.weighted_median_rule(outputs = probs, targets=targets)

        else:
            raise ValueError(msg)
        return prob

    def indiv_forward(self, image, decision, decision_type, vote, weights=None):
        outputs = [
            F.softmax(model.eval()(image), dim=1) for model in self.Models.values()
        ]
        outputs = torch.stack(outputs)
        pred = np.array([torch.Tensor.cpu(t).detach().numpy() for t in outputs])

        if decision == "Confidence":
            _preds = self.distr_Confid_agreement(Predictions = pred, targets = self.targets ,decision_type=decision_type, combine_type= vote)
        elif decision == "Accuracy":
            _preds = self.distr_acc_Agreement(Predictions = pred, targets = self.targets, combine_type= vote)
        return torch.tensor(_preds)


    def distr_agreement(self, vote="Average" ,decision="Confidence" ,decision_type='normal'
                        ,centralized_aggregation = False, central_vote = None,
                        weights=None, return_OHE=False, return_acc=True, dataset_name=None):

        y_pred = np.empty((self.Agents, 0 ,self.n_classes), float)
        y_pred_centr = np.empty((0, self.n_classes), float)
        modelnames = [modeln for modeln in self.Models.keys()]
        acc = 0.0
        results = []
        self.vote = vote
        print("---- Individualized Ensembling is starting the testing process-----")
        with torch.no_grad():
            # model.eval()
            for img, label in tqdm(self.test_load):

                img = img.to(self.device)
                output = self.indiv_forward(img, decision, decision_type, vote, weights)
                if centralized_aggregation:
                    output1 = [F.softmax(torch.Tensor.float(output[i]), dim=1) for i in range(output.shape[0])]
                    output1 = torch.stack(output1)
                    outputs = self.combination(outputs=output1, vote=central_vote, weights=None, is_weight_norm=False)

                    if (vote == "median" or vote == "max" or vote == "min"):
                        y_pred_centr = np.append(y_pred_centr, torch.Tensor(outputs).detach().numpy(), axis=0)
                    else:
                        y_pred_centr = np.append(y_pred_centr, torch.Tensor.cpu(outputs).detach().numpy(), axis=0)

                if (vote == "median" or vote == "max" or vote == "min"):
                    y_pred = np.append(y_pred, torch.Tensor(output).detach().numpy(), axis=1)
                else:
                    y_pred = np.append(y_pred, torch.Tensor.cpu(output).detach().numpy(), axis=1)

                # acc += self.Accuracy(torch.Tensor.cpu(label).detach().numpy(), torch.Tensor.cpu(outputs).detach().numpy())
            predic = []
            for i in range(self.Agents):
                predic.append(np.array([np.argmax(y_pred[i][j]) for j in range(len(y_pred[i]))]))

            if return_acc:
                predic = np.array(predic)
                for i in range(self.Agents):
                    print("\n Classifier {} finished with accuracy of {:.2f}%".format(i,
                                                                                      accuracy_score(
                                                                                          np.array(self.targets),
                                                                                          predic[i]) * 100))
                    acc, pre, rec, f1, clasRep = self.calculate_metrics(y_true=np.array(self.targets),
                                                                        name=modelnames[i], y_pred=predic[i])

                    results.append(
                        {
                            'Classifier': i,
                            'Model_Name': modelnames[i],
                            'Accuracy': acc * 100,
                            'Precision': pre * 100,
                            'Recall': rec * 100,
                            'F1': f1 * 100,
                            'Classification_Report': clasRep})
            if centralized_aggregation:
                results = None
                predic_cent = np.array([np.argmax(y_pred_centr[i]) for i in range(len(y_pred_centr))])
                print(
                    "Centralized Aggregation with vote rule: {} finished with accuracy of {:.2f}%".format(central_vote,
                                                                                                          accuracy_score(
                                                                                                              np.array(
                                                                                                                  self.targets),
                                                                                                              predic_cent) * 100))

        results = pd.DataFrame(results)
        if centralized_aggregation == False and return_OHE:

            return y_pred, results
        else:
            return predic, predic_cent, y_pred

    def distr_Confid_agreement(self, Predictions, targets, decision_type, combine_type):

        self.Samples, self.Classes = self.matrix_shape(Predictions=Predictions)
        final_preds = np.zeros([self.Agents, self.Samples, self.Classes])
        for agent in range(self.Agents):

            current_agent = agent
            for sample in range(self.Samples):

                for clas in range(self.Classes):

                    current_clas = clas
                    # current_confs = []
                    current_conf = Predictions[current_agent][sample][current_clas]
                    # current_confs.append(current_conf)
                    current_confs = np.empty([])
                    current_confs = np.vstack([current_confs, current_conf])
                    for diff_agent in range(self.Agents):
                        diff_conf = Predictions[diff_agent][sample][current_clas]

                        if decision_type == "normal":

                            if current_conf < diff_conf:
                                current_confs = np.vstack([current_confs, diff_conf])
                        elif decision_type == "advance":
                            if current_conf * self.acc_score[current_agent] < diff_conf * self.acc_score[diff_agent]:
                                current_confs = np.vstack([current_confs, diff_conf])
                        else:
                            raise ValueError(
                                "Please insert a valid decision_type value. Got {} but expected 'normal' or 'advance'".format(
                                    decision_type))
                            # current_confs.append(diff_conf)

                    decision_conf = self.combine_rule(probs=current_confs, targets=targets, vote_type=combine_type,
                                                      axis=0)
                    del current_confs
                    final_preds[current_agent][sample][current_clas] = decision_conf

        return final_preds

    def distr_acc_Agreement(self, Predictions, targets, combine_type):
        All_agents = []

        for agent in range(self.Agents):

            current_agent = agent
            acc_agent = self.acc_score[current_agent]
            current_agt = []
            current_agt.append(Predictions[current_agent])
            for diff_agent in range(self.Agents):

                if acc_agent < self.acc_score[diff_agent]:
                    current_agt.append(Predictions[diff_agent])

            s = np.stack(current_agt)

            agent_decision = self.combine_rule(probs=s, targets=targets, vote_type=combine_type, axis=0)

            All_agents.append(agent_decision)

        All_agents = np.stack(All_agents)
        return All_agents