import numpy as np
from Train_Test.Synthetic_Version.Voting_Ensemble import *


class Individualized_Ens(VotingClass):

    def __init__(self, Agents):
        self.Agents = Agents


    def matrix_shape(self, Predictions):
        Samples, Classes = Predictions.shape[1], Predictions.shape[2]
        return Samples, Classes

    def combine_rule(self ,probs, targets, vote_type, axis = None):
        msg = "Vote type must be 'Average', 'Median', 'W_Average or 'W_Median' but got {} instead.".format(vote_type)

        if vote_type == "Average":
            prob = np.average(probs, axis = 0)

        elif vote_type == "Median":
            prob = np.median(probs, axis= 0)

        elif vote_type == "W_Average":
            prob = self.weight_avg_rule(outputs= probs, targets=targets)

        elif vote_type == "W_Median":
            prob = self.weighted_median_rule(outputs = probs, targets=targets)

        else:
            raise ValueError(msg)
        return prob

    def distr_Confid_agreement(self, Predictions, targets, vote_type, decision_type, acc=None):

        self.Samples, self.Classes = self.matrix_shape(Predictions=Predictions)
        All_agent = np.zeros([self.Agents, self.Samples, self.Classes])
        if acc is None:

            acc_score = self.probs_(outputs = Predictions, targets = targets, return_="Acc")
            print(acc_score)
        else:

            acc_score = acc
            print(acc_score)
        total_cost = []
        for agent in range(self.Agents):

            current_agent = agent
            cost_per_sample = []
            for sample in range(self.Samples):
                cost_per_class=[]
                for clas in range(self.Classes):

                    current_clas = clas
                    # current_confs = []
                    current_conf = Predictions[current_agent][sample][current_clas]
                    current_confs = np.empty([])
                    current_confs = np.vstack([current_confs, current_conf])

                    for diff_agent in range(self.Agents):
                        diff_conf = Predictions[diff_agent][sample][current_clas]
                        if decision_type == "normal":
                            if current_conf < diff_conf  :  # and acc_score[diff_agent] > np.average(acc_score):
                                current_confs = np.vstack([current_confs, diff_conf])
                        else:
                            if current_conf * acc_score[current_agent] < diff_conf * acc_score[
                                diff_agent]:  # and acc_score[diff_agent] > np.average(acc_score):
                                current_confs = np.vstack([current_confs, diff_conf])
                            # current_confs.append(diff_conf)

                    # current_confs = np.stack(current_confs)
                    cost_per_class.append((len(current_confs)-1)/self.Agents)
                    decision_conf = self.combine_rule(probs=current_confs, targets=targets, vote_type=vote_type)
                    del current_confs
                    All_agent[current_agent][sample][current_clas] = decision_conf
                cost_per_sample.append(sum(cost_per_class)/self.Classes)
            total_cost.append(sum(cost_per_sample)/self.Samples)

        return All_agent, total_cost
