import numpy as np
from Voting_Ensemble import *


class Distr_Agents(VotingClass):

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

    def distr_Confid_agreement(self, Predictions, targets, combine_type, decision_type, acc=None):

        self.Samples, self.Classes = self.matrix_shape(Predictions=Predictions)
        All_agent = np.zeros([self.Agents, self.Samples, self.Classes])
        if acc is None:

            acc_score = self.probs_(outputs = Predictions, targets = targets, return_="Acc")
            print(acc_score)
        else:

            acc_score = acc
            print(acc_score)
        for agent in range(self.Agents):

            current_agent = agent
            for sample in range(self.Samples):

                for clas in range(self.Classes):

                    current_clas = clas
                    # current_confs = []
                    current_conf = Predictions[current_agent][sample][current_clas]
                    current_confs = np.empty([])
                    current_confs = np.vstack([current_confs, current_conf])

                    for diff_agent in range(self.Agents):
                        diff_conf = Predictions[diff_agent][sample][current_clas]
                        if decision_type == "Normal":
                            if current_conf < diff_conf  :  # and acc_score[diff_agent] > np.average(acc_score):
                                current_confs = np.vstack([current_confs, diff_conf])
                        else:
                            if current_conf * acc_score[current_agent] < diff_conf * acc_score[
                                diff_agent]:  # and acc_score[diff_agent] > np.average(acc_score):
                                current_confs = np.vstack([current_confs, diff_conf])
                            # current_confs.append(diff_conf)

                    # current_confs = np.stack(current_confs)
                    decision_conf = self.combine_rule(probs=current_confs, targets=targets, vote_type=combine_type)
                    del current_confs
                    All_agent[current_agent][sample][current_clas] = decision_conf

        return All_agent

    def distr_acc_Agreement(self, Predictions, targets, combine_type):
        All_agents = []
        acc_score = self.probs_(outputs=Predictions, targets=targets, return_="Acc")
        for agent in range(self.Agents):

            current_agent = agent
            acc_agent = acc_score[current_agent]
            current_agt = []
            current_agt.append(Predictions[current_agent])
            for diff_agent in range(self.Agents):

                if acc_agent < acc_score[diff_agent]:
                    current_agt.append(Predictions[diff_agent])

            s = np.stack(current_agt)

            agent_decision = self.combine_rule(probs=s, targets=targets, vote_type=combine_type, axis=0)

            All_agents.append(agent_decision)

        All_agents = np.stack(All_agents)
        return All_agents