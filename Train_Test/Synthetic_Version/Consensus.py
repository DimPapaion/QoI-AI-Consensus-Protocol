import numpy as np
import random




class QoI_Consensus_(object):
    def __init__(self, predictions, acc_score):
        self.agents, self.samples, self.Classes = predictions.shape[0], predictions.shape[1], predictions.shape[2]
        self.predictions = predictions
        self.acc_score = acc_score
        self.sum_agents = (self.agents * (self.agents - 1)) / 2
        self.final_preds = []
        self.Agent_scores = dict(list(enumerate([1 for i in range(self.agents)])))


    def init_leader(self):
        self.leader = random.randint(0, self.agents - 1)
        return self.leader

    def init_flags(self):
        self.is_leader = "Honest"
        self.leader_change = False
        self.is_finished = False
        self.is_check_prev_state = False

    def init_leader_specs(self, sample):
        leader_probs = self.predictions[self.leader][sample]
        leader_pred = np.argmax(leader_probs)

        del leader_probs
        return leader_pred

    def set_most_honest_leader(self):
        for key, value in self.Agent_scores.items():
            if max(self.Agent_scores.values()) == value:
                self.leader = key

    def vote_agent(self, sample):
        count_agent = 0
        for agent in range(self.agents):

            agent_pred = np.argmax(self.predictions[agent][sample])

            if self.leader_pred != agent_pred:
                count_agent += 1
        self.vote_percent = count_agent / self.agents
        return self.vote_percent

    def choose_leader(self):
        self.is_leader = "Faulty"
        while self.leader in self.leaders:
            self.leader = self.init_leader()
        self.leaders.append(self.leader)
        self.leaders_count += self.leader
        # print("Current count of leaders is:", self.leaders_count)
        self.leader_change = True

        # print("Leader is changed to:", self.leader)
        del self.leader_pred

    def check_vote(self ,sample, decision):
        if self.vote_percent > 0.51:
            self.Agent_scores[self.leader] = self.Agent_scores[self.leader] * 0.5
            # print(self.Agent_scores)
            # print("Ooops we must change leader!")
            self.leader_change = True
            self.check_leaders_(sample = sample, decision = decision)

        else:
            agent_decision_ = self.combine_agents( sample = sample, decision = decision)

            # print("Honest leader is: ", self.leader)
            # print("All fine here!!")
            self.final_preds.append(agent_decision_)
            # self.check_leaders_(sample = sample, decision = decision)
            self.is_finished = True

            del self.leader_pred


    def check_leaders_(self ,sample, decision):
        """
        Function to check if all agents are already choosed in order to serve as a leader.

        If it's true and no valid leader have found then we select the agent with the highest accurasy score
        in order to serve as leader in this iteration of consensus.

        As valid leader we define one which is supported by the majority of the agents.
        """
        if sample + 1 == self.samples:
            # self.leader = np.argmax(self.acc_score)
            self.set_most_honest_leader()

            agent_decision_ = self.combine_agents(sample, decision = decision)
            self.final_preds.append(agent_decision_)

            self.leader_change = False
            self.is_finished = True

        elif self.leaders_count >= self.sum_agents and sample + 1 != self.samples:

            self.Agent_decisions, self.conf_sample = self.individiual_decision(sample=sample)

            self.final_preds.append(0)
            # self.set_most_honest_leader()

            # print("We dont have consensus for: {} with value: {} ".format(self.conf_sample, self.final_preds[self.conf_sample]))
            self.leader_change = False
            self.is_finished = True


        else:
            self.choose_leader()
    def check_prev_state(self, sample):

        if self.is_check_prev_state and self.conf_sample + 1 == sample:
            decision__ = self.pow_decision()
            self.final_preds[self.conf_sample] = decision__
        elif sample == None:
            decision__ = self.pow_decision()
            self.final_preds[self.conf_sample] = decision__
        else:
            pass

    def leader_score_in_fork(self, Agents_decision, agents_scores):
        is_in_fork = False
        fork_scores_ = dict()
        leader_score =None
        for index, key in enumerate(Agents_decision.keys()):

            if isinstance(key, tuple):
                if (self.leader in key):
                    is_in_fork = True
                    score = 0
                    for agent in key:
                        score += agents_scores[agent]
                    fork_scores_[key] = (score / sum(agents_scores.values()))
                    leader_score = (score / sum(agents_scores.values()))
                else:
                    score = 0
                    for agent in key:
                        score += agents_scores[agent]
                    fork_scores_[key] = (score / sum(agents_scores.values()))

            elif isinstance(key, int):
                if self.leader == key:
                    is_in_fork = True
                    leader_score = (agents_scores[key] / sum(agents_scores.values()))
                    fork_scores_[key] = (agents_scores[key] / sum(agents_scores.values()))
                else:
                    fork_scores_[key] = (agents_scores[key] / sum(agents_scores.values()))
            else:
                is_in_fork = False
        return leader_score, fork_scores_, is_in_fork

    def pow_decision(self):

        while self.is_check_prev_state:
            leader_score_, fork_scores_, is_in_fork = self.leader_score_in_fork(Agents_decision=self.Agent_decisions
                                                                                ,agents_scores=self.Agent_scores)
            if is_in_fork:
                # print("Leader {} is inside the Fork....!!".format(self.leader))

                # print("We have:   ",fork_scores_)
                not_done = True

                if leader_score_ >=  0.5:
                    decision = self.predictions[self.leader][self.conf_sample]
                    self.is_check_prev_state = False
                    # print("done with using the long chain rule!!", self.conf_sample)
                    not_done = False
                    return decision
                    del self.Agent_decisions
                    break
                if not_done:
                    # sort_orders = sorted(self.Agent_scores.items(), key=lambda x: x[1], reverse=True)
                    # self.leader = sort_orders[0][0]
                    self.set_most_honest_leader()
                    decision = self.predictions[self.leader][self.conf_sample]
                    self.is_check_prev_state = False
                    # print("done with the most honest one!!:  ",self.conf_sample)
                    del self.Agent_decisions
                    return decision

            else:
                self.is_check_prev_state = True
                self.leader = self.init_leader()


    def individiual_decision(self, sample):
        Agents_decision = dict()
        self.is_check_prev_state = True
        Agent_list = []
        conf_sample = sample
        sort_orders = sorted(self.Agent_scores.items(), key=lambda x: x[1], reverse=True)
        for indx, (agent, count) in enumerate(sort_orders):

            # print("The {}st agent has votes: {}".format(agent, count))

            while agent not in Agent_list:
                candi_agt = []
                # print("Agent is: ", agent)
                agent_decision = np.argmax(self.predictions[agent][sample])
                for diff_agent in range(self.agents):
                    Agent_list.append(agent)
                    if agent_decision == np.argmax(self.predictions[diff_agent][sample]) and agent != diff_agent:
                        Agent_list.append(diff_agent)
                        candi_agt.append(diff_agent)

                if len(candi_agt) != 0:
                    candi_agents = (agent,) + tuple(candi_agt)
                    Agents_decision[(candi_agents)] = agent_decision
                else:
                    Agents_decision[agent] = agent_decision
        return Agents_decision, conf_sample


    def combine_agents(self ,sample, decision):
        self.Agent_scores[self.leader] += 20
        msg = "Please select a valid argument for the decision parameter. Got {} but expected 'both', 'acc' or 'class' instead.".format \
            (decision)

        self.leaders_pred = self.init_leader_specs(sample = sample)
        acc_leader = self.acc_score[self.leader]
        if self.decision_rule == "Normal":
            current_agt = []
            current_agt.append(self.predictions[self.leader][sample])
            for diff_agent in range(self.agents):
                if decision == "both":
                    if acc_leader < self.acc_score[diff_agent] and self.leaders_pred == np.argmax \
                            (self.predictions[diff_agent][sample]):
                        current_agt.append(self.predictions[diff_agent][sample])
                elif decision == "acc":
                    if acc_leader < self.acc_score[diff_agent]:
                        current_agt.append(self.predictions[diff_agent][sample])
                elif decision == "class":
                    if self.leaders_pred == np.argmax(self.predictions[diff_agent][sample]):
                        current_agt.append(self.predictions[diff_agent][sample])
                else:
                    raise ValueError(msg)
            s = np.stack(current_agt)
            if self.decision_type == "Average":
                agent_decision = np.average(s, axis = 0)
                # return agent_decision
            else:
                agent_decision = np.median(s, axis = 0)
                # return agent_decision
        elif self.decision_rule == "Confid":
            agent_decision = np.zeros([self.Classes])

            for clas in range(self.Classes):
                current_clas = clas
                current_conf = self.predictions[self.leader][sample][current_clas]
                current_confs = np.empty([])
                current_confs = np.vstack([current_confs, current_conf])

                for diff_agent in range(self.agents):
                    diff_conf = self.predictions[diff_agent][sample][current_clas]

                    if current_conf < diff_conf:
                        current_confs = np.vstack([current_confs, diff_conf])
                        # current_confs.append(diff_conf)

                    # current_confs = np.stack(current_confs)
                if self.decision_type == "Average":
                    decision_conf = np.average(current_confs)
                else:
                    decision_conf = np.median(current_confs)
                agent_decision[current_clas] = decision_conf

        return agent_decision

    def normal_operation(self, decision ,sample, count_leader=True):

        while (self.is_leader == "Honest" and self.is_finished == False):
            self.leader_pred = self.init_leader_specs(sample = sample)
            self.vote_percent = self.vote_agent(sample = sample)
            # print("Vote is:", self.vote_percent)

            if count_leader:
                self.leaders_count += self.leader

            self.check_vote(sample = sample, decision = decision)
            self.check_prev_state(sample=sample)

    def consensus(self, decision, decision_rule, vote_type):
        self.init_flags()
        self.leader = self.init_leader()
        self.decision_type = vote_type
        self.decision_rule = decision_rule


        for sample in range(self.samples):
            # print(self.Agent_scores)
            # print( "Data Sample:", sample)
            self.leaders = []
            self.leaders.append(self.leader)
            self.leaders_count = 0
            self.is_finished = False

            while self.is_finished == False:
                if self.leader_change:

                    while(self.leader_change):
                        self.init_flags()
                        self.normal_operation(decision = decision ,sample = sample, count_leader = True)

                else:
                    self.normal_operation(decision = decision ,sample= sample, count_leader = True)


        return self.final_preds