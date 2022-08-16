import torch
import random
from keras.api._v2.keras.utils import to_categorical
import numpy as np

class SyntheticDataset_v2(object):
    def __init__(self, args):
        self.samples = args.n_Synth_samples
        self.classes = args.n_Synth_classes

    def generate_labels(self):
        self.target = np.array([random.randrange(self.classes) for i in range(self.samples)])
        self.target_OHE = to_categorical(self.target)
        return self.target_OHE, self.target

    def confidence_(self, confidence_level):
        msg = "confidence_level must be 'Hight', 'Medium' or 'Low' but got {} instead.".format(confidence_level)
        if confidence_level == "High":
            value = random.uniform(15, 18)
        elif confidence_level == "Medium":
            value = random.uniform(10, 14)
        elif confidence_level == "Low":
            value = random.uniform(5, 9)
        else:
            raise ValueError(msg)
        return value

    def find_random_values(self, p, max_percent):
        """
        -p is an array of shape 1 x n_classes.
        - max_percent is for the random floats of the classes which are not equal with the
        max one. Perferable value is 0.1.
        - num_classes.
        """
        count = 0
        while (count <= 1):
            random_num = 1
            random_num = random.uniform(-3, 3)
            max_int = np.argmax(p, axis=0)
            for i in range(len(p)):
                random_int = random.randint(0, self.classes - 1)
                if (random_int == max_int):
                    pass
                else:
                    p[random_int] = random_num
                    break
            count = np.sum(p, axis=0)

        while (count > 1):
            diff = count - 1
            for i in range(len(p)):
                random_int = random.randint(0, self.classes - 1)
                if (random_int != max_int):
                    if p[random_int] > diff:
                        p[random_int] = p[random_int] - diff
                        break
                    else:
                        diff_1 = diff / 2
                        diff = diff - diff_1
                        if p[random_int] > diff_1:
                            p[random_int] = p[random_int] - diff_1
                            break
                        else:
                            diff_2 = diff_1 / 2
                            doff = diff - diff_2
                            if p[random_int] > diff_2:
                                p[random_int] = p[random_int] - diff_2
                                break
                else:
                    pass
            count = np.sum(p, axis=0)
        return p

    def construct_clf(self, args, max_percent):
        n_classifiers = args.n_classifiers
        conf_lvl = args.conf_lvl


        p = np.zeros([n_classifiers, self.samples, self.classes], float)
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                true = np.argmax(self.target_OHE[j])
                for k in range(p.shape[2]):
                    if k == true:
                        p[i][j][k] = self.confidence_(confidence_level=conf_lvl)
                    else:
                        pass
                p[i][j][k] = random.uniform(-3, 3)

                # p[i][j] = self.find_random_values(p[i][j], max_percent= max_percent)
                # p[i][j][k] = random.uniform(1e-6, 9e-3)

        self.preds = p  # torch.Tensor(p) * 10
        m = torch.nn.Softmax(dim=2)
        # self.preds = np.array(m(self.preds))
        return self.preds

    def reform_preds_v2(self, pred, percent_start, percent_stop):
        """
        percent is for the number of samples that are going to be changed.
        """
        p = pred
        for i in range(p.shape[0]):
            count = 0
            random_samples = random.uniform(percent_start, percent_stop)
            while (count <= p.shape[1] * random_samples):

                # random_clas = random.randint(0, p.shape[0] - 1)
                random_sample = random.randint(0, p.shape[1] - 1)
                true = np.argmax(p[i][random_sample])
                max = np.max(p[i][random_sample])
                for k in range(p.shape[2]):
                    if k == true:
                        while (p[i][random_sample][k] >= max):
                            rand_num = random.randint(0, p.shape[2] - 1)
                            if rand_num != true:
                                p[i][random_sample][k] = p[i][random_sample][rand_num]
                                p[i][random_sample][rand_num] = max  # - random.uniform(0, 1)
                            else:
                                rand_num = true - 1
                                p[i][random_sample][k] = p[i][random_sample][rand_num]
                                p[i][random_sample][rand_num] = max  # - random.uniform(0, 1)
                count = count + 1
        pr = torch.Tensor(p)
        m = torch.nn.Softmax(dim=2)
        return np.array(m(pr))


class SyntheticDataset_v3(object):
    def __init__(self,args):
        self.samples = args.n_Synth_samples
        self.classes = args.n_Synth_classes

    def generate_labels(self):
        self.target = np.array([random.randrange(self.classes) for i in range(self.samples)])
        self.target_OHE = to_categorical(self.target)
        return self.target_OHE, self.target

    def confidence_(self, confidence_level):
        msg = "confidence_level must be 'Hight', 'Medium' or 'Low' but got {} instead.".format(confidence_level)
        if confidence_level == "High":
            value = random.uniform(15, 18)
        elif confidence_level == "Medium":
            value = random.uniform(10, 14)
        elif confidence_level == "Low":
            value = random.uniform(5, 9)
        else:
            raise ValueError(msg)
        return value

    def construct_clf(self, args, max_percent):
        n_classifiers = args.n_classifiers
        conf_lvl = args.conf_lvl


        p = np.zeros([n_classifiers, self.samples, self.classes], float)
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                true = np.argmax(self.target_OHE[j])

                for k in range(p.shape[2]):

                    if k == true:
                        p[i][j][k] = self.confidence_(confidence_level=conf_lvl)
                    else:
                        pass
                p[i][j][k] = random.uniform(-2, 3)

                # p[i][j] = self.find_random_values(p[i][j], max_percent= max_percent)
                # p[i][j][k] = random.uniform(1e-6, 9e-3)

        # self.preds = torch.Tensor(p) * 10
        # m = torch.nn.Softmax(dim=2)
        # self.preds = np.array(m(self.preds))
        return p

    def reform_preds_v2(self, pred, random_samples_count):
        """
        percent is for the number of samples that are going to be changed.
        """
        p = pred
        for i in range(p.shape[0]):
            count = 0
            random_samples = random_samples_count * random.uniform(0.12, 0.99)
            print(random_samples)
            while (count <= random_samples):

                # random_clas = random.randint(0, p.shape[0] - 1)
                random_sample = random.randint(0, p.shape[1] - 1)
                true = np.argmax(p[i][random_sample])
                max = np.max(p[i][random_sample])
                for k in range(p.shape[2]):
                    if k == true:
                        while (p[i][random_sample][k] >= max):
                            rand_num = random.randint(0, p.shape[2] - 1)
                            if rand_num != true:
                                p[i][random_sample][k] = p[i][random_sample][rand_num]
                                p[i][random_sample][rand_num] = max  # - random.uniform(0, 1)
                            else:
                                rand_num = true - 1
                                p[i][random_sample][k] = p[i][random_sample][rand_num]
                                p[i][random_sample][rand_num] = max  # - random.uniform(0, 1)
                count = count + 1

        return p

    def reform_preds_v3(self, pred, percent_start, step, percent_stop):
        """
        percent is for the number of samples that are going to be changed.
        """
        p = pred
        random_samples_initial = p.shape[1] * random.uniform(percent_start, percent_stop)
        random_samples_count = random_samples_initial * step
        random_samples = random_samples_initial - random_samples_count
        condition = int(random_samples)
        print(condition)
        print("rand sample count", random_samples_count)
        random_samples_list = [random.randint(0, p.shape[1] - 1) for i in range(condition)]
        for i in range(p.shape[0]):
            count = 0
            # random_agent = random.randint(0, p.shape[0] - 1)
            random_agent = i

            while (count <= condition):

                # random_clas = random.randint(0, p.shape[0] - 1)
                for j in range(len(random_samples_list)):
                    random_sample = random_samples_list[j]
                    true = np.argmax(p[random_agent][random_sample])
                    max = np.max(p[random_agent][random_sample])
                    for k in range(p.shape[2]):
                        if k == true:
                            while (p[random_agent][random_sample][k] >= max):
                                rand_num = random.randint(0, p.shape[2] - 1)
                                if rand_num != true:
                                    p[random_agent][random_sample][k] = p[random_agent][random_sample][rand_num]
                                    p[random_agent][random_sample][rand_num] = max  # - random.uniform(0, 1)
                                else:
                                    rand_num = true - 1
                                    p[random_agent][random_sample][k] = p[random_agent][random_sample][rand_num]
                                    p[random_agent][random_sample][rand_num] = max  # - random.uniform(0, 1)
                    count = count + 1

        m = torch.nn.Softmax(dim=2)
        p = self.reform_preds_v2(pred=p, random_samples_count=random_samples_count)
        pr = torch.Tensor(p)
        return np.array(m(pr))