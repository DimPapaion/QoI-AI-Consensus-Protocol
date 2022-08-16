from argparse import ArgumentParser
from Utils.utils import *
from Utils.Plots import *
from Utils.Config import *
from Utils.Faulty_Agents import *
import argparse



def main(args):

    preds_OHE, targets_OHE, targets = build_Synthetic_dataset(args)
    print("Targets Shape: ".format(targets.shape))


    vote_pred, vote_prob, acc_score = build_Centralized_Inference_Synthetic(preds = preds_OHE,
                                                                                      targets = targets)

    build_Centralized_Ens_Synthetic(preds=preds_OHE, targets=targets, vote_pred=vote_pred)

    preds_indiv_, acc_score_indiv_ = build_Individualized_Ensemble_Synthetic(args,agents = args.n_classifiers,
                                                                               preds=preds_OHE,
                                                                            targets=targets, weight=acc_score)

    preds_total = build_QoI_Consensus_Synthetic(args, preds = preds_OHE, targets = targets,
                                                      weights = acc_score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Synthetic Test.", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)