from argparse import ArgumentParser
from Utils.utils import *
from Utils.Plots import *
from Utils.Config import *
from Utils.Faulty_Agents import *
import argparse



def main(args):

    dataset, loaders = build_dataset(args)
    print("Dataset Format: {}, \n Loaders Format: {}".format(dataset, loaders))
    some_infos(loaders['Train_load'])

    # Dataset Visualization
    #show_Image(args, dataset=dataset['Test'], num=123, preds=None)

    #show_dataset(args, dataset=dataset["Test"])

    #show_batch(loaders['Train_load'], batch_size=args.batch_size)

    dictModels = get_models(args)
    print(dictModels.keys())

    #Training/Inference of Base line Models
    if args.is_trainable:
        Acc_scores_training = build_training(args, ModelParams=dictModels, loaders = loaders)
    else:
        preds_OHE, preds, results = build_Centralized_Inference(args, ModelParams=dictModels,
                                                                loaders=loaders, dataset=dataset)
        acc_score = results["Accuracy"].tolist()
        targets = np.stack(dataset['Test'].targets)
        pass
    preds_final, acc_final = build_QoI_Consensus(args,
                                                 ModelParams=dictModels,
                                                 loaders=loaders,
                                                 dataset=dataset, weights=acc_score)
    #Centralized Ensembling.

    #build_Centralized_Ensemble(args, ModelParams=dictModels,loaders=loaders, dataset=dataset)

    #Individualized Ensembling.
    preds_all_NA, preds_NA, preds_all_OHE_NA = build_Individualized_Ensemble(args,
                                                                             ModelParams=dictModels,
                                                                             loaders=loaders,
                                                                             dataset=dataset, weights =acc_score)

    preds_final, acc_final = build_QoI_Consensus(args,
                                                ModelParams=dictModels,
                                                loaders=loaders,
                                                dataset=dataset, weights=acc_score)

    if args.add_faulty:
        preds_faulty = test_faulty(args,preds_total = preds_OHE)

        vote_f_pred, vote_f_prob, acc_f_score = build_Centralized_Inference_Synthetic(preds = preds_faulty,
                                                                                      targets = targets)

        build_Centralized_Ens_Synthetic(preds=preds_faulty, targets=targets, vote_pred=vote_f_pred)

        preds_indiv_f, acc_score_indiv_f = build_Individualized_Ensemble_Synthetic(args, preds=preds_faulty,
                                                                            targets=targets, weight=acc_f_score)
        preds_f_total = build_QoI_Consensus_Synthetic(args, preds = preds_faulty, targets = targets,
                                                      weights = acc_f_score)

    else:
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cifar-10 Test Set.", parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
