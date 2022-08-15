from argparse import ArgumentParser

from Utils.utils import *
from Utils.Plots import *
from Utils.Config import *
import argparse



def main(args):

    dataset, loaders = build_dataset(args)
    print("Dataset Format: {}, \n Loaders Format: {}".format(dataset, loaders))
    some_infos(loaders['Train_load'])
    #show_Image(args, dataset=dataset['Test'], num=123, preds=None)

    #show_dataset(args, dataset=dataset["Test"])

    #show_batch(loaders['Train_load'], batch_size=args.batch_size)

    dictModels = get_models(args)

    #Training/Inference of Base line Models
    if args.is_trainable:
        Acc_scores_training = build_training(args, ModelParams=dictModels, loaders = loaders)
    else:
        preds_OHE, preds, results = build_Centralized_Inference(args, ModelParams=dictModels,
                                                                loaders=loaders, dataset=dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cifar-10 Test Set.", parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
