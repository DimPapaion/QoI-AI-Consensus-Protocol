
from Utils.utils import *
import argparse



def main(args):
    dataset, loaders = build_dataset(args)
    show_dataset(args, dataset=dataset, num=123, preds=None)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cifar-10 Test Set.", parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
