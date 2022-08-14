import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description="Arguments for setup DNN testing")

    # Config Dataset
    parser.add_argument("--path", type= str, default="./", help="Path for the local hosted datasets.")
    parser.add_argument("--target_type", type= str, default="Ohe",choices=[None,'Ohe'],
                        help="Transformation for the labels:'None', 'Ohe'.")
    parser.add_argument("--dataset_name", type= str, default="CIFAR10",
                        choices=['CIFAR10', 'CIFAR100','F-MNIST', 'MNIST','SVHN', 'STL-10'],
                        help="Name of the selected dataset.")
    parser.add_argument("--atch_size", type= int, default=16, help="Batch size for the DataLoader.")
    parser.add_argument("--n_classes", type= int,default=10, help="Number of classes for the selected Dataset.")
    parser.add_argument("--transform", type= bool, default=True, help="If augmentations should be applied in dataset.")

    #Config Models
    parser.add_argument("--models", type= dict, help="If augmentations should be applied in dataset.")
    return parser