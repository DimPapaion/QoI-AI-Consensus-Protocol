import argparse
import torch
from Trained_Models.TLModels import *


def get_args_parser():
    parser = argparse.ArgumentParser(description="Arguments for setup DNN testing", add_help=False)

    # Config Dataset
    parser.add_argument("--path", type= str, default="./", help="Path for the local hosted datasets.")
    parser.add_argument("--target_type", type= str, default="Ohe",choices=[None,'Ohe'],
                        help="Transformation for the labels:'None', 'Ohe'.")
    parser.add_argument("--dataset_name", type= str, default="CIFAR10",
                        choices=['CIFAR10', 'CIFAR100','F-MNIST', 'MNIST','SVHN', 'STL-10'],
                        help="Name of the selected dataset.")
    parser.add_argument("--batch_size", type= int, default=16, help="Batch size for the DataLoader.")
    parser.add_argument("--n_classes", type= int,default=10, help="Number of classes in the selected Dataset.")
    parser.add_argument("--n_samples", type=int, default=10000, help="Number of samples in the selected Dataset.")
    parser.add_argument("--transform", type= bool, default=True, help="If augmentations should be applied in dataset.")

    #Config Snythetic Dataset setup
    parser.add_argument("--n_Synth_classes", type=int, default=10, help="Number of classes in the Synthetic Dataset.")
    parser.add_argument("--n_Synth_samples", type=int, default=1000, help="Number of samples in the Synthetic Dataset.")
    parser.add_argument("--n_classifiers", type=int, default=5,help="Number of classifiers.")
    parser.add_argument("--conf_lvl", type=str, default='Medium', choices=['High', 'Medium', 'Low'],
                        help="Level of confidiality of the predected results for each classifier.")



    # Config Training setup
    parser.add_argument("--n_models", type=int, default=6, help="Number of models")
    parser.add_argument("--is_trainable", type=bool, default=False, help="Training the Models")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--optimizer", type=str, default='SGD', choices=['ADAM', 'SGD'],
                        help="Select the desired optimizer")
    parser.add_argument("--lr", type=float, default=0.001, help="Set learning rate")
    parser.add_argument("--weigt_decay", type=float, default=5e-4, help="Set weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="Set momentum")
    parser.add_argument("--check_point_path", type=str, default='./', help="Set check point saving path")

    #Config Centralized Inference
    parser.add_argument("--weights_isNone", type=bool, default=True,
                        help="Manually selected weights for Centralized Ensemble")

    # Config Individualized Ensemble
    parser.add_argument("--indiv_vote_type", type=str, default='Median', choices=['Average', 'Median','W_Average','W_Median'],
                        help="Voting type for Aggregation step")
    parser.add_argument("--indiv_decision_type", type=str, default='normal', choices=['normal', 'advance'],
                        help="Decision type: 'normal' for the probability-based condition, 'advance' for the weighted-based condition.")

    #Config QoI consensus

    parser.add_argument("--QOI_vote", type=str, default='median', choices=['average', 'median'],
                        help="Voting type for Aggregation step")
    parser.add_argument("--QOI_decision", type=str, default='both', choices=['prob','weight', 'both'],
                        help="Decision type: 'normal' for the probability-based condition, 'advance' for the weighted-based condition.")

    #Config Faulty Test
    parser.add_argument("--add_faulty", type=bool, default=True, help="Set faulty agents")


    return parser


def get_models(args):
  msg = "--dataset_name  must be 'CIFAR10', 'CIFAR100', 'F-MNIST', 'STL-10' or 'SVHN' but got {} instead.".format(args.dataset_name)
  if args.dataset_name == "CIFAR10":
    ParamsMod = dict()
    ParamsMod["Cifar10-ResNet20"] = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20",
                                                   pretrained=False)
    ParamsMod["Cifar10-ResNet32"] = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32",
                                                   pretrained=False)
    ParamsMod["Cifar10-Vgg11_bn"] = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg11_bn", pretrained=True)
    ParamsMod["Cifar10-Vgg16_bn"] = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True)
    ParamsMod["Cifar10-mobilenetv2_x0_5"] = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_mobilenetv2_x0_5", pretrained=True)
    ParamsMod["Cifar10-shufflenetv2_x0_5"] = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_shufflenetv2_x0_5", pretrained=True)
    #ParamsMod["Cifar10-repvgg_a1"] = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_repvgg_a1",
                                                    #pretrained=False)
    return ParamsMod
  elif args.dataset_name == "CIFAR100":
    ParamsMod = dict()
    ParamsMod["Cifar100-ResNet20"] = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet20",
                                                    pretrained=True)
    ParamsMod["Cifar100-ResNet32"] = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet32",
                                                    pretrained=True)
    ParamsMod["Cifar100-Vgg11_bn"] = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg11_bn",
                                                    pretrained=True)
    ParamsMod["Cifar100-Vgg16_bn"] = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg16_bn",
                                                    pretrained=True)
    ParamsMod["Cifar100-mobilenetv2_x0_5"] = torch.hub.load("chenyaofo/pytorch-cifar-models",
                                                            "cifar100_mobilenetv2_x0_5", pretrained=True)
    ParamsMod["Cifar100-shufflenetv2_x0_5"] = torch.hub.load("chenyaofo/pytorch-cifar-models",
                                                             "cifar100_shufflenetv2_x0_5", pretrained=True)
    ParamsMod["Cifar100-repvgg_a1"] = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_repvgg_a1",
                                                     pretrained=True)
    return ParamsMod

  elif args.dataset_name == "F-MNIST":
    Vgg16_FMnist = MyVgg16(dataset_name='MNIST', trained=True)
    shuffle_FMnist = MyShuffleNetv2(dataset_name='MNIST', trained=True)
    sq_FMnist = MySqueezeNet(dataset_name='MNIST')
    des_FMnist = MyDesnet161(dataset_name='MNIST', trained=True)
    ef_FMnist = MyEfficientNet(dataset_name='MNIST', trained=False)
    # google_FMnist = MyGoogleNet(dataset_name='MNIST')
    # Res18_FMnist = MyResnet()
    mobile_FMnist = MyMobileV2(dataset_name='MNIST', trained=True)
    ParamsMod = dict()
    # ParamsMod["SimpleCNN"] = FCnn(num_channels=1, num_classes=10)
    # ParamsMod["GoogleNet"] = google_FMnist.construct( n_classes = 10)
    # ParamsMod["SqueezeNet"] = sq_Mnist.construct( n_classes = 10)
    ParamsMod["VGG16"] = Vgg16_FMnist.construct(n_classes=10)
    ParamsMod["ShuffleNetv2"] = shuffle_FMnist.construct(n_classes=10)
    ParamsMod["DesNet161"] = des_FMnist.construct(n_classes=10)
    ParamsMod["EfficientNet"] = ef_FMnist.construct(n_classes=10)
    # ParamsMod["ResNet18"] = Res18_FMnist.construct(n_classes=10, dataset_name = "MNIST")
    ParamsMod["MobileNetV2"] = mobile_FMnist.construct(n_classes=10)

    return ParamsMod

  elif args.dataset_name == "SVHN":
    ParamsMod = dict()
    ParamsMod["SVHN-ResNet20"] = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=False)
    ParamsMod["SVHN-ResNet32"] = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=False)
    ParamsMod["SVHN-Vgg11_bn"] = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg11_bn", pretrained=False)
    ParamsMod["SVHN-Vgg16_bn"] = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=False)
    ParamsMod["SVHN-mobilenetv2_x0_5"] = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_mobilenetv2_x0_5", pretrained=False)
    ParamsMod["SVHN-shufflenetv2_x0_5"] = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_shufflenetv2_x0_5",
                                                         pretrained=False)
    ParamsMod["SVHN-repvgg_a1"] = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_repvgg_a1", pretrained=False)
    return ParamsMod

  elif args.dataset_name == "STL-10":
    ParamsMod = dict()
    ParamsMod["STL10-ResNet20"] = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=False)
    ParamsMod["STL10-ResNet32"] = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=False)
    # ParamsMod["STL10-Vgg11_bn"] = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg11_bn", pretrained=False)
    # ParamsMod["STL10-Vgg16_bn"] = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=False)
    ParamsMod["STL10-shufflenetv2_x0_5"] = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_shufflenetv2_x0_5",
                                                          pretrained=False)
    ParamsMod["STL10-repvgg_a1"] = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_repvgg_a1",
                                                  pretrained=False)
    return ParamsMod

  else:
    raise ValueError(msg)