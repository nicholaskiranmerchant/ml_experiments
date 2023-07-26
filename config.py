# Schema:
# experiment_uuid : UUID - uuid for this experiment
# data_source : str - references a database by name as specified in ___
# data_partition_splits : list[int] - train/valid/test split sizes
# data_partition_seed : int - random seed for train/test/validation split reproducibility
# batch_size : int - minibatch size for gradient descent
# num_epochs : int - number of epochs to train for
# epoch : int - current epoch
# model_state_dict : OrderedDict - model weights and parameters
# optimizer_state_dict : OrderedDict - optimizer state and hyperparameters
# minibatch_metrics : - metrics calculated at each minibatch during training
#                     e.g. train_loss_history, train_accuracy_history
# epoch_metrics : - metrics calculated at each epoch during training
#                     e.g. valid_loss_history, valid_accuracy_history
# final_metrics : - metrics calculated at the end of training
#                     e.g. test_loss, test_accuracy

import uuid
import argparse
import torch

parser =  argparse.ArgumentParser("Configure a new experiment")
parser.add_argument("model", type=str)
parser.add_argument("--data_source", type=str, default="mnist")
parser.add_argument("--data_partition_splits", type=list, default=[40000, 10000, 10000])
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_epochs", type=int, default=10)
args = parser.parse_args()

experiment = {
    "uuid" : uuid.uuid4(),
    "model" : args.model,
    "data_source" : args.data_source,
    "data_partition_splits" : args.data_partition_splits,
    "data_partition_seed" : torch.Generator().initial_seed(),
    "batch_size" : args.batch_size,
    "num_epochs" : args.num_epochs,
    "epoch" : 0,
    "model_state_dict" : None,
    "optimizer_state_dict" : None,
    "minibatch_metrics" : [],
    "epoch_metrics" : [],
    "final_metrics" : {}
}

path = f"experiments/{experiment['uuid']}.json"
torch.save(experiment, path)

print(f"Configured experiment: {experiment['uuid']}")
