# Schema:
# uuid : UUID - uuid for this experiment
# model : str - references a model by name as specified in /models
# dataset : str - references a database by name as specified in /experiments
# data_partition_splits : list[int] - train/valid/test split sizes
# data_partition_seed : int - random seed for train/test/validation split reproducibility
# batch_size : int - minibatch size for gradient descent
# num_epochs : int - number of epochs to train for
# epoch : int - current epoch
# model_state_dict : OrderedDict - model weights and parameters
# optimizer_state_dict : OrderedDict - optimizer state and hyperparameters
# metrics : - metrics calculated throughout training or testing
#                     e.g. train_loss_history, test_accuracy, etc.
import uuid
import argparse
import torch

parser =  argparse.ArgumentParser("Configure a new experiment")
parser.add_argument("model", type=str)
parser.add_argument("--dataset", type=str, default="mnist")
parser.add_argument("--data_partition_splits", type=list, default=[40000, 10000, 10000])
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_epochs", type=int, default=10)
args = parser.parse_args()

class MlExperiment:
    def __init__(self, experiment):
        self.uuid = experiment['uuid']
        self.model = experiment['model']
        self.dataset = experiment['dataset']
        self.data_partition_splits = experiment['data_partition_splits']
        self.data_partition_seed = experiment['data_partition_seed']
        self.batch_size = experiment['batch_size']
        self.num_epochs = experiment['num_epochs']
        self.epoch = experiment['epoch']
        self.model_state_dict = experiment['model_state_dict']
        self.optimizer_state_dict = experiment['optimizer_state_dict']
        self.metrics = experiment['metrics']
        self.device = experiment['device']

    def to_dict(self):
        return {
            "uuid" : self.uuid,
            "model" : self.model,
            "dataset" : self.dataset,
            "data_partition_splits" : self.data_partition_splits,
            "data_partition_seed" : self.data_partition_seed,
            "batch_size" : self.batch_size,
            "num_epochs" : self.num_epochs,
            "epoch" : self.epoch,
            "model_state_dict" : self.model_state_dict,
            "optimizer_state_dict" : self.optimizer_state_dict,
            "metrics" : self.metrics,
            "device": self.device
        }
    
    def from_args(model, dataset, data_partition_splits, batch_size, num_epochs):
        experiment = {
            "uuid" : uuid.uuid4(),
            "model" : model,
            "dataset" : dataset,
            "data_partition_splits" : data_partition_splits,
            "data_partition_seed" : torch.Generator().initial_seed(),
            "batch_size" : batch_size,
            "num_epochs" : num_epochs,
            "epoch" : 0,
            "model_state_dict" : None,
            "optimizer_state_dict" : None,
            "metrics" : {
                "train_loss_history" : [],
                "train_accuracy_history" : [],
                "validation_loss_history" : [],
                "validation_accuracy_history" : [],
                "test_loss" : [],
                "test_accuracy" : []
            },
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }

        return MlExperiment(experiment)
        

experiment = MlExperiment.from_args(
    args.model, 
    args.dataset, 
    args.data_partition_splits,
    args.batch_size, 
    args.num_epochs)

path = f"experiments/{experiment.uuid}.json"
torch.save(experiment.to_dict(), path)

print(f"Configured experiment: {experiment.uuid}")
