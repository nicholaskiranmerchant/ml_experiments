import argparse
import torch

parser =  argparse.ArgumentParser("Start or continue training an experiment")
parser.add_argument("experiment_uuid", type=str)
args = parser.parse_args()

path = f"experiments/{args.experiment_uuid}.json"
experiment = torch.load(path)

print(f"Beginning experiment: {experiment['uuid']} at epoch: [{experiment['epoch']}/{experiment['num_epochs']}]")

