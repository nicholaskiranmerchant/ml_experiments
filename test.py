import argparse
import torch
from config import  MlExperiment

parser =  argparse.ArgumentParser("Test a trained model")
parser.add_argument("experiment_uuid", type=str)
args = parser.parse_args()

path = f"experiments/{args.experiment_uuid}.json"
experiment = MlExperiment(torch.load(path))

# Import and split the dataset
dataset = __import__(f"datasets.{experiment.dataset}", fromlist=["*"]).dataset

random_split_generator = torch.Generator().manual_seed(experiment.data_partition_seed)
_, _, test_dataset = torch.utils.data.random_split(
    dataset, experiment.data_partition_splits, generator=random_split_generator)

test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=len(test_dataset),
                                               shuffle=True)

# Model setup
model = __import__(f"models.{experiment.model}", fromlist=["*"]).model

if experiment.model_state_dict is not None:
    model.load_state_dict(experiment.model_state_dict)

# Test the model
model.eval()

for (x,y) in test_dataloader:
    x = x.to(experiment.device)
    y = y.to(experiment.device)

y_hat = model(x)
loss = model.loss_fn(y_hat, y)

loss_item = loss.item()
accuracy_item = (y_hat.argmax(1) == y.argmax(1)).float().mean().item()

experiment.metrics["test_loss"] = loss_item
experiment.metrics["test_accuracy"] = accuracy_item

torch.save(experiment.to_dict(), path)

print(f"Test loss: {loss_item}, accuracy: {accuracy_item}")