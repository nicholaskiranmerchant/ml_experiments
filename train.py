import argparse
import torch
from config import  MlExperiment
import ipdb

parser =  argparse.ArgumentParser("Start or continue training an experiment")
parser.add_argument("experiment_uuid", type=str)
args = parser.parse_args()

path = f"experiments/{args.experiment_uuid}.json"
experiment = MlExperiment(torch.load(path))

# Check if the experiment is already complete
if experiment.epoch == experiment.num_epochs:
    print(f"Training completed after {experiment.epoch} epochs")
    exit()

print(f"Beginning experiment: {experiment.uuid} at epoch: [{experiment.epoch}/{experiment.num_epochs}]")

# Import and split the dataset
dataset = __import__(f"datasets.{experiment.dataset}", fromlist=["*"]).dataset

random_split_generator = torch.Generator().manual_seed(experiment.data_partition_seed)
train_dataset, valid_dataset, _ = torch.utils.data.random_split(
    dataset, experiment.data_partition_splits, generator=random_split_generator)

train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                         batch_size=experiment.batch_size,
                                         shuffle=True)

valid_dataloader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=len(valid_dataset),
                                               shuffle=True)

# Model setup
model = __import__(f"models.{experiment.model}", fromlist=["*"]).model

if experiment.model_state_dict is not None:
    model.load_state_dict(experiment.model_state_dict)

# Optimizer setup
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

if experiment.optimizer_state_dict is not None:
    optimizer.load_state_dict(experiment.optimizer_state_dict)

# Training
while (experiment.epoch < experiment.num_epochs):
    experiment.epoch += 1
    print(f"Training epoch: [{experiment.epoch}/{experiment.num_epochs}]")

    model.train()

    for batch, (x, y) in enumerate(train_dataloader):
        x = x.to(experiment.device)
        y = y.to(experiment.device)

        optimizer.zero_grad()
        y_hat = model(x)
        loss = model.loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()

        loss_item = loss.item()
        experiment.metrics["train_loss_history"].append(loss_item)
        experiment.metrics["train_accuracy_history"].append((y_hat.argmax(1) == y.argmax(1)).float().mean().item())

        if batch % 100 == 0:
            print(f"Batch: [{batch}/{len(train_dataloader)}] Loss: {loss_item}")

    # Validate the model on the validation set
    model.eval()

    for (x, y) in valid_dataloader:
        x = x.to(experiment.device)
        y = y.to(experiment.device)

        y_hat = model(x)
        loss = model.loss_fn(y_hat, y)

        loss_item = loss.item()
        accuracy_item = (y_hat.argmax(1) == y.argmax(1)).float().mean().item()
        experiment.metrics["validation_loss_history"].append(loss_item)
        experiment.metrics["validation_accuracy_history"].append(accuracy_item)

    print(f"Validation loss: {loss_item}, accuracy: {accuracy_item}")

    # Checkpoint the model
    experiment.model_state_dict = model.state_dict()
    experiment.optimizer_state_dict = optimizer.state_dict()

    torch.save(experiment.to_dict(), path)






