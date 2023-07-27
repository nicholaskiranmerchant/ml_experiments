import torch
import matplotlib.pyplot as plt
import argparse
from config import MlExperiment
import ipdb
import numpy as np

# Load the experiment from the command line
parser =  argparse.ArgumentParser("Plot the training history of a trained model")
parser.add_argument("experiment_uuid", type=str)
args = parser.parse_args()

path = f"experiments/{args.experiment_uuid}.json"
experiment = MlExperiment(torch.load(path))

def moving_average(x, n):
    return np.convolve(x, np.ones(n)/n, mode='valid')

# Plot two plots side by side, one for train loss, the other for train validation
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

# Plot the training loss
ax1.set_title("Loss")
ax1.plot(moving_average(experiment.metrics["train_loss_history"], 100), label="Train loss")
ax1.axhline(y=experiment.metrics["test_loss"], color="r", label="Test loss")

# Plot the validation loss
ax2.set_title("Validation loss")
ax2.plot(experiment.metrics["validation_loss_history"], label="Validation loss")
ax2.axhline(y=experiment.metrics["test_loss"], color="r", label="Test loss")

# Plot the training accuracy
ax3.set_title("Accuracy")
ax3.plot(moving_average(experiment.metrics["train_accuracy_history"], 100), label="Train accuracy")
ax3.axhline(y=experiment.metrics["test_accuracy"], color="r", label="Test accuracy")

# Plot the validation accuracy
ax4.set_title("Validation accuracy")
ax4.plot(experiment.metrics["validation_accuracy_history"], label="Validation accuracy")
ax4.axhline(y=experiment.metrics["test_accuracy"], color="r", label="Test accuracy")

# Show the plot
plt.show()
