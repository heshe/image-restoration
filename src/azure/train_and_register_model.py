import argparse
import os
import sys
from pathlib import Path

import azureml.core
from azureml.core import Environment, Experiment, ScriptRunConfig, Workspace


# Add parser
parser = argparse.ArgumentParser(description="Deployment arguments")

# Operations
parser.add_argument("--experiment_name", default="image_resto", type=str)

# Hyperparameters
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--n_epochs", default=10, type=int)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--latent_dim", default=256, type=int)
parser.add_argument("--dropout", default=0.5, type=float)
parser.add_argument("--conv_img_dim", default=33, type=int)
parser.add_argument("--use_wandb", default=False, type=bool)
parser.add_argument("--plot_results", default=False, type=bool)
parser.add_argument("--use_cuda", default=True, type=bool)
parser.add_argument("--use_CNN", default=True, type=bool)
parser.add_argument("--azure", default=True, type=bool)
parser.add_argument("--model_name", default="image_resto", type=str)
parser.add_argument("--optuna", default=False, type=bool)
parser.add_argument("--n_trials", default=10, type=int)
parser.add_argument("--small_dataset", default=False, type=bool)
parser.add_argument("--run_name", default="default_run", type=str)
parser.add_argument("--save_model", default=True, type=bool)
parser.add_argument("--data_name", default="image-resto", type=str)

args = parser.parse_args(sys.argv[1:])
print(sys.argv)

# Load the workspace from the saved config file
ws = Workspace.from_config()
print("Ready to use Azure ML {} to work with {}".format(azureml.core.VERSION, ws.name))

# Get root of project
ROOT = str(Path(__file__).parent.parent.parent)

# Create a Python environment for the experiment
project_env = Environment.from_pip_requirements(
    "project_env", os.path.join(ROOT, "requirements.txt")
)

# Get the training dataset
image_data = ws.datasets.get(args.data_name)

# Create a script config
script_config = ScriptRunConfig(
    source_directory=ROOT,
    script=os.path.join(ROOT, "src", "models", "train_azure.py"),
    arguments=[
        "--lr", args.lr,
        "--n_epochs", args.n_epochs,
        "--latent_dim", args.latent_dim,
        "--dropout", args.dropout,
        "--input_data", image_data.as_named_input("image_resto").as_mount()
    ],
    environment=project_env,
)  # Use the environment created previously

# submit the experiment
experiment_name = args.experiment_name
experiment = Experiment(workspace=ws, name=experiment_name)
run = experiment.submit(config=script_config)
run.wait_for_completion()

# Get logged metrics and files
metrics = run.get_metrics()
for key in metrics.keys():
    print(key, metrics.get(key))
