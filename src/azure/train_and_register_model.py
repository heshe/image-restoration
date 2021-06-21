import argparse
import os
import sys
from pathlib import Path

import azureml.core
from azureml.core import Environment, Experiment, ScriptRunConfig, Workspace


# Add parser
parser = argparse.ArgumentParser(description="Deployment arguments")

# Operations
parser.add_argument("--data_name", default="image-resto", type=str)
parser.add_argument("--experiment_name", default="image_resto", type=str)

# Hyperparameters
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--n_epochs", default=5, type=int)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--x_dim", default=224 * 224, type=float)
parser.add_argument("--latent_dim", default=20, type=float)
parser.add_argument("--hidden_dim", default=400, type=float)

# Logging and optimization
parser.add_argument("--use_wandb", default=True, type=bool)
parser.add_argument("--plot_results", default=True, type=bool)
parser.add_argument("--use_cuda", default=False, type=bool)
parser.add_argument("--make_reconstructions", default=False, type=bool)

# Save model and
parser.add_argument("--save_model", default=True, type=bool)
parser.add_argument("--model_name", default="image_resto", type=str)

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
    script=os.path.join(ROOT, "src", "models", "train_azure_new.py"),
    arguments=["--input-data", image_data.as_named_input("image_resto").as_mount()],
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
