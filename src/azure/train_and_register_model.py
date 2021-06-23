import argparse
import os
import sys
from pathlib import Path

import azureml.core
from azureml.core import Environment, Experiment, ScriptRunConfig, Workspace


# Hyperparameters
parser = argparse.ArgumentParser(description="Deployment arguments")
parser.add_argument("--experiment_name", default="image_resto", type=str)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--n_epochs", default=1, type=int)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--latent_dim", default=256, type=int)
parser.add_argument("--dropout", default=0.5, type=float)
parser.add_argument("--conv_img_dim", default=33, type=int)
parser.add_argument('--use_wandb', dest='use_wandb', action='store_true')
parser.set_defaults(use_wandb=False)
parser.add_argument('--plot_results', dest='plot_results', action='store_true')
parser.set_defaults(plot_results=False)
parser.add_argument('--no-cuda', dest='use_cuda', action='store_false')
parser.set_defaults(use_cuda=True)
parser.add_argument('--no-CNN', dest='use_CNN', action='store_false')
parser.set_defaults(use_CNN=True)
parser.add_argument('--no-azure', dest='azure', action='store_false')
parser.set_defaults(azure=True)
parser.add_argument("--model_name", default="image_resto", type=str)
parser.add_argument("--n_trials", default=10, type=int)
parser.add_argument('--small_dataset', dest='small_dataset', action='store_true')
parser.set_defaults(small_dataset=True)
parser.add_argument("--run_name", default="default_run", type=str)
parser.add_argument('--no-save_model', dest='save_model', action='store_false')
parser.set_defaults(store_model=True)
parser.add_argument("--data_name", default="image-resto", type=str)

args = parser.parse_args(sys.argv[1:])
print(args)

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
