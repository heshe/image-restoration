import azureml.core
from azureml.core import Workspace
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig
from azureml.core import Environment, Model

import os
import sys
import argparse
from pathlib import Path


# Add parser
parser = argparse.ArgumentParser(description="Deployment arguments")
parser.add_argument("--model_name", default="image_resto", type=str)
parser.add_argument("--model_version", default="", type=str)
parser.add_argument("--service_name", default="image-resto-service", type=str)
args = parser.parse_args(sys.argv[1:])
print(sys.argv)


# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))

# Get root of project
ROOT = str(Path(__file__).parent.parent.parent)

# ADD version control
model = ws.models[args.model_name]
print(model.name, 'version', model.version)

env = Environment.from_pip_requirements('image_resto_env', os.path.join(ROOT, 'requirements.txt'))

# Set path for scoring script
script_file = os.path.join(ROOT, "src", "azure", "score.py")

# Configure the scoring environment
inference_config = InferenceConfig(entry_script=script_file, environment=env)

deployment_config = AciWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1)

service_name = args.service_name
if service_name in ws.webservices:
    ans = input("Service with that name exists. Do you want to replace it? [y]/[n]")
    if ans=="y":
        ws.webservices[service_name].delete()
    else:
        print("Exiting.")
        exit()

# Deploy model
service = Model.deploy(ws, service_name, [model], inference_config, deployment_config)

service.wait_for_deployment(True)
print(service.state)