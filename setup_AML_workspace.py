import os
import json
import azureml
from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute

# check core SDK version number
print("Azure ML SDK Version: ", azureml.core.VERSION)

config_json = os.path.join('config.json')
with open(config_json, 'r') as f:
    config = json.load(f)

# initialize workspace from config.json
ws = Workspace.create(
    name=config['workspace_name'],
    subscription_id=config['subscription_id'],
    location=config['workspace_region'],
    resource_group=config['resource_group'],
    exist_ok=True)


print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep='\n')

# create or attach to a remote compute target
cluster_name = config['gpu_compute']

try:
    compute_target = AmlCompute(workspace=ws, name=cluster_name)
    print('Found existing compute target')
except:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(
        vm_size='STANDARD_NC6',
        max_nodes=10)

    # create the cluster
    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)

    # can poll for a minimum number of nodes and for a specific timeout. 
    # if no min node count is provided it uses the scale settings for the cluster
    compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

# use get_status() to get a detailed status for the current cluster. 
print(compute_target.get_status().serialize())
