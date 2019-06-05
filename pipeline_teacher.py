# This file is used to define an AML pipeline for training the teacher on new labeled data

import json
import shutil
import os

from azureml.core import Workspace, Run, Experiment, Datastore
from azureml.data.data_reference import DataReference
from azureml.pipeline.core.schedule import ScheduleRecurrence, Schedule
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import PublishedPipeline

from azureml.core.runconfig import CondaDependencies, RunConfiguration

from azureml.core.runconfig import DEFAULT_GPU_IMAGE

from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget

from azureml.core.authentication import ServicePrincipalAuthentication

config_json = 'config.json'
with open(config_json, 'r') as f:
    config = json.load(f)

try:
    svc_pr = ServicePrincipalAuthentication(
        tenant_id=config['tenant_id'],
        service_principal_id=config['service_principal_id'],
        service_principal_password=config['service_principal_password'])
except KeyError as e:
    print("WARNING: No Service Principal found in config.json. This is fine if we are operating in DevOps.")
    svc_pr = None
    pass

ws = Workspace.from_config(path=config_json, auth=svc_pr)

print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep = '\n')

def_blob_store  = ws.get_default_datastore()

print("Blobstore's name: {}".format(def_blob_store.name))

base_dir = '.'
    
def_blob_store = ws.get_default_datastore()

# folder for scripts that need to be uploaded to Aml compute target
script_folder = './scripts'
os.makedirs(script_folder, exist_ok=True)
os.makedirs(os.path.join(script_folder, 'utils'), exist_ok=True)

# copy all relevant assets into the `script_folder` so they will be uploaded to the cloud and made available to the remote compute target
shutil.copy(os.path.join(base_dir, 'train_xception.py'), script_folder)
shutil.copy(os.path.join(base_dir, 'xception.py'), script_folder)
shutil.copy(os.path.join(base_dir, 'config.json'), script_folder)

cpu_compute_name = config['cpu_compute']
try:
    cpu_compute_target = AmlCompute(ws, cpu_compute_name)
    print("found existing compute target: %s" % cpu_compute_name)
except:# ComputeTargetException:
    print("creating new compute target")
    
    provisioning_config = AmlCompute.provisioning_configuration(
        vm_size='STANDARD_D2_V2', 
        max_nodes=4,
        idle_seconds_before_scaledown=1800)    
    cpu_compute_target = ComputeTarget.create(ws, cpu_compute_name, provisioning_config)
    cpu_compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
    
# use get_status() to get a detailed status for the current cluster. 
print(cpu_compute_target.get_status().serialize())

# choose a name for your cluster
gpu_compute_name = config['gpu_compute']

try:
    gpu_compute_target = AmlCompute(workspace=ws, name=gpu_compute_name)
    print("found existing compute target: %s" % gpu_compute_name)
except: 
    print('Creating a new compute target...')
    provisioning_config = AmlCompute.provisioning_configuration(
        vm_size='STANDARD_NC6',
        max_nodes=10,
        idle_seconds_before_scaledown=1800)

    # create the cluster
    gpu_compute_target = ComputeTarget.create(ws, gpu_compute_name, provisioning_config)

    # can poll for a minimum number of nodes and for a specific timeout. 
    # if no min node count is provided it uses the scale settings for the cluster
    gpu_compute_target.wait_for_completion(
        show_output=True, 
        min_node_count=None, 
        timeout_in_minutes=20)

# use get_status() to get a detailed status for the current cluster. 
try:
    print(gpu_compute_target.get_status().serialize())
except BaseException as e:
    print("Could not get status of compute target.")
    print(e)


print("PipelineData object created")

# This is where data is expected to be found in Azure Blob storage
path_on_datastore = os.path.join("knowledge_distillation", "data")

# DataReference to where is the input dataset stored
labeled_data = DataReference(
    datastore=def_blob_store,
    data_reference_name="labeled_data",
    path_on_datastore=path_on_datastore)
print("DataReference object created")


# Conda dependencies for compute targets
gpu_cd = CondaDependencies.create(
    conda_packages=['cudatoolkit=10.0.130'], 
    pip_packages=['keras', 'tensorflow', 'tensorflow-gpu', 'matplotlib', 'pillow', 'six', 'numpy', 'azureml-sdk', 'tqdm'])

# Runconfig
gpu_compute_run_config = RunConfiguration(conda_dependencies=gpu_cd)
gpu_compute_run_config.environment.docker.enabled = True
gpu_compute_run_config.environment.docker.gpu_support = True
gpu_compute_run_config.environment.docker.base_image = DEFAULT_GPU_IMAGE
gpu_compute_run_config.environment.spark.precache_packages = False

# Training step for Xception
train_xception = PythonScriptStep(
    name='train_Xception',
    script_name="train_xception.py", 
    arguments=["--data-folder", labeled_data, "--remote_execution"],
    inputs=[labeled_data],
    compute_target=gpu_compute_target, 
    source_directory=script_folder,
    runconfig=gpu_compute_run_config,
    allow_reuse=True,
    hash_paths=['.']
)
print("training step created")

# Define Pipeline
pipeline = Pipeline(workspace=ws, steps=[train_xception])
print ("Pipeline is built")

# Validate Pipeline
pipeline.validate()
print("Validation complete") 

pipeline_name = 'kd_train_the_teacher'
# We need to disable (delete) previously published pipelines, because we can't have two published pipelines with the same name
from utils.azure import disable_pipeline
disable_pipeline(pipeline_name=pipeline_name, prefix='', dry_run=False)

# Publish Pipeline
published_pipeline = pipeline.publish(name=pipeline_name)
print("Pipeline is built")

# Put the pipeline on a schedule
schedule = Schedule.create(
    workspace=ws, 
    name=pipeline_name + "_sch", 
    pipeline_id=published_pipeline.id, 
    experiment_name=pipeline_name,
    datastore=def_blob_store,
    wait_for_provisioning=True,
    description="Datastore scheduler for Pipeline" + pipeline_name,
    path_on_datastore=path_on_datastore,
    polling_interval=60)
