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

from azureml.train.estimator import Estimator

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

shutil.copy(os.path.join(base_dir, 'config.json'), script_folder)
shutil.copy(os.path.join(base_dir, 'get_logits_from_xception.py'), script_folder)
shutil.copy(os.path.join(base_dir, 'squeezenet.py'), script_folder)
shutil.copy(os.path.join(base_dir, 'kd_squeezenet.py'), script_folder)
shutil.copy(os.path.join(base_dir, 'model_registration.py'), script_folder)
shutil.copy(os.path.join(base_dir, 'squeezenet_weights.hdf5'), script_folder)
shutil.copy(os.path.join('./utils', 'image_preprocessing_ver1.py'), os.path.join(script_folder, 'utils'))
shutil.copy(os.path.join('./utils', 'image_preprocessing_ver2.py'), os.path.join(script_folder, 'utils'))

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
    cpu_compute_target.wait_for_completion(
        show_output=True, 
        min_node_count=None, 
        timeout_in_minutes=20)
    
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


# conda dependencies for compute targets
gpu_cd = CondaDependencies.create(
    # conda_packages=['cudatoolkit'],
    pip_packages=['keras', 'tensorflow', 'tensorflow-gpu', 'matplotlib', 'pillow', 'six', 'numpy', 'azureml-sdk', 'tqdm'])

# Runconfigs
gpu_compute_run_config = RunConfiguration(conda_dependencies=gpu_cd)
gpu_compute_run_config.environment.docker.enabled = True
gpu_compute_run_config.environment.docker.gpu_support = True
gpu_compute_run_config.environment.docker.base_image = DEFAULT_GPU_IMAGE
gpu_compute_run_config.environment.spark.precache_packages = False

print("PipelineData object created")

path_on_datastore = os.path.join("knowledge_distillation", "data")

# DataReference to where video data is stored.
labeled_data = DataReference(
    datastore=def_blob_store,
    data_reference_name="labeled_data",
    path_on_datastore=path_on_datastore)
print("DataReference object created")
    
# Naming the intermediate data as processed_data1 and assigning it to the variable processed_data1.
# raw_data = PipelineData("raw_video_fames", datastore=def_blob_store)
logits_data = PipelineData("logits_from_xception", datastore=def_blob_store)
data_metrics = PipelineData("data_metrics", datastore=def_blob_store)
data_output = PipelineData("output_data", datastore=def_blob_store)

# prepare dataset for training/testing prednet
get_logits_from_xception = PythonScriptStep(
    name='get_logits_from_xception',
    script_name="get_logits_from_xception.py", 
    arguments=["--data-folder", labeled_data, "--output_data", logits_data],
    inputs=[labeled_data],
    outputs=[logits_data],
    compute_target=gpu_compute_target, 
    source_directory=script_folder,
    runconfig=gpu_compute_run_config,
    allow_reuse=True,
    hash_paths=['.']
)
print("logit step created")


# upload data to default datastore
def_blob_store = ws.get_default_datastore()

# script_params = {
#     '--data-folder': def_blob_store.path('256_ObjectCategories_preproc').as_mount(),
#     '--remote_execution': ""
#         estimator_entry_script_arguments=[
#             '--data-folder', preprocessed_data, 
#             '--remote_execution',
#             '--dataset', dataset
#             ],
# }

est = Estimator(source_directory=script_folder,
                #  script_params=script_params,
                 compute_target=gpu_compute_target,
                 pip_packages=['keras', 'tensorflow', 'tensorflow-gpu', 'matplotlib', 'pillow', 'six', 'numpy', 'azureml-sdk', 'tqdm'],
                #  conda_packages=['cudatoolkit'],
                 entry_script='kd_squeezenet.py', 
                 use_gpu=True,
                 node_count=1)

from azureml.train.hyperdrive import RandomParameterSampling, BanditPolicy, HyperDriveConfig, PrimaryMetricGoal
from azureml.pipeline.steps import HyperDriveStep
from azureml.train.hyperdrive import choice, loguniform, uniform

ps = RandomParameterSampling(
    {
        '--learning_rate': uniform(1e-3, 2e-2),
        '--momentum': uniform(.1, .95),
        '--weight_decay': loguniform(-5, -3),
        '--temperature': uniform(1, 9),
        # '--lambda_const': uniform(.1, .3),
        '--transfer_learning': choice("True", "False")
    }
)

policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1, delay_evaluation=10)

hdc = HyperDriveConfig(estimator=est, 
                          hyperparameter_sampling=ps, 
                          policy=policy, 
                          primary_metric_name='val_loss', 
                          primary_metric_goal=PrimaryMetricGoal.MINIMIZE, 
                          max_total_runs=5, #100,
                          max_concurrent_runs=5)

hd_step = HyperDriveStep(
    name="train_w_hyperdrive",
    hyperdrive_config=hdc,
    estimator_entry_script_arguments=[
        '--data-folder', labeled_data, 
        '--logits-folder', logits_data, 
        '--remote_execution'
        ],
    # estimator_entry_script_arguments=script_params,
    inputs=[labeled_data, logits_data],
    metrics_output = data_metrics,
    allow_reuse=True
)
hd_step.run_after(get_logits_from_xception)

registration_step = PythonScriptStep(
    name='register_model',
    script_name='model_registration.py',
    arguments=['--input_dir', data_metrics, '--output_dir', data_output],
    compute_target=gpu_compute_target,
    inputs=[data_metrics],
    outputs=[data_output],
    source_directory=script_folder,
    runconfig=gpu_compute_run_config,
    allow_reuse=True,
    hash_paths=['.']
)
registration_step.run_after(hd_step)

pipeline = Pipeline(workspace=ws, steps=[get_logits_from_xception, hd_step, registration_step])
print ("Pipeline is built")

pipeline.validate()
print("Simple validation complete") 

pipeline_name = 'kd_teach_the_student'

# We need to disable (delete) previously published pipelines, because we can't have two published pipelines with the same name
from utils.azure import disable_pipeline
disable_pipeline(pipeline_name=pipeline_name, prefix='', dry_run=False)

published_pipeline = pipeline.publish(name=pipeline_name)
print("Student pipeline published")

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
