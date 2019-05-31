import json
import requests
import time

from azureml.pipeline.core import PublishedPipeline
from azureml.core import Workspace, Run, Experiment
from azureml.core.authentication import ServicePrincipalAuthentication

def find_pipeline(ws, name):
    all_pub_pipelines = PublishedPipeline.get_all(ws)
    for p in all_pub_pipelines:
        if p.name == name:
            return p

def run_pipeline(ws, published_pipeline, aad_token):
    # specify the param when running the pipeline
    response = requests.post(published_pipeline.endpoint,
                            headers=aad_token,
                            json={"ExperimentName": published_pipeline.name,
                            "RunSource": "SDK"})

    try:
        run_id = response.json()["Id"]
    except:
        print(response)
        exit(1)

    experiment = Experiment(ws, pipeline_name)

    run = Run(experiment, run_id)

    while run.get_status() != 'Completed':
        print("Run status: %s" % run.get_status())
        time.sleep(5)
    
    print("Run status: %s" % run.get_status())

config_json = 'config.json'
with open(config_json, 'r') as f:
    config = json.load(f)

try:
    svc_pr = ServicePrincipalAuthentication(
        tenant_id=config['tenant_id'],
        service_principal_id=config['service_principal_id'],
        service_principal_password=config['service_principal_password'])

    aad_token = svc_pr.get_authentication_header() 
except KeyError as e:
    print("WARNING: No Service Principal found in config.json. This is fine if we are operating in DevOps.")
    svc_pr = None
    aad_token = None
    pass


ws = Workspace.from_config(path=config_json)

pipeline_name = 'kd_train_the_teacher'
published_pipeline = find_pipeline(ws, pipeline_name)

run_pipeline(ws, published_pipeline, aad_token)

print("Completed: %s" % pipeline_name)

pipeline_name = 'kd_teach_the_student'
published_pipeline = find_pipeline(ws, pipeline_name)

run_pipeline(ws, published_pipeline, aad_token)

print("Completed: %s" % pipeline_name)