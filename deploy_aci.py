import os, json, sys, datetime
from azureml.core import Workspace
from azureml.core.image import ContainerImage, Image
from azureml.core.model import Model
from azureml.core.webservice import Webservice
from azureml.core.webservice import AciWebservice


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

model_name = "kd_teach_the_student"
# model_name = "trained_xception"

model = Model(ws, name=model_name)

image_config = ContainerImage.image_configuration(
    execution_script = "score.py",
    runtime = "python-slim",
    conda_file = "environment.yml",
    description = "Image with squeezenet model",
    dependencies=['squeezenet.py'])

image_name = "kd-image"

image = Image.create(name = image_name,
                     models = [model],
                     image_config = image_config,
                     workspace = ws)

image.wait_for_creation(show_output = True)

if image.creation_state != 'Succeeded':
  raise Exception('Image creation status: {image.creation_state}')

print('{}(v.{} [{}]) stored at {} with build log {}'.format(image.name, image.version, image.creation_state, image.image_location, image.image_build_log_uri))


aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, 
                                            memory_gb=1, 
                                            tags={'area': "visual object classificiation"},
                                            description='A sample description')

aci_service_name='aciwebservice'+ datetime.datetime.now().strftime('%m%d%H')

try:
    service = AciWebservice(ws, aci_service_name)
    service.delete()
except Exception as e:
    print(e)
    pass

service = Webservice.deploy_from_image(deployment_config=aciconfig,
                                        image=image,
                                        name=aci_service_name,
                                        workspace=ws)

service.wait_for_deployment()


# print(service.get_logs())

print('Deployed ACI Webservice: {} \nWebservice Uri: {}'.format(service.name, service.scoring_uri))
