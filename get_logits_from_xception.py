import numpy as np
from tqdm import tqdm
import sys, os
import json

from utils.image_preprocessing_ver1 import ImageDataGenerator
# it outputs not only x_batch and y_batch but also image names

from keras.models import Model as KerasModel
from xception import Xception, preprocess_input

import argparse

parser = argparse.ArgumentParser(description='Process input arguments')
parser.add_argument('--data-folder', default='256_ObjectCategories_preproc', type=str, dest='data_dir', help='data folder mounting point')
parser.add_argument('--batch_size', dest="batch_size", default=64, help='Batch size', type=int, required=False)
parser.add_argument('--output_data', default='./data/preprocessed/UCSDped1', type=str, dest='output_data', help='data folder mounting point')

args = parser.parse_args()
data_dir = args.data_dir
batch_size = args.batch_size
output_data = args.output_data

data_generator = ImageDataGenerator(
    data_format='channels_last',
    preprocessing_function=preprocess_input
)

train_generator = data_generator.flow_from_directory(
    os.path.join(data_dir, 'train_no_resizing'), 
    target_size=(299, 299),
    batch_size=batch_size, shuffle=False
)

val_generator = data_generator.flow_from_directory(
    os.path.join(data_dir, 'val_no_resizing'), 
    target_size=(299, 299),
    batch_size=batch_size, shuffle=False
)

train_samples = train_generator.__dict__['samples']
val_samples = val_generator.__dict__['samples']


# # Get model and remove the last layer
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.model import Model
from azureml.exceptions._azureml_exception import ModelNotFoundException

config_json = 'config.json'
with open(config_json, 'r') as f:
    config = json.load(f)

try:
    svc_pr = ServicePrincipalAuthentication(
        tenant_id=config['tenant_id'],
        service_principal_id=config['service_principal_id'],
        service_principal_password=config['service_principal_password'])
except KeyError as e:
    print("Getting Service Principal Authentication from Azure Devops")
    svr_pr = None
    pass
    
ws = Workspace.from_config(path=config_json, auth=svc_pr)

try:
    model_root = Model.get_model_path('trained_xception', _workspace=ws)
except ModelNotFoundException as e:
    print("Didn't find model, cannot perform knowledge distillation.")

model = Xception()
model.load_weights(os.path.join(model_root, "xception_weights.hdf5")) 

# Remove softmax
model.layers.pop()

# Now model outputs logits
model = KerasModel(model.input, model.layers[-1].output)


# # Save logits as a dict: image name -> logit (256 dimensional vector)
train_logits = {}

batches = 0

for x_batch, _, name_batch in tqdm(train_generator):
    
    batch_logits = model.predict_on_batch(x_batch)
    
    for i, n in enumerate(name_batch):
        train_logits[n] = batch_logits[i]
    
    batches += 1
    if batches >= train_samples / batch_size:
        break



# We do the same for the validation set
val_logits = {}
batches = 0

for x_batch, _, name_batch in tqdm(val_generator):
    
    batch_logits = model.predict_on_batch(x_batch)
    
    for i, n in enumerate(name_batch):
        val_logits[n] = batch_logits[i]
    
    batches += 1
    if batches >= val_samples / batch_size:
        break


# Save logits
os.makedirs(output_data, exist_ok=True)
np.save(os.path.join(output_data, 'train_logits.npy'), train_logits)
np.save(os.path.join(output_data, 'val_logits.npy'), val_logits)

