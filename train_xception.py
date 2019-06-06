import os

import numpy as np
import argparse

import keras
from keras import optimizers
from keras.losses import categorical_crossentropy as logloss
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, Callback
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

import matplotlib.pyplot as plt

from xception import Xception, preprocess_input
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core import Workspace

def str2bool(v):
    """
    convert string representation of a boolean into a boolean representation
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Process input arguments')
parser.add_argument('--remote_execution', dest="remote_execution",  action='store_true', help='remote execution (AML compute)', required=False)
parser.add_argument('--data-folder', default='256_ObjectCategories_preproc', type=str, dest='data_dir', help='data folder mounting point')
parser.add_argument('--batch_size', dest="batch_size", default=8, help='Batch size', type=int, required=False)

args = parser.parse_args()
data_dir = args.data_dir
remote_execution = args.remote_execution


if remote_execution:
    print("Running on remote compute target:", remote_execution)
    from azureml.core import VERSION
    print("azureml.core.VERSION", VERSION)
    from azureml.core import Run

    # start an Azure ML run
    run = Run.get_context()

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

output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)
weights_file = os.path.join(output_dir, 'xception_weights.hdf5')  

data_generator = ImageDataGenerator(
    rotation_range=30, 
    zoom_range=0.3,
    horizontal_flip=True, 
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.001,
    channel_shift_range=0.1,
    fill_mode='reflect',
    data_format='channels_last',
    preprocessing_function=preprocess_input
)

data_generator_val = ImageDataGenerator(
    data_format='channels_last',
    preprocessing_function=preprocess_input
)

train_generator = data_generator.flow_from_directory(
    os.path.join(data_dir, 'teacher_no_resizing'), 
    target_size=(299, 299),
    batch_size=64
)

val_generator = data_generator_val.flow_from_directory(
    os.path.join(data_dir, 'val_no_resizing'),
    shuffle=False,
    target_size=(299, 299),
    batch_size=64
)


# # Model

model = Xception(weight_decay=1e-5)
model.count_params()

# add entropy to the usual logloss (it is for regularization),
# "Regularizing Neural Networks by Penalizing Confident Output Distributions",
# https://arxiv.org/abs/1701.06548
# it reduces overfitting a little bit
def loss(y_true, y_pred):
    entropy = -K.mean(K.sum(y_pred*K.log(y_pred), 1))
    beta = 0.1
    return logloss(y_true, y_pred) - beta*entropy

model.compile(
#     optimizer=optimizers.SGD(lr=1e-2, momentum=0.9, nesterov=True), 
    optimizer=optimizers.Adam(lr=0.005, decay=0.01),
    loss=loss, metrics=['categorical_crossentropy', 'accuracy', 'top_k_categorical_accuracy']
)

callbacks = [
        ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=2, min_delta=0.007),
        EarlyStopping(monitor='val_acc', patience=4, min_delta=0.01),
        ModelCheckpoint(filepath=weights_file, monitor='val_loss', save_best_only=True)]

# log progress to AML workspace
if remote_execution:
    class LogRunMetrics(Callback):
        # callback at the end of every epoch
        def on_epoch_end(self, epoch, log):
            # log a value repeated which creates a list
            run.log('val_loss', log['val_loss'])
            run.log('loss', log['loss'])

    callbacks.append(LogRunMetrics())
    
model.fit_generator(
    train_generator, 
    steps_per_epoch=5, epochs=3, verbose=1,
    # steps_per_epoch=50, epochs=30, verbose=1,
    callbacks=callbacks,
    validation_data=val_generator, validation_steps=80, workers=4
)


# # Loss/epoch plots
plt.plot(model.history.history['categorical_crossentropy'], label='train');
plt.plot(model.history.history['val_categorical_crossentropy'], label='val');
plt.legend();
plt.xlabel('epoch');
plt.ylabel('logloss');

# log this plot to the aml workspace so we can see it in the azure portal
if remote_execution:
    run.log_image('logloss', plot=plt)
else:
    plt.savefig('logloss.png')
plt.close()

plt.plot(model.history.history['acc'], label='train');
plt.plot(model.history.history['val_acc'], label='val');
plt.legend();
plt.xlabel('epoch');
plt.ylabel('accuracy');

# log this plot to the aml workspace so we can see it in the azure portal
if remote_execution:
    run.log_image('accuracy', plot=plt)
else:
    plt.savefig('accuracy.png')
plt.close()

plt.plot(model.history.history['top_k_categorical_accuracy'], label='train');
plt.plot(model.history.history['val_top_k_categorical_accuracy'], label='val');
plt.legend();
plt.xlabel('epoch');
plt.ylabel('top5_accuracy');


# log this plot to the aml workspace so we can see it in the azure portal
if remote_execution:
    run.log_image('top5_accuracy', plot=plt)
else:
    plt.savefig('top5_accuracy.png')
plt.close()


# val_loss, val_acc, val_top_k_categorical_accuracy
if remote_execution:
    run.log_list('final eval', model.evaluate_generator(val_generator, 80))
else:
    print(model.evaluate_generator(val_generator, 80))


# serialize NN architecture to JSON
model_json = model.to_json()

# save model JSON
with open(os.path.join(output_dir, 'model.json'), 'w') as f:
    f.write(model_json)

if remote_execution:
    from azureml.core.model import Model
    from azureml.core import Workspace

    tags = {}
    tags['run_id'] = run.id
    
    registered_model = Model.register(
        model_name='trained_xception',
        model_path=output_dir,
        tags=tags,
        workspace=ws)