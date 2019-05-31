
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[2]:


import argparse
import os

import keras
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback

import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

from squeezenet import SqueezeNet, preprocess_input


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

# In[3]:

parser = argparse.ArgumentParser(description='Process input arguments')
parser.add_argument('--data-folder', default='/home/wopauli/256_ObjectCategories_preproc/', type=str, dest='data_dir', help='data folder mounting point')
parser.add_argument('--learning_rate', default=1e-2, help='learning rate', type=float, required=False)
parser.add_argument('--weight_decay', default=1e-2, help='weight_decay', type=float, required=False)
parser.add_argument('--momentum', default=9e-1, help='momentum', type=float, required=False)
parser.add_argument('--batch_size', dest="batch_size", default=64, help='Batch size', type=int, required=False)
parser.add_argument('--remote_execution', dest="remote_execution",  action='store_true', help='remote execution (AML compute)', required=False)
parser.add_argument('--transfer_learning', dest="transfer_learning", default="False", help='use the benchmark model and perform transfer learning', type=str, required=False)


args = parser.parse_args()
data_dir = args.data_dir
learning_rate = args.learning_rate
weight_decay = args.weight_decay
momentum = args.momentum
batch_size = args.batch_size
remote_execution = args.remote_execution
transfer_learning = str2bool(args.transfer_learning)

if remote_execution:
    print("Running on remote compute target:", remote_execution)
    from azureml.core import VERSION
    print("azureml.core.VERSION", VERSION)
    from azureml.core import Run

    # start an Azure ML run
    run = Run.get_context()

    run.log('learning_rate', learning_rate)
    run.log('momentum', momentum)
    run.log('batch_size', batch_size)
    run.log('transfer_learning', args.transfer_learning)
    run.log('weight_decay', weight_decay)

# In[4]:


# batch_size = 64

data_generator = ImageDataGenerator(
    data_format='channels_last',
    preprocessing_function=preprocess_input
)

train_generator = data_generator.flow_from_directory(
    os.path.join(data_dir, 'train_no_resizing'), 
    target_size=(299, 299),
    batch_size=batch_size
)

val_generator = data_generator.flow_from_directory(
    os.path.join(data_dir, 'val_no_resizing'), shuffle=False,
    target_size=(299, 299),
    batch_size=batch_size
)


# # Model

# In[5]:

if transfer_learning:
    trainable = False
else:
    trainable  = True

model = SqueezeNet(weight_decay=weight_decay, image_size=299, trainable=True)
model.count_params()


# # Training

# In[6]:


model.compile(
#     optimizer=optimizers.Adam(lr=0.005, decay=0.01),
    optimizer=optimizers.SGD(lr=learning_rate, momentum=momentum, nesterov=True), 
    loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy']
)


# In[ ]:

callbacks = [EarlyStopping(monitor='val_acc', patience=4, min_delta=0.01),
        ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=2, epsilon=0.007)
]

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
    steps_per_epoch=50, epochs=30, verbose=1,
    callbacks=callbacks,
    validation_data=val_generator, validation_steps=80, workers=4
)

if remote_execution:
    run.log('final_val_loss', model.history.history['val_loss'][-1])
    run.log('final_val_acc', model.history.history['val_acc'][-1])
# # Loss/epoch plots

# In[ ]:


plt.plot(model.history.history['loss'], label='train');
plt.plot(model.history.history['val_loss'], label='val');
plt.legend();
plt.xlabel('epoch');
plt.ylabel('logloss');

# log this plot to the aml workspace so we can see it in the azure portal
if remote_execution:
    run.log_image('Loss', plot=plt)
else:
    plt.savefig('val_log.png')
plt.close()

# In[ ]:


plt.plot(model.history.history['acc'], label='train');
plt.plot(model.history.history['val_acc'], label='val');
plt.legend();
plt.xlabel('epoch');
plt.ylabel('accuracy');

# log this plot to the aml workspace so we can see it in the azure portal
if remote_execution:
    run.log_image('Accuracy', plot=plt)
else:
    plt.savefig('accuracy.png')
plt.close()
# In[ ]:


plt.plot(model.history.history['top_k_categorical_accuracy'], label='train');
plt.plot(model.history.history['val_top_k_categorical_accuracy'], label='val');
plt.legend();
plt.xlabel('epoch');
plt.ylabel('top5_accuracy');

# log this plot to the aml workspace so we can see it in the azure portal
if remote_execution:
    run.log_image('Top k acc', plot=plt)
else:
    plt.savefig('top_k_acc.png')
plt.close()

# # Results

# In[ ]:


# val_loss, val_acc, val_top_k_categorical_accuracy
if remote_execution:
    run.log_list('final eval', model.evaluate_generator(val_generator, 80))
else:
    print(model.evaluate_generator(val_generator, 80))
