import numpy as np
import sys, os
import argparse


import keras
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback, ModelCheckpoint

# use non standard flow_from_directory
from utils.image_preprocessing_ver2 import ImageDataGenerator
# it outputs y_batch that contains onehot targets and logits
# logits came from xception

from keras.models import Model
from keras.layers import Lambda, concatenate, Activation
from keras.losses import categorical_crossentropy as logloss
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from keras import backend as K

from squeezenet import SqueezeNet, preprocess_input

import matplotlib.pyplot as plt

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
parser.add_argument('--logits-folder', default='256_ObjectCategories_preproc', type=str, dest='logits_dir', help='logits folder mounting point')
parser.add_argument('--learning_rate', default=1e-2, help='learning rate', type=float, required=False)
parser.add_argument('--weight_decay', default=1e-2, help='weight_decay', type=float, required=False)
parser.add_argument('--temperature', default=5.0, help='temperature', type=float, required=False)
# parser.add_argument('--lambda_const', default=2e-1, help='lambda_const', type=float, required=False)
parser.add_argument('--momentum', default=9e-1, help='momentum', type=float, required=False)
parser.add_argument('--batch_size', dest="batch_size", default=64, help='Batch size', type=int, required=False)
parser.add_argument('--transfer_learning', dest="transfer_learning", default="False", help='use the benchmark model and perform transfer learning', type=str, required=False)


args = parser.parse_args()
transfer_learning = str2bool(args.transfer_learning)
data_dir = args.data_dir
logits_dir = args.logits_dir
learning_rate = args.learning_rate
weight_decay = args.weight_decay
temperature = args.temperature
# lambda_const = args.lambda_const
momentum = args.momentum
batch_size = args.batch_size
remote_execution = args.remote_execution

if remote_execution:
    print("Running on remote compute target:", remote_execution)
    from azureml.core import VERSION
    print("azureml.core.VERSION", VERSION)
    from azureml.core import Run

    # start an Azure ML run
    run = Run.get_context()

    run.log('learning_rate', learning_rate)
    run.log('weight_decay', weight_decay)
    run.log('temperature', temperature)
    # run.log('lambda_const', lambda_const)
    run.log('momentum', momentum)
    run.log('batch_size', batch_size)
    run.log('transfer_learning', transfer_learning)

train_logits = np.load(os.path.join(logits_dir, 'train_logits.npy'), allow_pickle=True)[()]
val_logits = np.load(os.path.join(logits_dir, 'val_logits.npy'), allow_pickle=True)[()]


data_generator = ImageDataGenerator(
    data_format='channels_last',
    preprocessing_function=preprocess_input
)

# note: i'm also passing dicts of logits
train_generator = data_generator.flow_from_directory(
    os.path.join(data_dir, 'train_no_resizing'), train_logits,
    target_size=(299, 299),
    batch_size=batch_size
)

val_generator = data_generator.flow_from_directory(
    os.path.join(data_dir,'val_no_resizing'), val_logits,
    target_size=(299, 299),
    batch_size=batch_size
)


# # Show effect of the temperature

def softmax(x):
    return np.exp(x)/np.exp(x).sum()


# get a random batch
pictures, labels_and_logits = train_generator.next()
onehot_target, logits = labels_and_logits[:, :256], labels_and_logits[:, 256:]


# logits for a random image
v = logits[4]
plt.figure(figsize=(10, 5))
plt.plot(np.sort(softmax(v)), label='$T=1$', linewidth=2)
plt.plot(np.sort(softmax(v/2)), label='$T=2$', linewidth=2)
plt.plot(np.sort(softmax(v/3)), label='$T=3$', linewidth=2)
plt.plot(np.sort(softmax(v/7)), label='$T=7$', linewidth=2)
plt.plot(np.sort(softmax(v/temperature)), label='$T=T$', linewidth=3)
plt.legend()
plt.xlabel('classes sorted by probability, most probable ->')
plt.ylabel('probability')
plt.xlim([245, 255])


# log this plot to the aml workspace so we can see it in the azure portal
if remote_execution:
    run.log_image('soft target dist', plot=plt)
else:
    plt.savefig('soft_target_dist.png')
plt.close()

if transfer_learning:
    trainable = False
else:
    trainable  = True

model = SqueezeNet(weight_decay=weight_decay, image_size=299, trainable=trainable)

# remove softmax
model.layers.pop()

# usual probabilities
logits = model.layers[-1].output
probabilities = Activation('softmax')(logits)

# softed probabilities
logits_T = Lambda(lambda x: x/temperature)(logits)
probabilities_T = Activation('softmax')(logits_T)

output = concatenate([probabilities, probabilities_T])
model = Model(model.input, output)
# now model outputs 512 dimensional vectors


# # Create custom loss

def knowledge_distillation_loss(y_true, y_pred, temperature):    
    
    # split in 
    #    true targets
    #    logits from xception
    y_true, logits = y_true[:, :256], y_true[:, 256:]
    
    # convert logits to soft targets
    y_soft = K.softmax(logits/temperature)
    
    # split in 
    #    usual output probabilities
    #    probabilities made softer with temperature
    y_pred, y_pred_soft = y_pred[:, :256], y_pred[:, 256:]    
    
    return K.in_train_phase(logloss(y_soft, y_pred_soft), logloss(y_true, y_pred))
    

# # For testing use usual output probabilities (without temperature)

def accuracy(y_true, y_pred):
    y_true = y_true[:, :256]
    y_pred = y_pred[:, :256]
    return categorical_accuracy(y_true, y_pred)


def top_5_accuracy(y_true, y_pred):
    y_true = y_true[:, :256]
    y_pred = y_pred[:, :256]
    return top_k_categorical_accuracy(y_true, y_pred)



def categorical_crossentropy(y_true, y_pred):
    y_true = y_true[:, :256]
    y_pred = y_pred[:, :256]
    return logloss(y_true, y_pred)


# logloss with only soft probabilities and targets
def soft_logloss(y_true, y_pred):     
    logits = y_true[:, 256:]
    y_soft = K.softmax(logits/temperature)
    y_pred_soft = y_pred[:, 256:]    
    return logloss(y_soft, y_pred_soft)


model.compile(
    optimizer=optimizers.SGD(
        lr=learning_rate,
        momentum=momentum,
        nesterov=True), 
    loss=lambda y_true, y_pred: knowledge_distillation_loss(
        y_true, 
        y_pred, 
        temperature), 
    metrics=[accuracy, top_5_accuracy, categorical_crossentropy, soft_logloss]
)


output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)
weights_file = os.path.join(output_dir, 'squeezenet_weights.hdf5') 

callbacks = [EarlyStopping(monitor='val_accuracy', patience=4, min_delta=0.01),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=2, epsilon=0.007),
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
    steps_per_epoch=50, epochs=30, verbose=1,
    callbacks=callbacks,
    validation_data=val_generator, validation_steps=80, workers=4
)


if remote_execution:
    run.log('final_val_loss', model.history.history['val_loss'][-1])
    run.log('final_val_accuracy', model.history.history['val_accuracy'][-1])
# # Loss/epoch plots



plt.plot(model.history.history['categorical_crossentropy'], label='train')
plt.plot(model.history.history['val_categorical_crossentropy'], label='val')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('crossentropy')


# log this plot to the aml workspace so we can see it in the azure portal
if remote_execution:
    run.log_image('crossentropy', plot=plt)
else:
    plt.savefig('crossentropy.png')
plt.close()



plt.plot(model.history.history['accuracy'], label='train')
plt.plot(model.history.history['val_accuracy'], label='val')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy')


# log this plot to the aml workspace so we can see it in the azure portal
if remote_execution:
    run.log_image('accuracy', plot=plt)
else:
    plt.savefig('accuracy.png')
plt.close()

plt.plot(model.history.history['top_5_accuracy'], label='train')
plt.plot(model.history.history['val_top_5_accuracy'], label='val')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('top5_accuracy')


# log this plot to the aml workspace so we can see it in the azure portal
if remote_execution:
    run.log_image('top5_accuracy', plot=plt)
else:
    plt.savefig('top5_accuracy.png')
plt.close()
# # Results


val_generator_no_shuffle = data_generator.flow_from_directory(
    os.path.join(data_dir, 'val_no_resizing'), val_logits,
    target_size=(299, 299),
    batch_size=64, shuffle=False
)


# val_loss, val_acc, val_top_k_categorical_accuracy
if remote_execution:
    run.log_list('final eval', model.evaluate_generator(val_generator_no_shuffle, 80))
else:
    print(model.evaluate_generator(val_generator_no_shuffle, 80))
