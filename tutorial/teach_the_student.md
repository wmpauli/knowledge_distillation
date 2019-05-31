# Teach the Student (Knowledge Distillation)

Now comes the *fun* part of the tutorial. We will use the teacher (Xception) network to teach the student (SqueezeNet) network.

To do this, we are again using an AML pipeline, also with a DataStore Schedule, to trigger training when new unlabeled images are discovered in Azure Blob storage.

We are not going to look at the details of how the AML pipeline is defined, because these are similar to the AML pipeline we defined previously.  Instead, we focus on the how to create the soft targets that we will use for training. 

For reference, the AML pipeline has three steps:
1. Run Xception network on unlabeled data to generate soft targets (file: `get_logits_from_xception.py`).
1. Train SqueezeNet on the soft targets and tune hyperparameters (`kd_squeezenet.py`).
1. Register trained SqueezeNet in Azure Container Registry (`model_registration.py`).


## Generate soft targets

Let's the key steps in creating soft targets.  

First, we get the URL to where the network Xception network was stored in the Azure Container Registry (ACR).  This was done at the end of training, in the file `train_xception.py`, if you want to see how that was done.

    try:
        model_root = Model.get_model_path('trained_xception', _workspace=ws)
    except ModelNotFoundException as e:
        print("Didn't find model, cannot perform knowledge distillation.")

Next we create a new instance of the Xception model, but use the weights from the registered trained Xception model above.

    model = Xception()
    model.load_weights(os.path.join(model_root, "xception_weights.hdf5")) 

Lastly, we remove the softmax layer, because we actually want the logits rather than class probabilities.  This way we can make the temperature in softmax a hyperparameter when we teach the student.

    # Remove softmax
    model.layers.pop()

    # Now model outputs logits
    model = KerasModel(model.input, model.layers[-1].output)

With all these changes done, we simply run the Xception model over all the unlabeled data, to create the logits, which we will turn into soft-targets during training.

## Train SqueezeNet on soft-targets

During this step, we train SqueezeNet on the soft targets from Xception, and validate training progress using the true labeles from our validation set.

To this, we define our own cost function for optimization, which produces different output, depending on whether we are in training or validation mode:

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
        
        return K.in_train_phase(
            logloss(y_soft, y_pred_soft), 
            logloss(y_true, y_pred))
        
Here, `y_true` contains the true target in the first 256 columns, and the logits from Xception in the latter 256 columns (the dataset has 256 categories).

By using `K.in_train_phase`, we can use two different cost functions. The first is used during training, and the second one during validation.

Note how our custom cost function accepts the softmax `temperature` as input argument.  This makes it easier to handle this as a hyperparameter with HyperDrive.

Now we just have to make sure that this cost function is also used during training and validation.  We do this when we compile the model for execution, by setting the `loss` parameter to use a `lambda` function that uses our `knowledge_distillation_loss`:

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

> Optional: If you want to learn how to use HyperDrive for hyperparameter tuning, check out the optional tutorial step [hyperparameter_tuning](./hyperparameter_tuning.md).

Back to [main tutorial page](./tutorial.md)