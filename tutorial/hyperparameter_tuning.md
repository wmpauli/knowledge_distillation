
# Hyperparameter tuning

> File: kd_squeezenet_template.py

An important aspect of machine learning is [hyperparameter](https://en.wikipedia.org/wiki/Hyperparameter) tuning. AML's [HyperDrive](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-tune-hyperparameters) is a hyperparameter tuning service, offering:
- Random, Grid and Bayesian parameter sampling
- manages the jobs creation and monitoring process for the user
- early termination


Copy the below code snippets into the template file, and rename it by removing `_template` from the filename.

## Add script arguments for handling free parameters

    parser.add_argument('--learning_rate', default=1e-2, help='learning rate', type=float, required=False)
	parser.add_argument('--weight_decay', default=1e-2, help='weight_decay', type=float, required=False)
	parser.add_argument('--temperature', default=5.0, help='temperature', type=float, required=False)
	parser.add_argument('--lambda_const', default=2e-1, help='lambda_const', type=float, required=False)
	parser.add_argument('--momentum', default=9e-1, help='momentum', type=float, required=False)
	parser.add_argument('--batch_size', dest="batch_size", default=64, help='Batch size', type=int, required=False)
	parser.add_argument('--transfer_learning', dest="transfer_learning", default="False", help='use the benchmark model and perform transfer learning', type=str, required=False)
	transfer_learning = str2bool(args.transfer_learning)

## Add processing of script arguments

	learning_rate = args.learning_rate
	weight_decay = args.weight_decay
	temperature = args.temperature
	lambda_const = args.lambda_const
	momentum = args.momentum
	batch_size = args.batch_size
    
## Log hyperparameters in your AML workspace for ML experimentation

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
	    run.log('lambda_const', lambda_const)
	    run.log('momentum', momentum)
	    run.log('batch_size', batch_size)
	    run.log('transfer_learning', transfer_learning)
        
## Create plot of soft target distribution either locally or to your workspace

	if remote_execution:
	    run.log_image('soft target dist', plot=plt)
	else:
	    plt.savefig('soft_target_dist.png')
	plt.close()
	
## Create custom Callback to log progress to AML workspace

	# log progress to AML workspace
	if remote_execution:
	    class LogRunMetrics(Callback):
	        # callback at the end of every epoch
	        def on_epoch_end(self, epoch, log):
	            # log a value repeated which creates a list
	            run.log('val_loss', log['val_loss'])
	            run.log('loss', log['loss'])
	
	    callbacks.append(LogRunMetrics())
	
## Log performance on validation set after training is complete
	
	if remote_execution:
	    run.log('final_val_loss', model.history.history['val_loss'][-1])
	    run.log('final_val_accuracy', model.history.history['val_accuracy'][-1])

## Create plot of cross entropy

	if remote_execution:
	    run.log_image('crossentropy', plot=plt)
	else:
	    plt.savefig('crossentropy.png')
	plt.close()


Back to [main tutorial page](./tutorial.md)