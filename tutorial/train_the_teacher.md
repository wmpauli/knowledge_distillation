# Train the Teacher

> File: pipeline_teacher_template.py

In this part of the tutorial, you will setup the training of the teacher network for labeled data, so that the teacher network will know how to teach the student network.

To achieve this, you will define an AML pipeline to train the Xception network on the labeled data.  We will publish the AML pipeline, and assign a Scheduler to trigger a pipeline run whenever new labeled data is discovered in the Azure Blob storage.

Here we focus on three key aspects of defining an AML pipeline.  We define:
1. A DataReference to Where the input data for the pipeline are coming from
1. Conda dependencies and RunConfiguration for training
1. The training step
1. A scheduler for the pipeline

Copy the below code snippets into the template file, and rename it by removing `_template` from the filename.


## DataReference

We define a [DataReference](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.data.data_reference.datareference?view=azure-ml-py), which points to where the training data are stored. We define the `datastore` to be the storage account associated with our AML workspace, and where the data are stored in the datastore, by specificy `path_on_datastore`.

    # DataReference to where is the input dataset stored
    labeled_data = DataReference(
      datastore=def_blob_store,
      data_reference_name="labeled_data",
      path_on_datastore=path_on_datastore)
    print("DataReference object created")


## Conda dependencies and RunConfiguration for training

We defined the conda dependencies for conda environment in which we want to execute the training script.  We can specify all dependencies as conda_packages and pip_packages.

Typically, whether you add something as a conda package or a pip package depends on whether a package is a pure python package, or whether it is non-python or requires the installation of non-python dependencies.

Here, we are putting the [cudatoolkit](https://developer.nvidia.com/cuda-zone) as a conda dependency, because it requires the `libcublas` NVIDIA cuBLAS library.

    # Conda dependencies for compute targets
	gpu_cd = CondaDependencies.create(
	    conda_packages=['cudatoolkit'], 
	    pip_packages=['keras', 'tensorflow', 'tensorflow-gpu', 'matplotlib', 'pillow', 'six', 'numpy', 'azureml-sdk', 'tqdm'])

We define a [RunConfiguration](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.runconfig.runconfiguration?view=azure-ml-py).  We include the Conda dependencies defined above, enable docker support, enable GPU support for deep learning, and use the `DEFAULT_GPU_IMAGE`, because it comes pre-installed with most of the packages one needs for deep learning.

    # Runconfig
	gpu_compute_run_config = RunConfiguration(conda_dependencies=gpu_cd)
	gpu_compute_run_config.environment.docker.enabled = True
	gpu_compute_run_config.environment.docker.gpu_support = True
	gpu_compute_run_config.environment.docker.base_image = DEFAULT_GPU_IMAGE
	gpu_compute_run_config.environment.spark.precache_packages = False
  
    
## The training step

Here comes the heart of the pipeline.  We use the general purpose [PythonScriptStep](https://docs.microsoft.com/en-us/python/api/azureml-pipeline-steps/azureml.pipeline.steps.python_script_step.pythonscriptstep?view=azure-ml-py), which can be used to execute any kind of python script.

We tell provide the above DataReference object as `inputs` to the training script `train_xception.py`, to use the above `RunConfiguration`. 

*Note* the `allow_reuse=True` setting, which enables the pipeline to cache results from previous runs, if it recognizes that the results would be the same if the pipeline would be run again.

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
	    hash_paths=['.'])


## Put the pipeline on a Schedule

Finally, we put the pipeline on a [Schedule](https://docs.microsoft.com/en-us/python/api/azureml-pipeline-core/azureml.pipeline.core.schedule.schedule?view=azure-ml-py).  We set the `polling_interval` to 60 minutes.  This way the Scheduler will monitor the `datastore` for file changes in the `path_on_datastore`, and execute the pipeline accordingly.

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
        
Back to [main tutorial page](./tutorial.md)