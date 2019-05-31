# Setup

In this first part of the tutorial we are setting up the infrastructure for your project.

The main tasks are the configuration of:
1. Code Repository
1. Conda Environment
1. AML Workspace and configure remote compute target
1. Service Principal Authentication
1. DevOps Project

## Code Repository

1. Fork Repository - Go to our [repo](https://github.com/wmpauli/knowledge_distillation.git) on github and create a fork.
1. Clone Repository - Clone your fork of our repo.


## Conda Environment

The first step is to ensure that you have all python libraries installed.  Use the provided file `environment.yml` for this purpose:

> conda env create -f environmetn.yml

See the [Conda documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for more details.

Now you can use `Ctrl + P: Select Python Interpreter`, to select the conda environment `knowledge_distillation` when you debug/run scripts.


## AML Workspace and configure remote compute target

> Files: `setup_AML_workspace.py` and `config_template.json`

The AML workspace is the top-level resource for Azure Machine Learning service for managing all the artifacts you create when you use Azure Machine Learning service.  The workspace keeps a history of training runs, logs, metrics, output (e.g. processed data, trained model), and a snapshot of scripts.

A compute target is the compute resource that you use to run your training script.  Azure Machine Learning Compute is a managed-compute infrastructure that allows the user to easily create a single or multi-node compute.  The compute is created within your workspace region as a resource that can be shared with other users in your workspace.

We will do this in two steps.  First you need rename the file `config_template.json` to `config.json`.  Then update the definition of the `subscription_id` with your subscription ID.

Then you can execute the script, which will do two things:
1. Create or attach to an AML workspace
1. Create or attach to a remote compute target in that workspace

*Note*: You will be prompted to log-in to Azure in the process. Be on the lookout for tabs opening in your browser.


## Service Principal Authentication

For instructions on how to setup your Service Principal Authentication, go [here](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/manage-azureml-service/authentication-in-azureml/authentication-in-azure-ml.ipynb) and scroll down to the section on "Service Principal Authentication".

After you are done with this step.  Edit the `config.json` file again, to add `tenant_id`, `service_principal_id`, `service_principal_password`.

## DevOps Project

### Sign in to Azure DevOps

1. Go to the Azure DevOps [portal](https://dev.azure.com).
1. Click on **Sign in to Azure DevOps** to sign in.

### Create/Configure DevOps project

1. Click on **Create project** (top right).
1. Choose a **Project name**, e.g. "knowledge distillation".
1. Go to **Project Settings** (bottom left)
    1. **Pipelines** -> **Service connections**
    1. **New service connection** -> **Azure Resource Manager**
    1. **use the full version of the service connection dialog**
    1. Populate the form with the info from your `config.json` file. Make sure to set **Connection name** to exactly: "serviceConnection". Use "service_principal_password" as the "Service principal key".
    1. **Verify connection**
    1. Press **OK**
1. Go to **Pipelines** -> **Library** (left panel)
    1. **Secure files** -> click button "+ Secure file"
    1. Upload your `config.json` file.
    1. **Important:** Click on the newly uploaded file, check the box "Authorize for use in all pipelines", and **Save**.
1. Click on **New pipeline**
    1. Select **GitHub YAML**
    1. Select your fork of our repo. (*Note*: You may be asked to install the DevOps extension for GitHub at this point. It is sufficient to install it for this one repo)
    1. After a few seconds, you should see a new page, titled "Review your pipeline YAML". You can take a look at it, then press the **Run** button.

This should automatically trigger a build pipeline.  This pipeline will run for a while.  While it is running, you have time to go through the rest of the tutorial to learn what this pipeline does and how to perform knowledge distillation and hyperparameter tuning with Azure ML services.

Back to [main tutorial page](./tutorial.md)







