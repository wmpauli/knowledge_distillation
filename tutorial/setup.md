# Setup

In this first part of the tutorial we are setting up the infrastructure for your project.

The main tasks are the configuration of:
1. Code Repository
1. Conda Environment
1. AML Workspace and configure remote compute target
1. Service Principal Authentication
1. Upload Data to Azure cloud
1. DevOps Project

## Code Repository

1. Fork Repository - Go to our [repo](https://github.com/wmpauli/knowledge_distillation.git) on github and create a fork.
1. Clone Repository - Clone your fork of our repo.


## Conda Environment

The first step is to ensure that you have all python libraries installed.  Use the provided file `environment.yml` for this purpose:

> conda env create -f environment.yml

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

Here's how to setup your Service Principal Authentication: 
- Go to the [Azure portal](https://portal.azure.com/), search for and select **Azure Active Directory** and click on **App registrations** in the left pannel, then **+ New registration**. Enter a display name and click on **Register**. 
- You can now copy the **Application (client) ID** and **Directory (tenant) ID** and paste it into the filds called `service_principal_id` and `tenant_id` in `config.json` respectively. Now click on **Certificates and secrets** in the left pannel and **+ New client secret** and **Add**, then copy your client secret (under **VALUE**) and paste it in the field called `service_principal_password` in `config.json`.
- From the [Azure portal](https://portal.azure.com/) now search and click on your resource group name `mladsrg` and click on **Access control (IAM)** in the left pannel, then **+ Add**, and **Add role assignment**. Select **Contributor** as **Role**, and type the display name of your service principal in the search box titled **Select**.

**IMPORTANT:** The above instructions require you to use `App registrations (Legacy)`. the *Legacy* version!

After you are done with this step.  Edit the `config.json` file again, to add `tenant_id`, `service_principal_id`, `service_principal_password`.

## Upload Data to Azure cloud

We of course need some [data](https://en.wikipedia.org/wiki/Data) to play with. 

We uploaded the tutorial data to a public azure blob storage account. 

Please run the script `upload_data.py`, which downloads the data from the public azure storage account, and uploads it to the default data store of your AML workspace. If you are successful with this, you should be able to see the data appear in your AML workspace in the Azure portal: `Overview` -> `Storage` -> `Blobs` -> `azureml_blobstore_<sha>` -> `knowledge_distillation`.

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







