from utils import azure

# Download tutorial data from our public blob storage
azure.download_data()

# Upload data to default blob store of your AML workspace
splits = 'teacher_no_resizing','train_no_resizing','val_no_resizing'

for split in splits:
    azure.upload_data(folder=split)
