# Knowledge distillation with Keras

Keras implementation of Hinton's knowledge distillation (KD), a way of transferring knowledge from a large model into a smaller model.


## Dataset

Caltech-256 dataset for a demonstration of the technique.
* I resize all images to 299x299.
* For validation I use 20 images from each category.
* For training I use 100 images from each category.
* I use random crops and color augmentation to balance the dataset.


## CNN Architectures 

Teacher: Xception
Student: Squeezenet 


## Tutorial

Please complete the [tutorial](./tutorial/tutorial.md), if you want to learn how to use knowledge distillation to train a student model with unlabeled data.


## Prerequisites and Requirements

### Skills 

- Azure Subscription ID. Make sure you have an Azure subscription ID that allows you to create compute targets (compare VM sizes). 
- Knowledge of basic data science and machine learning concepts. Here and here you'll find short introductory material. 
- Moderate skills in coding with Python and machine learning using Python. A good place to start is here. 
- Prior experience with AML services is recommended. 

### Software Dependencies 

- A recent installation of Miniconda (or Anaconda) 
- An IDE for editing code: VS code https://code.visualstudio.com/ 



## References

- Buciluǎ, Cristian, Rich Caruana, and Alexandru Niculescu-Mizil. "Model compression." Proceedings of the 12th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2006.

- Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. "Distilling the knowledge in a neural network." arXiv preprint arXiv:1503.02531 (2015).

