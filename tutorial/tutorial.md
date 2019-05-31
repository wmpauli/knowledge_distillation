# Tutorial: Distilling DNNs for low-cost, high-performance deployments

## Overview

### Use-Case

Imagine the following scenario: We have a large, mostly **unlabeled image dataset** and are asked to build deep-learning object classifier that can be deployed to edge devices with **little processing power**.

### Challenges

How can we train a classifier, if we only have classification labels for a small fraction of the dataset? How can we keep the neural network architecture small enough to work with edge devices?

### Approach

Luckily we have the weights of an Xception neural network architecture that has been trained by somebody else on similar dataset, using an unknown and potentially idiosyncratic or esoteric learning method.  We can use this trained model as a teacher for the smaller Squeezenet architecture (student), which requires much less memory and computational resources.

| Architecture | Memory | Parameters | Depth |
| --- | --- | --- | --- |
| Xception | 88 MB | 22,910,480 | 126 |
| Squeezenet | 4.8 MB | 421,098 | 67 |

## Instructions

1. [Setup](./setup.md) - Setup Development Environment (DO NOT SKIP THIS STEP!)
1. [Train the teacher](./train_the_teacher.md) - Define an AML pipeline to continuously train the teacher network on any labeled data you can get.
1. [Teach the student](./teach_the_student.md) - Define an AML pipeline to continuously teach the student network on unlabeled data.