# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
**This dataset _("https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv")_ contains data about 32950 individuals. The data includes their age, marital status, education, housing, loans,contact etc. We seek to predict the column 'y' with highest possible accuracy by training an optimized ML model on the given dataset.**



## Scikit-learn Pipeline

The pipeline includes a Random Parameter Sampler, Bandit Policy, SKLearn estimator which are used in the Hyperdrive configuration for maximum optimazation. The file _train.py_ is passed to the estimator as an entry_file and the estimator _est_ along with the policy, Parameter Sampler and some other parameters like primary metric (_Accuracy_) are passed to the HyderDrive Config Method which is then submitted and the Rn details are displayed using the widget.

**Random sampling**

Random sampling supports discrete and continuous hyperparameters. It supports early termination of low-performance runs. In random sampling, hyperparameter values are randomly selected from the defined search space. I used _choice_ to pass the parameters _--C, max iter_ to the random Sampler


**Bandit policy**

Bandit policy is based on slack factor/slack amount and evaluation interval. Bandit terminates runs where the primary metric is not within the specified slack factor/slack amount compared to the best performing run. I chose the following parameters to the Bandit Policy  as _slack_factor = 0.1, evaluation_interval=1, delay_evaluation=5_.


## AutoML

Parameters like task, primary metric, experiment timeout, training data etc. are passed into the AutoMlConfig to create an optimzed pipeline run on AutoMl that test various models and displays the best ML Algorithm based on the metrics and run time.


## Pipeline comparison

The HyperDrive Model had an Accuracy of about _0.9096611026808296_ and the AutoML Model had around _0.9167569911120748_. Although there doesn't seem to be much of a difference in the Primary Metric, I think using AutoML helped in exploring many of the machine learning algorithms that would have otherwise not considered.


## Future work

In future experiments, I would like to explore the data cleaning part and observe it's impact on the overall performance. I will also try tweaking some parameters and run the models for a longer time to study the changes that causes in the ultimate result.

