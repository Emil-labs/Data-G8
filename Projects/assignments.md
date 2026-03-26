# Project assignments

## Energy - Aeronautics - Robotics

### Foreword

The project should be done by a **group of 3 to maximum of 4**. The expected workload per student is approximately one week of full-time work (38h) - including the project itself, writing the report and preparing for the defence.\
You should submit a full report describing the dataset and your initial observations, the method(s) chosen to analyse it and/or build a model and your results along with your opinion about their quality and relevance. It should not exceed **6,000 words** (including abstract and reference list).\
The oral presentation will last ~ 1 hours (20 minutes presentation and 40 minutes for the Q&A), during which you must defend your project, present the main steps of the work, and explain how each group member contributed to the case study.\
The report should be sent to alberto.procacci@ulb.be (the energy and robotics students should include also Pr. Parente and Pr. Garone in the email), before the **9th of June at 23h59**. Any report received late will be subject to 5 points penalty per day.
Questions can be sent by mail, or you can make an appointment (see [persons of contact](/README.md)).

> [!NOTE]
> We do not require you to submit your code itself (it will not be evaluated), but you must be able to provide access to it upon request for verification purposes. In all cases, the presentation of your results must be sufficiently clear to allow anyone to reproduce your methodology. Particular attention will be given to the interpretation of the results. For instance, you may be asked to discuss the impact of modifying certain parameters in your code.

> [!WARNING] 
> Any attempt at fraud or plagiarism will be severely punished. A project submitted without appropriate references will not be considered admissible. A consistency check will be carried out in all cases.

### Project context

The objective of this project is to develop a structured and rigorous machine-learning workflow by applying both **unsupervised** and **supervised** learning techniques **seen during the course** to two types of datasets:

1.	A **controlled toy dataset** (or several), designed to validate your methodology and implementation.
2.	A **real-world dataset** (or several), where you will apply and adapt the same methodology to solve a real case (more complex and less structured) problem of your choice.

The project is divided into two main parts:

#### Part 1 – Toy Dataset(s): Methodology development and validation

In this first part, you will work with **simple, synthetic datasets** (provided or self-generated) to:

* Understand the behaviour of the models in a **controlled environment**
* Validate your implementation (see Part 2)
* Build a **reusable workflow/template** for data analysis

You are required to implement:
* **One unsupervised learning method** (e.g. clustering or dimensionality reduction)
* **One supervised learning method** (e.g. linear regression or a simple neural network)

The chosen method should be well-suited to help answering your specific use-case (see Part 2).

The key objective is to ensure that:
* The results are **interpretable and expected**
* You can **visually validate** your outputs (plots are required)
* You understand the **limitations of each method** (e.g. linearity assumptions, overfitting, sensitivity to noise)

> [!NOTE]
> Several Toy datasets will be provided (check the [Task 1 GithHub repository](Datasets/task_1_toy.md)), but you are free to choose your own.

#### Part 2 – Real Dataset: Application and adaptation

In the second part, you will apply the **same workflow** developed in Part 1 to answer a specific research question based on a **real-world dataset**.

Unlike the toy case, real datasets typically involve:
* Noise and missing data
* Unknown underlying relationships
* Higher dimensionality and complexity
  
You are expected to:
* Justify your **choice of model(s)**
* Adapt your methodology to the dataset’s constraints
* Critically analyse the **quality and limitations** of your results
  
In addition to the research question, you must explicitly address the following questions:

* How did you handle missing and outlier values in the dataset?
* How did you choose and tune the potential hyperparameters of the model?
* How did you assess the goodness of the proposed model? What are the pros and cons of each indicator?

> [!NOTE]
> Several datasets and examples of research question will be provided (check the [Task 2 GithHub repository](Datasets/task_2_real.md)), but you are free to choose your own and/or combine several of them. Note that it must be relevant and sufficiently complex, but not too much! Remember that the project should not require more than 40h of work per person – so avoid spending too much time deciding on your research question.

#### Bonus Part

Explore one additional technique not seen during the course that you think could be relevant to apply in your own use-case. You can also try to explore Bayesian methods (see *Introduction to Bayesian statistics* lecture) and compare the results with your previous approach.

> [!WARNING]
> You must send to alberto.procacci@ulb.be **by April 6th a half-page document including: the group members, your chosen toy dataset, the real-world dataset, and your research question for approval.** This first deadline is mandatory but will not be part of the evaluation per se.

### Written report

Your written report should contain the following:

* Title page
* Abstract
* Section 1: Introduction and Methods
  * Briefly introduce the techniques used, including their main ideas, key equations, assumptions, and important hyperparameters. Discuss their strengths, limitations, and potential issues such as overfitting.
* Section 2: Toy dataset results
  * Results corresponding to Tasks 1, compared to the analytical solution. Each result has to be justified by referring to the mathematical formulation of the model.
* Section 3: Real dataset results
  * Results corresponding to Tasks 2, including relevant figures.
* Section 4: Discussion and Conclusions
  * Final discussion and conclusions, highlighting the different situations in which it is preferable to use one model over the others. Don’t forget to cite the relevant literature to strengthen your conclusions.
* AI disclosure
* Reference
* Appendix

> [!NOTE] 
> You can adapt the title of each section to suit your project, but the structure of the report must remain the same.

### Resources

The *Scikit-Learn* [User Guide](https://scikit-learn.org/stable/user_guide.html) contains most of the linear and non-linear regression models that you can use. In particular, the User Guide offers a good introduction on the mathematical formulation of the linear regression techniques, and it can be used to access the functions’ documentation. You can find a more in-depth discussion of linear regression and shrinkage methods in the third chapter of the book *“The Elements of Statistical Learning”* [1].\
In case you are building a neural network as your regression model, you can use either *PyTorch* (more complex but highly flexible) or *TensorFlow* (mare black box type of software). Also, if you are using Colab, you can request GPU resources for the training.\
If you encounter problems in the download of the data or installation of the libraries, you can contact [persons of contact](/README.md) on TEAMS or by email.

[1] Trevor Hastie, Robert Tibshirani, and Jerome Friedman. The Elements of Statistical Learning. Springer Series in Statistics. Springer New York Inc., New York, NY, USA, 2009.

### Assessment criteria

* **Clarity of presentation: 40%**, for a presentation of 20 minutes, showing the main steps of the work and how each member of the group contributed to the case study.
* **Understating of the theoretical concepts: 30%**, measuring the ability of making connections between the practical work and the theory.
* **Critical assessment: 30%**, measuring the ability to justify the results taking into account the potential and limitations of the approaches used

## Sustainable transport
See the guidelines from your section representative.

## Operations & management
See the guidelines from your section representative.
