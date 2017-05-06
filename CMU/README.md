# Notes of CMU-10-601 Machine Learning Class

## k-Nearest Neighbors

## MLE & MAP

## Naive Bayes

## Linear Regression

## Logic Regression

## Regularization

## Perceptron

## Kernels

## SVM

## k-Means

### Algorithm

### Initialization

## EM & Gaussian Mixture Model

                          Table.1 Relationship between GNB, GDA and GMM

|          |Guassian Naive Bayes | Guassian Discriminative Analysis| Guassian Mixture Model |
|:-------- |:------------------- | :-------------------------------- | :------------------- |
| Data   | options             | object                           | Small preset options |
| Model  | options             | object                            | Small preset option  |
| Decision Rule  | options             | object                            | Small preset option  |
| Objective Function  | options             | object                            | Small preset option  |
| Optimization Method | options             | object                            | Small preset option  |

### Expectation Maximization

#### Calculation Formula

#### A report Card for EM

#### Good Things

#### Bad Things

     Table.2 Comparision Between EM for GMM and EM for k-means

|          |EM for Guassian Mixture Model           | EM for K-means |
|:-------- |:------------------- | :-------------------------------- |
| Data   | options             | object                             | 
| Model  | options             | object                            |

## PCA

## Nerual Network & CNN

### Back Propagation

## Bayes Network

### Evaluate Params

### Conditional Dependency Analysis

     Table.1 Familiar Models as Bayes Nets

|          |  Formula |
|:-------- |:------------------- |
| Bernoulli / Guassian Naïve Bayes   | s             |
| Guassian Discriminative / Guassian Mixture Model  | s             | 
| Logistic Regression  | s             | 
| 1-D Guassian  | s             | 

### Independencies of a Bayes Net Model

Cascade

Common Parent

V-Structure

#### D-Separation

If variables X and Z are d-separated given a set of varaibles E Then X and Z are conditionally independent given the set E

Definition#1:

Variables X and Z are d-separated given a set of evidence variables E iff every path from X to Z is blocked
A path is "blocked" whenever:

+ ∃ Y on path s.t. Y ∈ E and Y is a "common parent"
+ ∃ Y on path s.t. Y ∈ E and Y is in a "cascade"
+ ∃ Y on path s.t. {Y, descendants(Y)} ∉ E and Y is in a "v-structure"

Definition#12

Variables X and Z are d-separated given a set of evidence variables E iff there does not exist a path in the undirected ancestral moral graph with E removed

+ Ancestral graph: keep only X,Z,E and their ancestros
+ Moral graph: add undirected edge between all pairs of each node's parent
+ Undirected graph: convert all directed edges to undirected
+ Given Removed: delete any nodes in E

![alt text](https://github.com/liute62/Machine-Learning-In-Practice/CMU/Images/D-separation.png)

## Hidden Markov Model

### Forward Algorithm

### Backward Algorithm

## Matrix Factorization

### Alternating Least Square Error

## Learning Theory

### PAC Theory
