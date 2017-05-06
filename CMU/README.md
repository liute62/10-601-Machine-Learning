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

#### Algorithm

#### Initialization

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

#### Back Propagation

## Bayes Network

#### Background

+ Chain Rule of Probability

For random varaiables X1, X2, X3, X4:

P(X1,X2,X3,X4) = P(X1|X2,X3,X4)P(X2|X3,X4)P(X3|X4)P(X4)

+ Conditional Independence

P(A,B|C) = P(A|C)P(B|C): Random variables A and B are condionally independent given C or equivalently:

P(A|B,C) = P(A|C) We write this as: A ⫫ B | C

#### Conditional Dependency Analysis

     Table.1 Familiar Models as Bayes Nets

|          |  Formula |
|:-------- |:------------------- |
| Bernoulli / Guassian Naïve Bayes   | s             |
| Guassian Discriminative / Guassian Mixture Model  | s             | 
| Logistic Regression  | s             | 
| 1-D Guassian  | s             | 

#### Independencies of a Bayes Net Model

|  Cascade | Common Parent | V-Structure |
|:-------- |:------------- | :-----------|
| ![alt text](https://github.com/liute62/Machine-Learning-In-Practice/blob/master/CMU/Images/Cascade.png) | ![alt text](https://github.com/liute62/Machine-Learning-In-Practice/blob/master/CMU/Images/Common-Parent.png)|![alt text](https://github.com/liute62/Machine-Learning-In-Practice/blob/master/CMU/Images/V-structure.png) |

#### D-Separation

If variables X and Z are d-separated given a set of varaibles E Then X and Z are conditionally independent given the set E

Definition#1:

Variables X and Z are d-separated given a set of evidence variables E iff every path from X to Z is blocked
A path is "blocked" whenever:

+ ∃ Y on path s.t. Y ∈ E and Y is a "common parent"
+ ∃ Y on path s.t. Y ∈ E and Y is in a "cascade"
+ ∃ Y on path s.t. {Y, descendants(Y)} ∉ E and Y is in a "v-structure"

Definition#2

Variables X and Z are d-separated given a set of evidence variables E iff there does not exist a path in the undirected ancestral moral graph with E removed

+ Ancestral graph: keep only X,Z,E and their ancestros
+ Moral graph: add undirected edge between all pairs of each node's parent
+ Undirected graph: convert all directed edges to undirected
+ Given Removed: delete any nodes in E

![alt text](https://github.com/liute62/Machine-Learning-In-Practice/blob/master/CMU/Images/D-separation.png)

#### Markov Blanket

- Def: the **co-parents** of a node are the parents of its chidren
- Def: the **Markov Blanket** of a node is the set containing the node's parents, chidren, and co-parents
- Thm: a node is **conditionally independent** of every other node in the graph given its **Markov blanket**

#### Learning Fully Observer BNs

How do we **learn** these **conditional** and **marginal** distributions for a Bayes Net?

![alt text](https://github.com/liute62/Machine-Learning-In-Practice/blob/master/CMU/Images/Learn-BN.png)

## Hidden Markov Model

#### Forward Algorithm

#### Backward Algorithm

## Learning Paradigms

### Matrix Factorization
#### Alternating Least Square Error
### Reinforcement Learning
### Information Theory
#### Contents
+ Information Representation
+ Information is addictive and non-negative (Conditional Information)
+ Average surprise of events.
+ Expected Surprise
+ CROSS ENTROPY 0 <= CH(p,q) = Ep[Iq(e)] <= ∞
+ The calculation of CROSS ENTROPY
+ Conditional ENTROPY & Joint ENTROPY

The definition of Entropy: E[I(x)] = Eplog(1 / p(x))=−∑(x∈X) p(x)logp(x), if log2(...), the units is bit

The properties of Entropy:
+ Non-negative: H(P) >= 0
+ Invariant wrt permutation of its inputs
+ For any other probability distribution {q1,q2,...,qk}: H(P) = ∑Pelog(1/pe) < ∑Pelog(1/qe) ( CH(p,q) >= CH(p, p))
+ H(P) <= logk, with equality iff pi = 1 / k ∀¡
+ The further P is from uniform, the lower the entropy

Ep[Iq] > Ep[Ip]

The more accurarcy q is, the less surprised

#### Conditional ENTROPY

|      |  cold | mild | hot  |    |
|:-----|:------|:-----|:-----|:---|
| low  | 0.1 | 0.4 | 0.1 | 0.6|
| high | 0.2 | 0.1 | 0.1 | 0.4|
|      | 0.3 | 0.5 | 0.2 | 1.0|

H(T) = H(0.3,0.5,0.2) = 1.48548

H(M) = H(0.6,0.4) = 0.970951

H(T) + H(M) = 2.456431

H(0.1,0.4,0.1,0.2,0.1,0.1) = 2.32193

H(T,M) < H(T) + H(M) (The reason is the dependency inside this two events)

#### Average Mutual Information

I(X;Y) = H(X) - H(X/Y)

if X and Y is independent, so mutual information will be zero

Properties of Average Mutual Information:

+ Symmetric
+ Non-negative
+ Zero
+ Additive

## Learning Theory

### PAC Theory
