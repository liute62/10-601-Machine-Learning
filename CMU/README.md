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
| Data   |              |                            |  |
| Model  |              |                             |   |
| Decision Rule  |              |                             |   |
| Objective Function  |              |                             |   |
| Optimization Method |              |                             |   |

### Expectation Maximization

#### Calculation Formula
+ Randomly initialize ϴ
+ Iterate until convergence:
+    E-step: Compute expected p(Zi | Xi, ϴ) using current parameters ϴ
+    M-step: Update ϴ <- argmax(ϴ') Q(ϴ'|ϴ) ϴ' is the new value
+ Each step increases Q(ϴ'|ϴ) which in turn increases L(ϴ) marginal likelihood.

#### A report Card for EM

#### Good Things
+ no learning rate (step-size) parameter
+ automatically enforces parameter constraints
+ very fast for low dimensions
+ each iteration guaranteed to improve likelihood
#### Bad Things
+ can get stuck in local minima
+ can be slower than conjugate gradient (especially near convergence)
+ requires expensive inference step
+ is a maximum likelihood/MAP method

### Guassian Mixture Model

#### EM for GMM
+ Randomly initialize u1,...uk
+ Iterate until converge:
+    E-step Compute qj(i) = p(z(i) = j|x(i),u) (Current model's estimate of prob. that x(i) came from gaussian j)
+    M-step Update parameters to maximize Q(u'|u): uj = ( ∑ qj(i)x(i) / ∑qj ( ∑from i=1 to N) (Average of all points weighted by how likely each point came from Gaussian j)

#### Connections between EM for GMM and K-Means

+ K-means is EM for GMM where δ^2 -> 0 and ∑= matrix[δ^2....] (diagonal is δ^2, other is zero)
+ K-means is the result of Block Coordinate Descent applied to a different objective for GMM

## PCA

#### Definition

PCA, Kernel PCA, ICA: Powerful unsupervised learning techniques for extracting hidden (potentially lower dimensional) structure from high dimensional datasets.

Use for:
+ Visualization
+ More efficient use of resources 
+ Statistical: fewer dimensions à better generalization
+ Noise removal (improving data quality)
+ Further processing by machine learning algorithms

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

#### Sampling from a Joint Distribution 

For estimate the probability of distributions, but not so good, instead using Gibbs Sampling.

#### Gibbs Sampling

Full conditionals only need to condition on the **Markov Blanket**

## Hidden Markov Model

#### Definition

![alt text](https://github.com/liute62/Machine-Learning-In-Practice/blob/master/CMU/Images/HMM-define.png)

1st Order Markov Assumption

yt ⫫ yj | yt-1 ∀j < t - 1

#### Baum-Welch Algorithm (EM for HMM)

#### Three Inference Problems for HMM

+ Evaluation: Compute the probability of a given sequence of observations
+ Decoding:   Find the most-likely sequence of hidden states, given a sequence of observations
+ Marginals:  Compute the marginal distribution for a hidden state, given a sequence of observations

#### Forward Algorithm


#### Backward Algorithm


#### Viterbi Algorithm


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

I(X;Y) = H(X) - H(X|Y)

if X and Y is independent, so mutual information will be zero

Properties of Average Mutual Information:

+ Symmetric ( but H(X) ≠ H(Y) and H(X|Y) ≠ H(Y|X) )
+ Non-negative (but H(X) - H(X|y) may be negative!)
+ Zero iff X,Y independent
+ Additive

## Learning Theory

### PAC Theory
