# Comparative Analysis of Clustering and Classification Algorithms in Data Mining

## Authors
- Dino Huang

## Abstract
This study explores four common methods in data prospecting: two types of cluster analysis (K-means and DBSCAN) and two types of classification analysis (logistic regression and SVM). The objective is to address clustering and classification problems in data mining.

## Keywords
data mining, k-means, DBSCAN, logistic regression, SVM

## Introduction
The project focuses on cluster and classification analytics. We delve into K-means and DBSCAN for cluster analysis and logistic regression and SVM for classification, addressing the challenges and peculiarities of each method.

## Pre-processing
Data pre-processing includes cleaning, standardizing, and dimensionality reduction using Principal Component Analysis (PCA).

### Data Cleaning
- Dataset 1: Normal distribution, no missing values, significant multicollinearity addressed with PCA.
- Dataset 2: Early records removed due to lack of comparability, normalization and PCA applied due to collinearity.

### Principal Component Analysis (PCA)
PCA was used to reduce dimensionality while preserving data variance. Explained variances for clustering and classification datasets were examined to determine the final dimensionality.

## Clustering
The essay explores K-means and DBSCAN for cluster analysis. K-means, a partitioning method, and DBSCAN, a density-based method, are applied to address unsupervised learning problems.

### K-means
K-means clustering’s effectiveness is explored using the elbow method for determining the optimal number of clusters.

### DBSCAN
DBSCAN's ability to identify clusters of any shape without pre-specified cluster numbers is demonstrated.

## Classification
Classification algorithms, logistic regression and SVM, are applied. Their unique approaches in dealing with data points relevant to classification are discussed.

### Logistic Regression (LR)
The LR model's performance is evaluated with a confusion matrix, and its decision boundaries are visualized.

### Support Vector Machine (SVM)
SVM's effectiveness in creating a categorical hyperplane is assessed, along with its decision boundaries.

## Conclusion
The study compares and contrasts different clustering and classification algorithms. The effectiveness of each method in the context of the provided datasets is discussed.

## Visualizations and Results

### Clustering Results
- ![K-means Clustering (2D)](images/k-mean.png)
- ![DBSCAN Clustering (2D)](images/dbscan.png)

### Classification Results
- ![LR Decision Boundary](images/lr.png)
- ![SVM Decision Boundary](images/svm.png)

## References
<a id="1">[1]</a> Ay, M. et al. (2023) “FC-Kmeans: Fixed-centered K-means algorithm,” Expert Systems with Applications, 211, p. 118656.

<a id="2">[2]</a> Cai, W. (2017) “A dimension reduction algorithm preserving both global and local clustering structure,” Knowledge-Based Systems, 118, pp. 191–203.

<a id="3">[3]</a> Ertekin, Ş., Bottou, L. and Giles, C.L. (2011) “Nonconvex online support Vector Machines,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 33(2), pp. 368–381.

<a id="4">[4]</a> Gao, J., Tao, X. and Cai, S. (2023) “Towards more efficient local search algorithms for constrained clustering,” Information Sciences, 621, pp. 287–307.

<a id="5">[5]</a> Genin, Y.V. (1996) “Euclid algorithm, orthogonal polynomials, and generalized Routh-Hurwitz algorithm,” Linear Algebra and its Applications, 246, pp. 131–158.

<a id="6">[6]</a> Ikotun, A.M. et al. (2023) “K-means Clustering Algorithms: A comprehensive review, variants analysis, and advances in the era of Big Data,” Information Sciences, 622, pp. 178–210.

<a id="7">[7]</a> Jaya Hidayat, T.H. et al. (2022) “Sentiment analysis of Twitter data related to Rinca Island development using doc2vec and SVM and logistic regression as classifier,” Procedia Computer Science, 197, pp. 660–667.

<a id="8">[8]</a> Majhi, S.K. and Biswal, S. (2018) “Optimal cluster analysis using hybrid K-means and Ant Lion optimizer,” Karbala International Journal of Modern Science, 4(4), pp. 347–360.

<a id="9">[9]</a> Mikołajewski, K. et al. (2022) “Development of Cluster Analysis Methodology for identification of model rainfall hyetographs and its application at an urban precipitation field scale,” Science of The Total Environment, 829, p. 154588.

<a id="10">[10]</a> Wu, G. et al. (2022) “HY-DBSCAN: A hybrid parallel DBSCAN clustering algorithm scalable on distributed-memory computers,” Journal of Parallel and Distributed Computing, 168, pp. 57–69.

<a id="11">[11]</a> Xu, G. et al. (2017) “Robust support vector machines based on the rescaled hinge loss function,” Pattern Recognition, 63, pp. 139–148.

<a id="12">[12]</a> Zhou, Y. et al. (2014) “A cluster-based method to map urban area from DMSP/OLS nightlights,” Remote Sensing of Environment, 147, pp. 173–185.


## Appendices
- Appendix A: Correlation diagrams for cluster and classification datasets.
- Appendix B: Code snippets for clustered data analysis.
