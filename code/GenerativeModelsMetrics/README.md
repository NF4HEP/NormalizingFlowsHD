# GenerativeModelsMetrics
Metrics and 2-sample tests for evaluation of Generative Models

Overleaf: https://www.overleaf.com/6325513872wjspjxbhzsyf

code folder contains two files: 

    - Metrics.py 
        Includes several metrics among which:
        - Mean of 1D Kolmogorov-Smirnov test statistics and p-values
        - Mean of 1D Anderson-Darling test statistics and p-values
        - Values of Frobenius norm of the difference of the correlation matrices
        - Mean of 1D Wasserstein distance
        - Sliced Wasserstein distance computed as mean over random 1D projections of data.

    - MixtureDistributions.py
        Includes several Mixture of Gaussian models distributions
