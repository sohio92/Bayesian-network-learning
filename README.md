# Bayesian network learning

This project aims to reproduce the results of Li, H.; Cabeli, V.; Sella, N.; and Isambert, H. 2019, _Constraint-based Causal Structure Learning with Consistent Separating Sets_, Advances in Neural Information Processing Systems.

We implemented several PC-derived algorithms, which learn a Bayesian Network's causal structure based on a given data set. Because conditional probabilities are unknown, independence relations must be inferred using a test such as the _Chi-squared test_ (or the _G-test_). The latter's output is dependant on a threshold value $\alpha$.

In this project, we sought to study the comparative robustness and performances of each algorithm based on the $\alpha$ value and the size of the data set used for the independence test.
