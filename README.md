# Global Optimization Strategies for Standard Quadratic Problems
## Abstract
**Standard quadratic optimization problems** (**StQPs**) are quadratic problems with simple linear constraints whose resolution focuses in the search for their global minimums. StQPs are generally not convex and belong to the class of NP-hard problems. In this thesis work, a fast decomposition algorithm, **Sequential Minimal Optimization** (**SMO**), was applied to the StQPs with the aim of approaching as much as possible to the global minimum of the problem. The SMO algorithm was used as a local solver within a global optimization strategy. SMO updates, to each iteration, two variables chosen with an adequate selection rule. The main characteristics of the algorithm are that:
* The two variables are updated by solving an sub-production which, although not convex, can be resolved analytically;
* The selection rule adopted guarantees the convergence towards stationary points of the problem.

A fundamental point of the SMO algorithm is the choice of the starting point from which to start the optimization process. Various initialization strategies have been proven in this thesis and various experiments, but the one that has given better results in terms of computational performance, was that relating to the multi-start strategy based on the **convexity graph** associated with the starting problem (**SMO-CG**). These experiments have shown that SMO-CG is a valid alternative to the state of the art algorithm for standard quadratic problems (StQPs).

## Files
- Thesis (in italian) → [PDF](thesis_pdf/thesis.pdf)
- Presentation (in italian) → [PDF](thesis_pdf/presentation.pdf)

## To Run
To run the project, you have to run ***test_xxx.py*** files:
```shell
python3 test_xxx.py
```