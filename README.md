# A Combined Column-Generation and Lagrangian-Relaxation Approach for a Decentralized Scheduling Strategy of Heating Systems

- [Final presentation](Presentation.pdf)
- [Thesis](Thesis.pdf)

## Astract:

In this thesis, a decentralized day-ahead scheduling strategy for the coordination of heating supply systems has been implemented. The aim is thereby to enhance the integration of renewable energy sources and balance electricity demand and supply. This scheduling problem can be formulated as a mixed integer linear program (MILP) and solved using a column generation algorithm based on the Dantzig-Wolfe decomposition technique. Instead of solving the Dantzig-Wolfe masterproblem in each iteration, we propose a concept that combines a column generation algorithm with a Lagrangian relaxation approach and uses the subgradient method to optimize the shadow prices from the dual side. Thereby, the proposed algorithm preserves the advantages of a conventional column generation approach, i.e. its flexibility, scalability and limited data exchange. The results for a cluster of 100 buildings show that this method substantially improves the convergence of the column generation algorithm and delivers fairly optimal solutions for the scheduling problem in a reasonable amount of time. Moreover, different strategies for obtaining integer solutions at the end of the algorithm have been investigated to resolve problems arising from non- unique shadow prices. While the proposed integering step in this thesis does not deliver solutions of acceptable quality, the basic approach seems promising with room for improvement.

## Software:

- Python 2.7
- Gurobi 6.0.0
