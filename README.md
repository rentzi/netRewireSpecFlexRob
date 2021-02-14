netRewireSpecFlexRob: A rewiring model based on heat diffusion that shows specificity, flexibility and robustness  
=================================


This package replicates the Figures for a paper that models adaptive rewiring in the brain (Rentzeperis I and van Leeuwen C (2021) Adaptive Rewiring in Weighted Networks Shows Specificity, Robustness, and Flexibility. Front. Syst.).You can simply run the jupyter notebooks that start with _Figure to replicate the Figures in the paper. The data necessary have already been calculated and stored in the data folder. 

If you wish to replicate the data yourself, you will have to run variations of the jupyter notebooks that start with 1ArandAGetSave to get all the adjacency matrices necessary (A), before and after different stages of rewiring. Note that this step if run in its entirety may take weeks and considerable space. However, you can run a smaller number of repetitions as a proof of concept. Then you will have to run all the jupyter notebooks that start with 2GetSave that runs the summary metrics from the adjacency matrices that were stored from the previous step. These particular metrics I have already stored in the data folder, so that you will not have to run those two steps. 

The abstract of the paper is as follows:

Brain network connections rewire adaptively in response to neural activity. Adaptive rewiring may be understood as a process which, at its every step, is aimed at optimizing the efficiency of signal diffusion. In evolving model networks, this amounts to creating shortcut connections in regions with high diffusion and pruning where diffusion is low. Adaptive rewiring leads over time to topologies akin to brain anatomy: small worlds with rich club and modular or centralized structures. We continue our investigation of
adaptive rewiring by focusing on three desiderata: specificity of evolving model network architectures, robustness of dynamically maintained architectures, and flexibility of
network evolution to stochastically deviate from specificity and robustness. Our adaptive rewiring model simulations show that specificity and robustness characterize alternative modes of network operation, controlled by a single parameter, the rewiring interval. Small control parameter shifts across a critical transition zone allow switching between the two modes. Adaptive rewiring exhibits greater flexibility for skewed, lognormal connection weight distributions than for normally distributed ones. The results qualify adaptive rewiring as a key principle of self-organized complexity in network architectures, in particular of those that characterize the variety of functional architectures in the brain.

Prerequisites
-------------

- Python 3+
- igraph
- scipy
- numpy
- matplotlib
- sklearn


Installation
------------
the package ``netRewireSpecFlexRob`` is the folder that will be created once you clone this repository. To run the functions of the package include the path with the package folder. For example

```
import sys
sys.path.append('the directory path')
import netRewireSpecFlexRob
```

Author
------

Ilias Rentzeperis 
