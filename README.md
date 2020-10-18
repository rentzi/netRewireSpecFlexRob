netRewireSpecFlexRob: A rewiring model based on heat diffusion that shows specificity, flexibility or robustness for different values of a parameter (tau)  
=================================


This package replicates the Figures for a paper on adaptive rewiring that is currently under review (_Adaptive rewiring in weighted networks shows specificity, robustness, and flexibility._ Ilias Rentzeperis, Cees van Leeuwen). You can simply run the jupyter notebooks that start with _Figure to replicate the Figures in the paper. The data necessary have already been calculated and stored in the data folder. 

If you wish to replicate the data yourself, you will have to run variations of the jupyter notebooks that start with 1ArandAGetSave to get all the adjacency matrices necessary (A), before and after different stages of rewiring. Note that this step if run in its entirety may take weeks and considerable space. However, you can run a smaller number of repetitions as a proof of concept. Then you will have to run all the jupyter notebooks that start with 2GetSave that runs the summary metrics from the adjacency matrices that were stored from the previous step. These particular metrics I have already stored in the data folder, so that you will not have to run those two steps. 

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
