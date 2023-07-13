# AENAEpy

Code for calculting the available energy of near-axis expansion configurations

This repository allows the calculation of the available energy of trapped electrons in near-axis expansion configurations. This is part of my master thesis where this repo is a supporting part by providing the numerical implementations taken to optimize the NAE configurations and to calculate the available energies of them. In here all code, data, plots, are contained. The main script for a reader of the thesis is perhaps the script `ae_nor_func_pyqsc.py` where the numerical implementation of the semi-analytical expression at NAE 1st is used to calculated the available energy. 

This code relies heavily in the following repositories: 

+ [`pyQSc`](https://github.com/landreman/pyQSC) code for generating the NAE 1st and 2nd order configurations.
+ [`BAD`](https://github.com/RalfMackenbach/BAD) code for calculating precession frequencies numerically.
+ [`AEpy`](https://github.com/RalfMackenbach/AEpy) code for calculating the available energy, with option of the NAE 2nd order as configuration input.

Later this code will incorporated as added functions for calculating the available energy of near-axis configurations at first order in the repos mentioned above. 
