# MDP
Multidisciplinary project.  
  
Uncertainty in the experiment data side of the project.  
There is one python file consisting of functions that perform the calculations and one that use the functions for testing and acquiring results.

- Calculate_uncertainty.py: Consists of functions that does the uncertainty calculation based on an execution. CalculateUncertainty() and PlotData() are the only functions that should be called outside of this file.
- Data_uncertainty_experiment.py: This script uses the previously mentioned functions to calulcate uncertainties for different experiment data. It also contains plots and tests needed for the presentation of the project. This file works well as an example of how to use the solution.
- experiment_stds.csv: This is a file that is meant as a database for standard deviations of different experiment groups. The file is read from and written to automatically by Calculate_uncertainty.py. A file with this name needs to exist in the same folder as Calculate_uncertainty.py for the standard deviations to be saved. The calculations work without it, but they become slower.

Uncertainty in the model side of the project.  
There are two jupyter notebooks that have been included, namely:  
  
- curve-matching.ipynb: has the purpose of gathering the data from the Sciexpem API, creating the two datasets, one for regression and one for classification.
- curve-matching-machine-learning.ipynb:contains the machine learning analysis conducted on the classification dataset.
- dataset_c.csv: it is the dataset for classification and the only one to be included, since the analysis has been mainly done on it. It is obtained thanks to the curve-matching.ipynb notebook.
- model.h5: file containing the model that yielded the best results. It's resulted from the curve-matching-machine-learning.ipynb notebook, and stems from the analysis done on the dataset_c.csv dataset
