from cmath import exp
import numpy as np
from matplotlib import pyplot as plt
import pylab
from SciExpeM_API.SciExpeM import SciExpeM
from scipy.stats import norm, probplot, shapiro
from Calculate_uncertainty import CalculateUncertainty, PlotData
from scipy.stats import t

#my_sciexpem = SciExpeM(username='alexandersebastian.bjorklund', password='mdp2022_')
#print("My token is:", my_sciexpem.user.token)

my_sciexpem = SciExpeM(token='7df1e52b17c0bffb5daeb0448d8854c3d07c3665')
my_sciexpem.testConnection(verbose = True)

#Test CalculateUncertainty with only one experiment
#my_exp = my_sciexpem.filterDatabase(model_name = 'Experiment', id = '201')[0]
#my_sciexpem.initializeSimulation(experiment=my_exp.id, chemModel=24, verbose = True)
#my_execution = my_sciexpem.filterDatabase(model_name='Execution', id='4366')[0]
#uncertainty,exec_data, test = CalculateUncertainty(my_execution, my_sciexpem)
#PlotData(exec_data,uncertainty)

my_experiments = my_sciexpem.filterDatabase(model_name = 'Experiment', experiment_type='ignition delay measurement', fuels=['H2'])

#Filtering the database for experiements with the experiment type "ignition delay measurement" and fuels H2
exec_list = []
for exp in my_experiments:
    my_executions = my_sciexpem.filterDatabase(model_name='Execution', experiment=exp.id)
    for exec in my_executions:
        if exec.chemModel.name == 'CRECK_2100_PAH_2110_AN':
            exec_list.append(exec)

#Calculating the uncertanties for the experiment data
exp_uncertanties = []
exec_data_list = []
errors = []
year_uncer = {}
for i in range(0,len(exec_list)):
    uncertainty, exec_data, diffs = CalculateUncertainty(exec_list[i], my_sciexpem)
    exp_uncertanties.append(uncertainty)
    exec_data_list.append(exec_data)
    errors.append(diffs)

    #Collect year and corresponding uncertainty data
    if exec_list[i].experiment.file_paper.year not in year_uncer.keys():
        year_uncer[exec_list[i].experiment.file_paper.year] = uncertainty[2]
    else:
        year_uncer[exec_list[i].experiment.file_paper.year] = np.append(year_uncer[exec_list[i].experiment.file_paper.year], uncertainty[2])

#TEST AND CHECK PLOTS
for i in range(len(exp_uncertanties)):
    PlotData(exec_data_list[i], exp_uncertanties[i])

#Checking if there are big changes in average uncertainty between years
for key, value in year_uncer.items():
    avg = np.average(value)
    year_uncer[key] = np.average(value)

years = list(year_uncer.keys())
values = list(year_uncer.values())

plt.clf
plt.figure(1)
plt.bar(years, values)
plt.title("Uncertainty by year of paper")
plt.xlabel("Year")
plt.ylabel("Average of uncertainty")
plt.show()

#Checking that the errors follow a standard distribution
errors_np = np.array(errors, dtype=object)
stack_errors = np.hstack(errors_np)

print("The standard deviation of the errors/distances {:.2f}".format(np.std(stack_errors)))
print("The mean of the errors/distances {:.2f}".format(np.mean(stack_errors)))

rng = np.arange(np.amin(stack_errors), np.amax(stack_errors), 0.1) #Creating range for gaussian curve for reference

plt.clf
plt.figure(2)
plt.hist(stack_errors, bins=20, rwidth=0.8, density=True)
plt.plot(rng, norm.pdf(rng, np.mean(stack_errors), np.std(stack_errors)))
plt.title("Histogram of distances between experiment data and model")
plt.xlabel("Euqlidian distance")
plt.ylabel("Number of instances")
plt.show()

shapiro_test = shapiro(rng)
print("Statistics: ", shapiro_test.statistic)
#Null hypothesis H0: error is normally distributed
print("Pvalue: ", shapiro_test.pvalue)
probplot(rng, dist="norm", plot=pylab)
pylab.show()

m = np.mean(stack_errors)
s = np.std(stack_errors)
dof = len(stack_errors)-1
print("Mean: ",m)
confidence = 0.95
t_crit = np.abs(t.ppf((1-confidence)/2,dof))
print(m-s*t_crit/np.sqrt(len(stack_errors)), m+s*t_crit/np.sqrt(len(stack_errors)))
values = [np.random.choice(stack_errors,size=len(stack_errors),replace=True).mean() for i in range(1000)]
print(np.percentile(values,[100*(1-confidence)/2,100*(1-(1-confidence)/2)]))

# fig, ax = plt.subplots()
# ax.plot(,y)
# ax.fill_between(x, (y-ci), (y+ci), color='b', alpha=.1)

