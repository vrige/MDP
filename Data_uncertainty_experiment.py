import numpy as np
from matplotlib import pyplot as plt
from SciExpeM_API.SciExpeM import SciExpeM
import scipy.stats

#my_sciexpem = SciExpeM(username='alexandersebastian.bjorklund', password='mdp2022_')
#print("My token is:", my_sciexpem.user.token)

def datastrToArray(str): 
    values_str = str.replace('[', '').replace(']', '').split(', ')
    values = [float(i) for i in values_str]
    return np.array(values)

def normalizeArray(ar):
    return (ar - np.amin(ar))/(np.amax(ar)- np.amin(ar))

my_sciexpem = SciExpeM(token='7df1e52b17c0bffb5daeb0448d8854c3d07c3665')
my_sciexpem.testConnection(verbose = True)

my_exp = my_sciexpem.filterDatabase(model_name = 'Experiment', id = '201')[0]
my_sciexpem.initializeSimulation(experiment=my_exp.id, chemModel=24, verbose = True)
my_execution = my_sciexpem.filterDatabase(model_name='Execution', id='4366')[0]

#print(my_execution.chemModel.name)
#print(my_execution.execution_columns[1].data)

#Format execution data into numpy arrays
exec_data_x = datastrToArray(my_execution.execution_columns[0].data)
exec_data_y = datastrToArray(my_execution.execution_columns[1].data)
exec_data_norm_y = normalizeArray(exec_data_y)

#Format experiment data into numpy arrays
exp_data_x = np.array(my_execution.experiment.data_columns[1].data) 
exp_data_y = np.array(my_execution.experiment.data_columns[0].data)
exp_data_norm_y = normalizeArray(exp_data_y)

#Formatting experiment data in ascending order
exp_data_x = np.flip(exp_data_x)
exp_data_norm_y = np.flip(exp_data_norm_y)

exp_data = np.column_stack((exp_data_x, exp_data_norm_y))
exp_data = exp_data[np.argsort(exp_data[:, 0])].T #Sorting x axis in ascending order
exec_data = np.column_stack((exec_data_x, exec_data_norm_y)).T

exp_std = np.std(exp_data[1])

#Approximating a line between every point in execution data and taking the vertical distance to the lines fro every experiment data point
res = np.array([exp_data[0],np.zeros(len(exp_data[0]))])
k = 0
for i in range(0, len(exec_data[0])-1):
    for j in range(k, len(exp_data[0])):
        if exp_data[0,j] >= exec_data[0,i] and exp_data[0,j] <= exec_data[0,i+1]:
            p = np.polyfit(exec_data[0,i:i+2], exec_data[1,i:i+2], 1)
            fn = np.poly1d(p)
            diff = np.abs(exp_data[1,j] - fn(exp_data[0,j])) #Calculates vertical distance between experiment data and polyfit line
            #print(scipy.stats.norm(fn(exp_data[0,j]), exp_std).cdf(exp_data[1,j]))
            res[1,j] = diff
        else:
            k = j
            break

#Plot data
plt.clf
plt.figure(1)
plt.plot(exec_data[0], exec_data[1], '--', label='Model')
plt.plot(exp_data[0], exp_data[1], '.', label='Experiment')
plt.plot(res[0], res[1], label='test')
plt.title("Plot of experiment data and model")
plt.xlabel("Temperature [K]")
plt.ylabel("Ignition delay [us]")
plt.legend(loc="upper right")
plt.show()

exit()

