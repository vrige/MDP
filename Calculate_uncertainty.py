import numpy as np
from matplotlib import pyplot as plt

def datastrToArray(str): 
    values_str = str.replace('[', '').replace(']', '').split(', ')
    values = [float(i) for i in values_str]
    return np.array(values)

def normalizeArray(ar):
    return (ar - np.amin(ar))/(np.amax(ar)- np.amin(ar))


def CalculateUncertainty(my_execution):
    #Format execution data into numpy arrays
    exec_data_x = datastrToArray(my_execution.execution_columns[0].data)
    exec_data_y = datastrToArray(my_execution.execution_columns[1].data)
    exec_data_norm_x = normalizeArray(exec_data_x)
    exec_data_norm_y = normalizeArray(exec_data_y)

    #Format experiment data into numpy arrays
    exp_data_x = np.array(my_execution.experiment.data_columns[1].data) 
    exp_data_y = np.array(my_execution.experiment.data_columns[0].data)
    exp_data_norm_x = normalizeArray(exp_data_x)
    exp_data_norm_y = normalizeArray(exp_data_y)

    exp_data = np.column_stack((exp_data_norm_x, exp_data_norm_y))
    exp_data = exp_data[np.argsort(exp_data[:, 0])].T #Sorting in ascending order
    exec_data = np.column_stack((exec_data_norm_x, exec_data_norm_y))
    exec_data = exec_data[np.argsort(exec_data[:, 0])].T #Sorting in ascending order

    #Approximating a line between every point in execution data and taking the vertical distance to the lines fro every experiment data point
    diffs = np.zeros(len(exp_data[0]))
    k = 0
    for i in range(0, len(exec_data[0])-1):
        for j in range(k, len(exp_data[0])):
            if exp_data[0,j] >= exec_data[0,i] and exp_data[0,j] <= exec_data[0,i+1]:
                p = np.polyfit(exec_data[0,i:i+2], exec_data[1,i:i+2], 1)
                fn = np.poly1d(p)
                diff = exp_data[1,j] - fn(exp_data[0,j]) #Calculates vertical distance between experiment data and polyfit line
                diffs[j] = diff
            else:
                k = j
                break

    wanted_mean = 0 #The mean is 0 because the distance is 0 if the experiment data point is on the model
    z_scores = (diffs-wanted_mean)/np.std(diffs)

    max_value = 3 #0 as min and 3 as max value, because 3*std is when the value is "outside" the normal distribution
    min_value = 0
    z_scores_norm = (np.abs(z_scores) -min_value)/(max_value-min_value) 
    uncertainty = np.array([exp_data[0], exp_data[1], 1-z_scores_norm])

    return uncertainty, exec_data, diffs

def PlotData(exec_data,uncertainty):
    #Plot data
    plt.clf
    plt.figure(1)
    plt.plot(exec_data[0], exec_data[1], '--', label='Model')
    plt.plot(uncertainty[0], uncertainty[1], '.', label='Experiment')
    for i in range(0,len(uncertainty[2])):
        if uncertainty[2,i] <= 0 or uncertainty[2,i] >= 1:
            plt.annotate("Outlier ({:.2f})".format(uncertainty[2,i]), (uncertainty[0,i],uncertainty[1,i]))
        else:
            plt.annotate("{:.2f}".format(uncertainty[2,i]), (uncertainty[0,i],uncertainty[1,i]))
    plt.title("Plot of experiment data and model")
    plt.xlabel("Normalized Temperature")
    plt.ylabel("Normalized Ignition delay")
    plt.legend(loc="upper right")
    plt.show()
