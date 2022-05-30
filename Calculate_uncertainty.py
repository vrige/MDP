import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from os.path import exists
from scipy.stats import t
import numpy as np
import csv
import matplotlib.pyplot as plt

#Convert string of data to numpy array
def dataFloatTypeToArray(dataStr): 
    values_str = dataStr.replace('[', '').replace(']', '').split(', ')
    values = [float(i) for i in values_str]
    return np.array(values)

#Convert string of data to list
def dataStrTypeToList(dataStr):
    values_str = dataStr.replace('[', '').replace(']', '').replace('"', '').split(', ')
    values = [str(i) for i in values_str]
    return values

#Normalize the values of a numpy array
def normalizeArray(ar):
    return (ar - np.amin(ar))/(np.amax(ar)- np.amin(ar))

#Check if the experiment group with an assigned standard deviation exists in the csv file
def CheckIfExpInCsv(exec):
    exp = exec.experiment
    exp_t = exp.experiment_type
    exp_f = dataStrTypeToList(exp.fuels) #Convert fuels to a list
    exp_r = exp.reactor

    #Checks the csv file if the experiment group is there already
    if exists('experiment_stds.csv'):
        df = pd.read_csv('experiment_stds.csv')
        if not df.empty:
            exp = df.loc[(df['Type'] == exp_t) & (df['Reactor'] == exp_r) & (df['Fuels'] == exp_f)]
            if not exp.empty:
                return float(exp['std']) #Return the standard deviation of experiment group
            else:
                return 0
        else:
            return 0
    else:
        print("Collection of standard deviations per experiment group does not exist!")
        return 0

#Drives the calculation of the standard deviation for experiment group
def ApproximateStd(my_execution, my_sciexpem):
    standard_deviation = CheckIfExpInCsv(my_execution)
    if not standard_deviation:
        standard_deviation = CalulateStdForExpGroup(my_execution, my_sciexpem) #Calculate the standard deviation for an experiment group if the group doesn't exist in the csv file
    return standard_deviation

#Calculates the standrad deviation for an experiment group
def CalulateStdForExpGroup(my_execution, my_sciexpem):
    exp = my_execution.experiment
    exp_type = exp.experiment_type
    exp_fuels = dataStrTypeToList(exp.fuels)
    exp_reactor = exp.reactor
    my_experiments = my_sciexpem.filterDatabase(model_name = 'Experiment', experiment_type=exp_type, fuels=exp_fuels, reactor=exp_reactor) #Retrieve all experiments with the same measurement type, reactor type and fuels
        
    #Go through the experiment group and see if there are executions that use the same chemModel
    exec_list = []
    for exp in my_experiments:
        my_execs = my_sciexpem.filterDatabase(model_name='Execution', experiment=exp.id)
        for exec in my_execs:
            if exec.chemModel.name == my_execution.chemModel.name:
                exec_list.append(exec) #Add executions to list that use the same chemModel and experiment group
    
    #Calculate the errors for the experiment group and append them to list
    errors = []
    for i in range(0,len(exec_list)):
        exp_data, exec_data = FormatData(exec_list[i])
        diffs = CalculateErrors(exp_data, exec_data)
        errors.append(diffs)
        
    errors_np = np.array(errors, dtype=object) #Convert list to numpy array
    stack_errors = np.hstack(errors_np) #Stack the errors into a one dimensional array
    standard_deviation = np.std(stack_errors)
    AddStdToCsv(exp_type, exp_reactor, exp_fuels, standard_deviation) #Add the experiment group and standard deviation to the csv for future use

    return standard_deviation

#Add the experiment group and corresponding standard deviation to csv file
def AddStdToCsv(exp_t, exp_r, exp_f,std):
    data = {
    'Type': exp_t,
    'Reactor': exp_r,
    'Fuels': exp_f,
    'std': std }

    df = pd.DataFrame(data)
    df.to_csv('experiment_stds.csv', mode='a', index=False, header=False)

#Main function that is called to calculate the uncertainties for experiment data points
def CalculateUncertainty(my_execution, my_sciexpem):
    exp_data, exec_data = FormatData(my_execution) #Format the the data into normalized numpy arrays
    standard_deviation = ApproximateStd(my_execution, my_sciexpem) #Retrieve the standard deviation of an experiment group
    diffs = CalculateErrors(exp_data, exec_data) #Calculate the vertical euqlidian distances between experiment data and model 
    wanted_mean = 0 #The mean is 0 because the distance is 0 if the experiment data point is on the model
    z_scores = (diffs-wanted_mean)/standard_deviation

    max_value = 3 #0 as min and 3 as max value, because 3*std is when the value is "outside" the normal distribution
    min_value = 0
    z_scores_norm = (np.abs(z_scores) -min_value)/(max_value-min_value) 
    uncertainty = np.array([exp_data[0], exp_data[1], 1-z_scores_norm]) #Create numpy array with experiment data points and the corresponding uncertainty index
    #The uncertainty index is 1-z_scores_norm because we want the index to be 1 when the experiment data is on the ground truth (error is zero)

    WriteUncerToCsv(my_execution, diffs, uncertainty) #Collect uncertainties to csv file

    return uncertainty, exec_data, diffs

def WriteUncerToCsv(my_execution, diffs, uncertainty):
    with open('./csv_uncertainty.csv', 'a+') as f:
        writer = csv.writer(f)
        for i in range(len(uncertainty[0])):
            try:
                writer.writerow([my_execution.experiment.id,my_execution.experiment.file_paper.year,my_execution.experiment.file_paper.author,my_execution.chemModel.name,my_execution.chemModel.id,uncertainty[0][i],uncertainty[1][i],uncertainty[2][i],diffs[i]])
            except:
                pass

def CalculateErrors(exp_data, exec_data):
    #Approximating a line between every point in execution data and taking the vertical distance to the lines fro every experiment data point
    diffs = np.zeros(len(exp_data[0]))
    k = 0
    for i in range(0, len(exec_data[0])-1):
        for j in range(k, len(exp_data[0])):
            if exp_data[0,j] >= exec_data[0,i] and exp_data[0,j] <= exec_data[0,i+1]:
                p = np.polyfit(exec_data[0,i:i+2], exec_data[1,i:i+2], 1)
                fn = np.poly1d(p)
                diff = exp_data[1,j] - fn(exp_data[0,j]) #Calculates vertical euqlidian distance between experiment data and polyfit line

                diffs[j] = diff #Append difference to numpy array
            else:
                k = j
                break

    return diffs

#Format and normalize raw data to make it easier to work with
def FormatData(my_execution):
    #Format execution data into numpy arrays
    exec_data_x = my_execution.execution_columns[0].data
    exec_data_y = my_execution.execution_columns[1].data
    if (isinstance(my_execution.execution_columns[0].data, str)):
        exec_data_x = dataFloatTypeToArray(my_execution.execution_columns[0].data)
        exec_data_y = dataFloatTypeToArray(my_execution.execution_columns[1].data)
    exec_data_norm_x = normalizeArray(exec_data_x)
    exec_data_norm_y = normalizeArray(exec_data_y)

    #Format experiment data into numpy arrays
    exp_data_x = my_execution.experiment.data_columns[1].data
    exp_data_y = my_execution.experiment.data_columns[0].data
    if (isinstance(my_execution.experiment.data_columns[1].data, str)):
        exp_data_x = dataFloatTypeToArray(my_execution.experiment.data_columns[1].data)
        exp_data_y = dataFloatTypeToArray(my_execution.experiment.data_columns[0].data)
    exp_data_norm_x = normalizeArray(exp_data_x)
    exp_data_norm_y = normalizeArray(exp_data_y)

    exp_data = np.column_stack((exp_data_norm_x, exp_data_norm_y))
    exp_data = exp_data[np.argsort(exp_data[:, 0])].T #Sorting in ascending order
    exec_data = np.column_stack((exec_data_norm_x, exec_data_norm_y))
    exec_data = exec_data[np.argsort(exec_data[:, 0])].T #Sorting in ascending order
    
    return exp_data,exec_data

#Plot the experiment and execution data together with uncertainties and confidence interval
def PlotData(exec_data,uncertainty):
    plt.clf
    plt.figure(1)
    confidence = 0.95
    dof = len(exec_data[0]) - 1
    s = np.std(uncertainty)
    t_crit = np.abs(t.ppf((1 - confidence) / 2, dof))
    below_y = []
    above_y = []
    for item in range(len(exec_data[0])):
        below_y.append((exec_data[1][item] - s * t_crit / np.sqrt(len(exec_data[0]))))
        above_y.append((exec_data[1][item] + s * t_crit / np.sqrt(len(exec_data[0]))))
    plt.plot()

    below = np.array([exec_data[0], below_y])
    above = np.array([exec_data[0], above_y])
    plt.plot(exec_data[0], exec_data[1], '--', label='Model')
    plt.plot(below[0], below[1], '--', label='Below')
    plt.plot(above[0], above[1], '--', label='Above')
    plt.plot(uncertainty[0], uncertainty[1], '.', label='Experiment')
    plt.fill_between(above[0], below[1], exec_data[1], color='g', alpha=.1)
    plt.fill_between(above[0],  exec_data[1],above[1], color='g', alpha=.1)
    for i in range(0,len(uncertainty[2])):
        if uncertainty[2,i] <= 0:
            plt.annotate("Outlier ({:.2f})".format(uncertainty[2,i]), (uncertainty[0,i],uncertainty[1,i]))
        else:
            plt.annotate("{:.2f}".format(uncertainty[2,i]), (uncertainty[0,i],uncertainty[1,i]))
    plt.title("Plot of experiment data and model")
    plt.xlabel("Normalized Temperature")
    plt.ylabel("Normalized Ignition delay")
    plt.legend(loc="upper right")
    plt.show()
    # create random data
    np.random.seed(0)
    # create regression plot
    # x, y = np.random.multivariate_normal(mean, cov, 80).T
    # ax = sns.regplot(exec_data[1], uncertainty[1], ci=80)