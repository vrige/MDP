import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from os.path import exists
from scipy.stats import t
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



def dataFloatTypeToArray(dataStr): 
    values_str = dataStr.replace('[', '').replace(']', '').split(', ')
    values = [float(i) for i in values_str]
    return np.array(values)

def dataStrTypeToList(dataStr):
    values_str = dataStr.replace('[', '').replace(']', '').replace('"', '').split(', ')
    values = [str(i) for i in values_str]
    return values

def normalizeArray(ar):
    return (ar - np.amin(ar))/(np.amax(ar)- np.amin(ar))

def CheckIfExpInCsv(exec):
    exp = exec.experiment
    exp_t = exp.experiment_type
    exp_f = dataStrTypeToList(exp.fuels)
    exp_r = exp.reactor

    if exists('experiment_stds.csv'):
        df = pd.read_csv('experiment_stds.csv')
        if not df.empty:
            exp = df.loc[(df['Type'] == exp_t) & (df['Reactor'] == exp_r) & (df['Fuels'] == exp_f)]
            if not exp.empty:
                return float(exp['std'])
            else:
                return 0
        else:
            return 0
    else:
        print("Collection of standard deviations per experiment group does not exist!")
        return 0

def ApproximateStd(my_execution, my_sciexpem):
    standard_deviation = CheckIfExpInCsv(my_execution)
    if not standard_deviation:
        standard_deviation = CalulateStdForExpGroup(my_execution, my_sciexpem)
    return standard_deviation

def CalulateStdForExpGroup(my_execution, my_sciexpem):
    exp = my_execution.experiment
    exp_type = exp.experiment_type
    exp_fuels = dataStrTypeToList(exp.fuels)
    exp_reactor = exp.reactor
    my_experiments = my_sciexpem.filterDatabase(model_name = 'Experiment', experiment_type=exp_type, fuels=exp_fuels, reactor=exp_reactor)
        
    exec_list = []
    for exp in my_experiments:
        my_execs = my_sciexpem.filterDatabase(model_name='Execution', experiment=exp.id)
        for exec in my_execs:
            if exec.chemModel.name == my_execution.chemModel.name:
                exec_list.append(exec)
        
    errors = []
    for i in range(0,len(exec_list)):
        exp_data, exec_data = FormatData(exec_list[i])
        diffs = CalculateErrors(exp_data, exec_data)
        errors.append(diffs)
        
    errors_np = np.array(errors, dtype=object)
    stack_errors = np.hstack(errors_np)
    standard_deviation = np.std(stack_errors)
    AddStdToCsv(exp_type, exp_reactor, exp_fuels, standard_deviation)

    return standard_deviation

def AddStdToCsv(exp_t, exp_r, exp_f,std):
    data = {
    'Type': exp_t,
    'Reactor': exp_r,
    'Fuels': exp_f,
    'std': std }

    df = pd.DataFrame(data)
    df.to_csv('experiment_stds.csv', mode='a', index=False, header=False)


def CalculateUncertainty(my_execution, my_sciexpem):
    exp_data, exec_data = FormatData(my_execution)
    standard_deviation = ApproximateStd(my_execution, my_sciexpem)
    diffs = CalculateErrors(exp_data, exec_data)

    wanted_mean = 0 #The mean is 0 because the distance is 0 if the experiment data point is on the model
    z_scores = (diffs-wanted_mean)/standard_deviation

    max_value = 3 #0 as min and 3 as max value, because 3*std is when the value is "outside" the normal distribution
    min_value = 0
    z_scores_norm = (np.abs(z_scores) -min_value)/(max_value-min_value) 
    uncertainty = np.array([exp_data[0], exp_data[1], 1-z_scores_norm])

    return uncertainty, exec_data, diffs

def CalculateErrors(exp_data, exec_data):
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

    return diffs

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

def PlotData(exec_data,uncertainty):
    #Plot data

    # fig, ax = plt.subplots()
    # ax.plot(x, y)
    # ax.fill_between(x, (y - ci), (y + ci), color='b', alpha=.1)
    plt.clf
    plt.figure(1)
    confidence = 0.95
    dof = len(exec_data[0]) - 1
    s = np.std(uncertainty)
    print("STD deviation: ", s)
    t_crit = np.abs(t.ppf((1 - confidence) / 2, dof))
    print("T crit: ", t_crit)
    below_y = []
    above_y = []
    print("Exec data: ",exec_data)
    print("Len: ",len(exec_data[0]))
    for item in range(len(exec_data[0])):
        # print("Below 1: ",item[1] - s * t_crit / np.sqrt(len(exec_data)))
        # print("Below 2: ",item[0]+(item[0] - s * t_crit / np.sqrt(len(exec_data))))
        # print("Above 1: ", item[0] + s * t_crit / np.sqrt(len(exec_data)))
        # print("Above 2: ", item[0] + (item[0] + s * t_crit / np.sqrt(len(exec_data))))
        below_y.append((exec_data[1][item] - s * t_crit / np.sqrt(len(exec_data[0]))))
        above_y.append((exec_data[1][item] + s * t_crit / np.sqrt(len(exec_data[0]))))
    plt.plot()
    print("ab : ",above_y)
    print("b : ", below_y)
    below = np.array([exec_data[0], below_y])
    above = np.array([exec_data[0], above_y])
    print("below:",below)
    print("Exec:",exec_data)
    plt.plot(exec_data[0], exec_data[1], '--', label='Model')
    plt.plot(below[0], below[1], '--', label='Below')
    plt.plot(above[0], above[1], '--', label='Above')
    plt.plot(uncertainty[0], uncertainty[1], '.', label='Experiment')
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