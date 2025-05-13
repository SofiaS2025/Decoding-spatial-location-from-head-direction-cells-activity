#TO REPRODUCE THE GRAPHS IN THE REPORT:
# - multiplying_factor_time = 25 OR multiplying_factor_time = 200 to allow experiments of 25 or 200 seconds
# - seed_wn = np.random.randint(0, 2**32) np.random.seed(seed_wn)
    #  OR
    #seed_wn = 54 
    # to allow different trajectories or same trajectory for all experiments in the same run




import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Ellipse
import statistics as stat
from scipy.stats import vonmises
import math
#import pycircular as pyc
from functions_file import *
from scipy import interpolate
from statsmodels.tsa.stattools import acf
from scipy.optimize import curve_fit
import scipy.special as sp
import pandas as pd
import os


def experiment():


    #ENCODING

    #variables time
    multiplying_factor_time = 200    #25
    T = 1000*multiplying_factor_time #milliseconds, total time 
    #note to self: increasing considerably the itme slows down the code
    T0 = 0 #ms
    dt_position = 1 #ms
    dt_spikewindow = 1 #ms

    N_steps = int(np.floor(T/dt_position))
    T_rounded = int(N_steps*dt_position)


    #the number of time points is different from the number of steps. In the array (0,1,2) there are 2 steps and 3 points
    #if we wanna include in the count both the edges, because our starting point number is 0 (if it was 1 that is not necessary)
    N_time_points = N_steps+1


    #the function linspace creates a linearly spaced points INCLUDING both 0 and T_rounded, so the number of data points is N_steps + 1
    T_vec0 = np.linspace(0, T_rounded, N_time_points)  


    #reshaping the time vector 
    T_vec = T_vec0.reshape(T_vec0.size, 1)

    #variable initialization
    sigma = 0.05  #[1/ms] #this would be in order to have a correlation time of about 20 ms, so the decoding time about 10 ms would be valid
    mu_X = 0 #0 #DRIFT #note: the velocity is directional +/-
    #theta_X = 10 #threshold

    X0 = mu_X #initial velocity equal to the mean velocity to avoid initial setting behaviour
    X = np.zeros((N_time_points, 1))
    dX = np.zeros((N_time_points, 1))
    X[0] = X0


    #RANDOMNESS
    seed_wn = np.random.randint(0, 2**32)
    np.random.seed(seed_wn)
    #seed_wn = 54

    #simulation of white noise
    diff_const = 2*np.pi*(10**(-3))*np.sqrt(2)*np.sqrt(sigma) #100*(10**(-9/2))/0.00063 #diffusion constant #
    mean_wn, std_wn = 0, 1

    np.random.seed(seed_wn)
    wnoise_value = diff_const*np.sqrt(dt_position)*np.random.normal(mean_wn, std_wn, size = N_time_points)



    #Ornstein-Uhlenbeck process 
    #((note that this t is not really the time but more the index indicator))
    for t in range(T0, N_time_points-1):   #the minus 1 is necessary because an array (21,1) has row indeces that goes from 0 to 20
        dX[t] = sigma*(mu_X - X[t])*dt_position + wnoise_value[t]
        X[t + 1] = X[t] + dX[t]

    #second time integration to derive HD
    Y0 = 0     #head starting direction
    Y = np.zeros((N_time_points, 1))
    Y[0] = Y0       
    for t in range(T0, N_time_points-1):
        Y[t+1] = Y[t] + X[t]*dt_position

    #define the HD as a circular variable
    theta_HD = Y % (2*np.pi) - np.pi #theta goes from -pi to pi


    #SPIKE GENERATION
    N_neurons = 50 #number of neurons

    #let's define the preferred head directions of all the neurons
    #consider that we need to esclude one of the edges otherwise the first and last neuron would have the same preferred head direction
    
    neuro_dir_tomod = np.linspace(-np.pi, np.pi, (N_neurons + 1), retstep= False)
    neuro_pref_dir_vector = neuro_dir_tomod[1:]
    

    #Von Mises Functions parameters, VonMisesFunction(a, b, k, w, d_i)
    a = 0
    r_max= 0.01 # units [ms] #10 spikes/s peak firing rate
    k = 5 

    N_points_function = 100
    x_points = np.linspace(-np.pi, np.pi, N_points_function)

    tuning_curves = np.zeros((N_neurons, N_points_function)) #initialization

    for i in range(0, N_neurons):

        pref_HD_eachneuron = neuro_pref_dir_vector[i]
        tuning_curves[i, :] = VonMisesFunction(a,r_max,k,x_points,pref_HD_eachneuron)




    #initialize matrix
    rates_matrix = np.zeros((N_neurons, N_time_points))

    theta_HD_reshaped = theta_HD.ravel()   #this is to fratten the array from 2D (21,1) to 1D (21,) because matrix_rates[n,:] is of type 1D

    for n in range(0, N_neurons):
        dir_neuro = neuro_pref_dir_vector[n]
        rates_matrix[n,:] = VonMisesFunction(a,r_max,k,theta_HD_reshaped, dir_neuro)



    #let's create a probability graph for a certain time point t

    poisson_prob_neuro = rates_matrix*dt_spikewindow

    neurons = np.linspace(1,N_neurons, N_neurons)
    neurons = neurons.astype(int)

    # SPIKES GENERATION
    rng = np.random.default_rng()  # Independent random generator
    seed_poiss = rng.integers(0, 2**32) 
    spike_matrix_01 = inhomog_poisson_spike_gen(poisson_prob_neuro, seed_poiss)


    # DECODING

    # Decoding with/without overlapping time bins and wiegthed past and present point
    #varibles
    decoding_time_interval_otb_alpha = int(50/dt_position) #int(50/dt_position)
    time_overlapping_otb_alpha = int(10/dt_position)

    # Initialize results
    alpha_values = np.linspace(0.01, 1.0, 50)  # Adjust range and step as needed
    mse_values = []

    num_bins_to_exclude_otb_alpha = decoding_time_interval_otb_alpha // time_overlapping_otb_alpha
    resultant_mean_real_positions = mean_position_vector(theta_HD, time_overlapping_otb_alpha)
    real_mean_HD = resultant_mean_real_positions[:-num_bins_to_exclude_otb_alpha].flatten()

    #RMSE over alpha value


    # Initialize results
    alpha_values = np.linspace(0.01, 1.0, 50)  # Adjust range and step as needed
    mse_values = []

    num_bins_to_exclude_otb_alpha = decoding_time_interval_otb_alpha // time_overlapping_otb_alpha
    resultant_mean_real_positions = mean_position_vector(theta_HD, time_overlapping_otb_alpha)
    real_mean_HD = resultant_mean_real_positions[:-num_bins_to_exclude_otb_alpha].flatten()



    # Loop over alpha values
    for alpha in alpha_values:
        
        alpha_value = alpha
        dec_pos_overlbins_alpha_compxy, norm_rex_vector_alpha, spike_count_alpha, pop_vec_alpha = resultant_vector_overlbins_alpha(spike_matrix_01,
                                                                                                                neuro_pref_dir_vector,
                                                                                                                decoding_time_interval_otb_alpha,
                                                                                                                time_overlapping_otb_alpha,
                                                                                                                alpha_value)

        dec_pos_overlbins_alpha = np.arctan2(dec_pos_overlbins_alpha_compxy[:,1], dec_pos_overlbins_alpha_compxy[:,0]) 
        dec_pos_overlbins_alpha_02pi = dec_pos_overlbins_alpha + np.pi #so now the angles range from 0 to 2\pi

        

        res_pos_otb_alpha = -np.pi*np.ones_like(real_mean_HD) + [(dec_pos_overlbins_alpha- real_mean_HD
            + np.pi*np.ones_like(real_mean_HD)) % (2*np.pi)]

        sum_res_RS_otb_alpha_x = (np.sum((np.cos(res_pos_otb_alpha))**2)/(res_pos_otb_alpha.size))
        sum_res_RS_otb_alpha_y = (np.sum((np.sin(res_pos_otb_alpha))**2)/(res_pos_otb_alpha.size))

        #Squared root of Sum Residuals Squared (SRSS)
        MSE_otb_alpha = np.arctan2(sum_res_RS_otb_alpha_y, sum_res_RS_otb_alpha_x)
        mse_values.append(MSE_otb_alpha)


    def calculate_mse(alpha):
        # Same steps as in the for loop above, but now as a function of alpha
        dec_pos_overlbins_alpha_compxy, norm_rex_vector_alpha, spike_count_alpha, pop_vec_alpha = resultant_vector_overlbins_alpha(
            spike_matrix_01,
            neuro_pref_dir_vector,
            decoding_time_interval_otb_alpha,
            time_overlapping_otb_alpha,
            alpha
        )

        dec_pos_overlbins_alpha = np.arctan2(dec_pos_overlbins_alpha_compxy[:, 1], dec_pos_overlbins_alpha_compxy[:, 0])
        
        num_bins_to_exclude_otb_alpha = decoding_time_interval_otb_alpha // time_overlapping_otb_alpha
        resultant_mean_real_positions = mean_position_vector(theta_HD, time_overlapping_otb_alpha)

        res_pos_otb_alpha = -np.pi * np.ones_like(resultant_mean_real_positions[:-num_bins_to_exclude_otb_alpha].flatten()) + [
            (dec_pos_overlbins_alpha - resultant_mean_real_positions[:-num_bins_to_exclude_otb_alpha].flatten()
            + np.pi * np.ones_like(resultant_mean_real_positions[:-num_bins_to_exclude_otb_alpha].flatten())) % (2 * np.pi)]

        sum_res_RS_otb_alpha_x = (np.sum((np.cos(res_pos_otb_alpha)) ** 2) / (res_pos_otb_alpha.size))
        sum_res_RS_otb_alpha_y = (np.sum((np.sin(res_pos_otb_alpha)) ** 2) / (res_pos_otb_alpha.size))
        
        MSE_otb_alpha = np.arctan2(sum_res_RS_otb_alpha_y, sum_res_RS_otb_alpha_x)

        return MSE_otb_alpha


    # Use scipy's minimize_scalar
    result = minimize_scalar(calculate_mse, bounds=(0.001, 1.0), method='bounded')
    best_alpha = result.x
    min_mse = result.fun



    alpha_minimum = best_alpha #0.196 * (1/dt_position) * (abs(mu_X) + 1)
    
    dec_pos_overlbins_alpha_compxy, norm_rex_vector_alpha, spike_count_alpha, pop_vec_alpha= resultant_vector_overlbins_alpha(spike_matrix_01,neuro_pref_dir_vector,decoding_time_interval_otb_alpha,time_overlapping_otb_alpha,alpha_minimum)

    dec_pos_overlbins_alpha = np.arctan2(dec_pos_overlbins_alpha_compxy[:,1], dec_pos_overlbins_alpha_compxy[:,0]) 
    
    num_bins_to_exclude_otb_alpha = decoding_time_interval_otb_alpha // time_overlapping_otb_alpha

    
    resultant_mean_real_positions = mean_position_vector(theta_HD, time_overlapping_otb_alpha)
    mean_real_pos_flatten = resultant_mean_real_positions[:-num_bins_to_exclude_otb_alpha].flatten()
    res_pos_otb_alpha = -np.pi*np.ones_like(mean_real_pos_flatten) + [(dec_pos_overlbins_alpha- mean_real_pos_flatten
        + np.pi*np.ones_like(mean_real_pos_flatten)) % (2*np.pi)]


    #sum Root Squared of residuals/distances components 
    sum_res_RS_otb_alpha_x = np.sqrt(np.sum((np.cos(res_pos_otb_alpha))**2)/(res_pos_otb_alpha.size))
    sum_res_RS_otb_alpha_y = np.sqrt(np.sum((np.sin(res_pos_otb_alpha))**2)/(res_pos_otb_alpha.size))


    #Squared root of Sum Residuals Squared (SRSS)
    MSE_dec = np.arctan2(sum_res_RS_otb_alpha_y, sum_res_RS_otb_alpha_x)
    


    #variance
    C = np.cos(dec_pos_overlbins_alpha)
    S = np.sin(dec_pos_overlbins_alpha)

    weights = norm_rex_vector_alpha/np.sum(norm_rex_vector_alpha)
    normalization_factor = 1/( norm_rex_vector_alpha.size )
    R = normalization_factor*np.sqrt((np.sum(weights*C))**2 + (np.sum(weights*S))**2)


    
    
    
    if alpha_values.ndim != 1: alpha_values.ravel()
    #if mse_values.ndim != 1: mse_values.ravel()
    neuro_pref_dir_data = neuro_pref_dir_vector.flatten()
    pop_vec_data = pop_vec_alpha.flatten()


    return alpha_values, mse_values, alpha_minimum, dec_pos_overlbins_alpha, mean_real_pos_flatten, neuro_pref_dir_data, pop_vec_data

if __name__ == "__main__":
    #SAVING DATA OF THE RUN IN A CSV FILE
    alpha_data, mse_data, alpha_min_data, dec_pos_data, real_pos_data, neuro_pref_dir_data, pop_vec_data = experiment()
    
    # Ensure the output directory exists
    output_dir = "experiment_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Define the file names and corresponding data
file_data_pairs = [
    ("alpha_data.csv", alpha_data),
    ("mse_data.csv", mse_data),
    ("alpha_min_data.csv", alpha_min_data),
    ("dec_pos_data.csv", dec_pos_data),
    ("real_pos_data.csv", real_pos_data), 
    ("neuro_pref_dir_data.csv", neuro_pref_dir_data),
    ("pop_vec_data.csv", pop_vec_data)
]

# Process each file and its data
for file_name, data in file_data_pairs:
    # Construct the file path dynamically
    file_path = os.path.join(output_dir, file_name)

    if not os.path.exists(file_path):
        # Initialize with experiment data (columns: Experiment_1)
        # Check if 'data' is a scalar
        if np.isscalar(data):
            # If data is a scalar, create a DataFrame with an explicit index
            df = pd.DataFrame({f"Experiment_{1}": [data]}, index=[0])  # Explicit index
        else:
            df = pd.DataFrame({f"Experiment_{1}": data})
            
        df.to_csv(file_path, index=False)
    else:
        # Append the new data as a new column for each experiment
        df = pd.read_csv(file_path)
        df[f"Experiment_{len(df.columns) + 1}"] = data
        df.to_csv(file_path, index=False)

    #print(f"Dist data and norm data saved in {output_dir}")

