# by introducing velocity tuning with this code I take the average over different realizations to have a more global
# valid tuning curve (similar to what would happen in an experimental setting)





import numpy as np
from functions_file import *
import os
import pandas as pd


def experiment():



    #variables time
    multiplying_factor_time = 100
    T = 100*multiplying_factor_time #milliseconds, total time
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


    #simulation of white noise

    diff_const = 2*np.pi*(10**(-3))*np.sqrt(2)*np.sqrt(sigma) #100*(10**(-9/2))/0.00063 #diffusion constant #
    mean_wn, std_wn = 0, 1

    rng = np.random.default_rng()  # Independent random generator
    seed_wn = rng.integers(0, 2**32)
    #seed_wn = 54
    np.random.seed(seed_wn)
    wnoise_value = diff_const*np.sqrt(dt_position)*np.random.normal(mean_wn, std_wn, size = N_time_points)



    #Ornstein-Uhlenbeck process 
    #((note that this t is not really the time but more the index indicator))

    for t in range(T0, N_time_points-1):   #the minus 1 is necessary because an array (21,1) has row indeces that goes from 0 to 20

        dX[t] = sigma*(mu_X - X[t])*dt_position + wnoise_value[t]

        X[t + 1] = X[t] + dX[t] #[rad/ms]


   

    #second time integration to derive HD
    Y0 = 0     #head starting direction
    Y = np.zeros((N_time_points, 1))
    Y[0] = Y0       

    for t in range(T0, N_time_points-1):
        Y[t+1] = Y[t] + X[t]*dt_position


    #define the HD as a circular variable
    theta_HD = Y % (2*np.pi) - np.pi #theta goes from -pi to pi

    N_neurons = 50 #numebr of neurons

    #let's define the preferred head directions of all the neurons
    #consider that we need to esclude one of the edges otherwise the first and last neuron would have the same preferred head direction
    neuro_dir_tomod, step = np.linspace(-np.pi, np.pi, (N_neurons + 1), retstep= True)
    neuro_pref_dir_vector = neuro_dir_tomod[1:]


    #Von Mises Functions parameters, VonMisesFunction(a, b, k, w, d_i)
    a = 0
    r_max= 0.01 # units [ms] #10 spikes/s peak firing rate
    k = 5 

    #Velocity tuning
    r_max_Hz = r_max *(10**3)

    #variation of 20% from the maximum rate of 10spikes/sec
    maximum = r_max_Hz + 0.2*r_max_Hz
    minimum = r_max_Hz - 0.2*r_max_Hz

    #to determine the angular coeffiecient we used observations form Finkelstein et al.(2019)
    #around 60degrees/sec they have the maximum in rate
    ang_coeff = (maximum - minimum)/(np.pi/3)

    #VELOCITY DATA [rad/sec]
    velocity_data = X*(10**3)
    rate_max_depending_velocity = np.zeros_like(velocity_data)

    # Choose model
    model_type = "NO-TUNING"  # Change to "L-SHAPED" ,"V-SHAPED", "NO-TUNING" for the model with no vellcity tuning

    if model_type == "V-SHAPED":
        rate_max_depending_velocity[velocity_data < 0] = -ang_coeff * velocity_data[velocity_data < 0] + minimum
        rate_max_depending_velocity[velocity_data > 0] = ang_coeff * velocity_data[velocity_data > 0] + minimum
        rate_max_depending_velocity[velocity_data == 0] = minimum


    elif model_type == "L-SHAPED":
        #points are ranodmly selected so that half of them will be assigned to theL-sx distribution
        #and the other half to the L_dx distribution, so there is a better balance between clockwise and anticlockwise orientations

        # Total number of points in X
        n_points = len(X)
        # Number of elements to assign to L_sx and L_dx
        num_L_sx = int(n_points * 0.5)  # 50% to L_sx and 50% to L_dx
        # Randomly select indices for L_sx (without shuffling X)
        L_sx_indices = np.random.choice(np.arange(n_points), num_L_sx, replace=False)
        L_dx_indices = np.setdiff1d(np.arange(n_points), L_sx_indices)  # The rest go to L_dx
        # Apply the rules for L_sx
        rate_max_depending_velocity[L_sx_indices] = np.where(X[L_sx_indices] > 0, minimum, -ang_coeff * X[L_sx_indices] + minimum)
        # Apply the rules for L_dx
        rate_max_depending_velocity[L_dx_indices] = np.where(X[L_dx_indices] < 0, minimum, ang_coeff * X[L_dx_indices] + minimum)
        '''
        rate_max_depending_velocity[velocity_data < 0] = minimum
        rate_max_depending_velocity[velocity_data > 0] = ang_coeff * velocity_data[velocity_data > 0] + minimum
        rate_max_depending_velocity[velocity_data == 0] = minimum
        '''
    elif model_type == "NO-TUNING":
        rate_max_depending_velocity = r_max_Hz*np.ones_like(velocity_data)
    
    model_info_path = os.path.join("velocity_tuning_exp_outputs", "model_info.txt")
    if not os.path.exists(model_info_path):
        with open(model_info_path, "w") as f:
            f.write(f"Model used: {model_type}\n")  # Inform the user about the model type

    #rates matrix

    rates_matrix = np.zeros((N_neurons, N_time_points))

    theta_HD_reshaped = theta_HD.ravel()   #this is to fratten the array from 2D (21,1) to 1D (21,) because matrix_rates[n,:] is of type 1D
    rate_max_depending_velocity = rate_max_depending_velocity.ravel()

    
    for n in range(0, N_neurons):
        dir_neuro = neuro_pref_dir_vector[n]
        rates_matrix[n,:] = VonMisesFunction(a,(rate_max_depending_velocity* 10**(-3)),k,theta_HD_reshaped, dir_neuro)

    #poisson spiking

    poisson_prob_neuro = rates_matrix*dt_spikewindow

    rng = np.random.default_rng()  # Independent random generator
    seed_poiss = rng.integers(0, 2**32) 
    spike_matrix_01 = inhomog_poisson_spike_gen(poisson_prob_neuro, seed_poiss)


    total_spike_count = sum(sum(spike_matrix_01))


    decoding_time_interval_otb_alpha = int(50/dt_position) #int(50/dt_position)
    time_overlapping_otb_alpha = int(10/dt_position)

    def calculate_mse(alpha):
    # Same steps as in the for loop above, but now as a function of alpha
        dec_pos_overlbins_alpha_compxy, norm_rex_vector_alpha, spike_count_alpha, pop_vec = resultant_vector_overlbins_alpha(
            spike_matrix_01,
            neuro_pref_dir_vector,
            decoding_time_interval_otb_alpha,
            time_overlapping_otb_alpha,
            alpha
        )

        dec_pos_overlbins_alpha = np.arctan2(dec_pos_overlbins_alpha_compxy[:, 1], dec_pos_overlbins_alpha_compxy[:, 0])
        dec_pos_overlbins_alpha_02pi = dec_pos_overlbins_alpha + np.pi

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

    

    #reordering theta so that data goes from -np.pi to np.pi
    #and the same ordering has to be applied to rate as well

    #indices that would sort theta_HD_reshaped
    sorted_indices = np.argsort(theta_HD_reshaped)

    #same sorting to both arrays
    sorted_theta_HD = theta_HD_reshaped[sorted_indices]
    sorted_rate_max = rate_max_depending_velocity[sorted_indices]

    rates_matrix_sorted = np.zeros((N_neurons, N_time_points))
    for n in range(0, N_neurons):
        dir_neuro = neuro_pref_dir_vector[n]
        rates_matrix_sorted[n,:] = VonMisesFunction(a,(sorted_rate_max* 10**(-3)),k,sorted_theta_HD, dir_neuro)

    #poisson spiking

    poisson_prob_neuro_sorted = rates_matrix_sorted*dt_spikewindow
    spike_matrix_01_sorted = inhomog_poisson_spike_gen(poisson_prob_neuro_sorted, seed_poiss)





    spike_count_neuron30 = spike_matrix_01_sorted[30, :]
    spike_count_neuron30 = spike_count_neuron30.flatten()

    rate_depending_velocity_data = rate_max_depending_velocity.flatten()
    neuro_pref_dir_data = neuro_pref_dir_vector.flatten()
    mse_data = min_mse/total_spike_count  #the MSE is normalized for the total number of spikes

    return neuro_pref_dir_data, rate_depending_velocity_data, mse_data, spike_count_neuron30, sorted_theta_HD

if __name__ == "__main__":
    #SAVING DATA OF THE RUN IN A CSV FILE
    neuro_pref_dir_data, rate_max_vel_data, mse_data, spike_count_n30_data, sorted_theta_HD_data = experiment()
    
    # Ensure the output directory exists
    output_dir = "velocity_tuning_exp_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Define the file names and corresponding data
file_data_pairs = [
    ("neuro_pref_dir_data.csv", neuro_pref_dir_data),
    ("rate_max_vel_data.csv", rate_max_vel_data),
    ("mse_data.csv", mse_data), 
    ("spike_count_n30_data.csv", spike_count_n30_data),
    ("sorted_theta_HD_data.csv", sorted_theta_HD_data)
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



