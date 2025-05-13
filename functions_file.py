#se hai warnings quando crei un file con una funzione che poi ti serve in un altro file ricordati di AGGIUNGERE IL FILE AL WORKSPACE
#just go to File > add folder to qorkspace


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Ellipse
from scipy.optimize import minimize_scalar

def VonMisesFunction(a, b, k, w, d_i): #VonMisesFunction(a, b, k, lambda_i, w, d_i):
    #parameters = np.array()  #same as above

    #note that python does element-wise operations if all arrays have the same shape so b,w, d_i possibly
    
    omega_i = np.zeros_like(w.size)
    omega_i = a + b*np.exp(k*(np.cos(w-d_i) -1))

    

    #omega_i = a + b*np.exp(k*(np.cos(2*np.pi*(w-d_i)/lambda-i) -1))
    
    return omega_i

def set_neuron_ticks(neuron_index):
    
    
    # Calculate ticks every 5 points within the range of neuron_index
    new_list = np.arange(-1, neuron_index[-1] + 1, 5)
    neuron_ticks = np.insert(new_list[1:],0,0)
    # Generate labels for ticks
    neuron_ticks_label = neuron_ticks+1
    
    #neuron_index_with0 = neuron_index -1
    #neuron_ticks = [neuron_index_with0[0],(neuron_index[-1]/2)-1 , neuron_index_with0[-1]]
    #neuron_ticks_label = [neuron_index[0], int(neuron_index[-1]/2), neuron_index[-1]]

    return neuron_ticks, neuron_ticks_label

def set_time_ticks(time_vector, number_steps, shortening_ticks_factor):
    
    time_points_shorterned = int((number_steps/shortening_ticks_factor)+1)
    ticks_time_points = np.squeeze(np.linspace(time_vector[0], (time_vector.size-1), time_points_shorterned))
    ticks_time_points_label = np.squeeze(np.linspace(time_vector[0], time_vector[-1], time_points_shorterned))

    return ticks_time_points, ticks_time_points_label

def probability_colormap(data, time_vector, number_steps, shortening_ticks_factor, neuron_index):

    neuron_ticks, neuron_ticks_label = set_neuron_ticks(neuron_index)

    ticks_time_points, ticks_time_points_label = set_time_ticks(time_vector, number_steps, shortening_ticks_factor)

    colors = [ "lightseagreen", "lawngreen","gold" ,"darkorange"]
    cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)

    fig, ax = plt.subplots(figsize=(12,12))
    cax = ax.imshow(data, cmap=cmap1) 

    cbar = fig.colorbar(cax, ax=ax,ticks=[np.min(data), (np.max(data)- np.min(data))/2, np.max(data)],orientation="horizontal") #, ticks=[-0, 0.5, 1]
    #cbar.ax.set_yticklabels([ '0', '0.5', '1'])

    ax.set_xlabel('time t [ms]')
    ax.set_xticks(ticks_time_points)
    ax.set_xticklabels(ticks_time_points_label)
    ax.set_ylabel('neuron index')
    ax.set_yticks(neuron_ticks)
    ax.set_yticklabels(neuron_ticks_label)
    ax.set_title('Probablity of spiking over time')

def inhomog_poisson_spike_gen(data, seed):

    """
    Generate spike times for a neuron with a time-dependent firing rate using an inhomogeneous Poisson process.
    
    Parameters:
    T (float): Total duration of the simulation (seconds).
    dt (float): Time step for simulation (seconds).
    
    Returns:
    spike_times (list): List of spike times.
    """
    np.random.seed(seed)
    random_values = np.random.rand(data.shape[0], data.shape[1])

    # Initialize the logical matrix with the same shape as 'data' (but will hold integers 0 and 1)
    logical_matrix = np.zeros(data.shape, dtype=int)

    # Use a nested for loop to compare each element in 'data' to the corresponding random value
    for i in range(data.shape[0]):         # Loop through rows
        for j in range(data.shape[1]):     # Loop through columns
            logical_matrix[i, j] = int(data[i, j] > random_values[i, j])
    
    return logical_matrix

def average_distance_per_row(matrix):
    # List to store the average distances for each row
    average_distances = []

    # Iterate through each row in the matrix
    for row in matrix:
        # Step 1: Find the positions of all the 1s in the row
        positions = [index for index, value in enumerate(row) if value == 1]

        # Step 2: Calculate distances between consecutive 1s
        if len(positions) < 2:
            # If fewer than 2 ones, no meaningful distance can be calculated
            average_distances.append(0)
        else:
            total_distance = sum(positions[i+1] - positions[i] for i in range(len(positions) - 1))
            average_distance = total_distance / (len(positions) - 1)
            average_distances.append(average_distance)

    return average_distances

def exponential_decay(lag, tau, A):
    return A * np.exp(-lag / tau)

def exponential_function(lag, tau, A):
    return A * np.exp(lag / tau)

def mean_position_vector(real_positions, decoding_time_interval):
    # Get the number of rows in real_positions
    num_columns = real_positions.shape[0]  

    # Calculate the number of intervals
    num_intervals = (num_columns + decoding_time_interval - 1) // decoding_time_interval

    # Initialize an array to hold the mean position vectors for each interval
    mean_position_vectors_comp = np.zeros((num_intervals, 2))  # Two components: x and y
    mean_position_vectors = np.zeros((num_intervals, 1))
    # Iterate over each interval
    for interval_start in range(0, num_columns, decoding_time_interval):
        # Define the start and end for the current interval
        interval_end = min(interval_start + decoding_time_interval, num_columns)

        # Initialize accumulators for the resultant vector components
        resultant_vec_x = 0
        resultant_vec_y = 0

        interval_index = (interval_start ) // decoding_time_interval

        # Process each column in the current interval
        for col_index in range(interval_start, interval_end):
            resultant_vec_x += np.sum(np.cos(real_positions[col_index]))  
            resultant_vec_y += np.sum(np.sin(real_positions[col_index]))

        norm_res_vector = np.sqrt(resultant_vec_x ** 2 + resultant_vec_y ** 2)
        if norm_res_vector != 0:  # Normalize only if the norm is not zero
            mean_position_vectors_comp[interval_index, 0] = resultant_vec_x / norm_res_vector
            mean_position_vectors_comp[interval_index, 1] = resultant_vec_y / norm_res_vector
    
    mean_position_vectors = np.arctan2(mean_position_vectors_comp[:,1], mean_position_vectors_comp[:,0]) #it goes from [-pi, tp pi]

    return mean_position_vectors


def resultant_vector_overlbins_alpha(spike_matrix_01,neuro_pref_dir_vector,decoding_time_interval,time_overlapping,alpha):

    num_columns = len(spike_matrix_01[0])
    # Initialize the resultant vector array with zeros for each interval
    # Assuming there are ceil(num_columns / decoding_time_interval) intervals

    num_intervals = ((num_columns + time_overlapping - 1) // time_overlapping)

    norm_res_vector = np.zeros((num_intervals, 1))
    resultant_ang_comp_xy = np.zeros((num_intervals, 2))
    spike_count = np.zeros((num_intervals, 1))
    population_vector_angles = np.zeros((num_intervals, 1))

        # Iterate over each interval of columns, starting from column 0
    for interval_start in range(0, num_columns, time_overlapping):
        interval_end = min(interval_start + decoding_time_interval, num_columns)
        interval_index = interval_start // time_overlapping
                # Initialize the resultant vector components for the interval
        resultant_vec_x_interval = 0
        resultant_vec_y_interval = 0
                
                # Process each column within the current interval and accumulate
        for col_index in range(interval_start, interval_end):

            spike_count[interval_index] += np.sum(spike_matrix_01[:,col_index])

                    # Step 1: Find the positions of all the 1s in the current column
            positions = [row_index for row_index, row in enumerate(spike_matrix_01) if row[col_index] == 1]

                    # Get the angular positions for the spikes in the current column
            estracted_ang_position = neuro_pref_dir_vector[positions]
                    
                    # Sum the x and y components for the current column and add them to the interval totals
            resultant_vec_x_interval += sum(np.cos(estracted_ang_position))
            resultant_vec_y_interval += sum(np.sin(estracted_ang_position))
                
                # Calculate the norm of the resultant vector for the entire interval
        norm_res_vector_interval = np.sqrt(resultant_vec_x_interval**2 + resultant_vec_y_interval**2)
                
        population_vector_angle = np.arctan2(resultant_vec_y_interval, resultant_vec_x_interval)
        population_vector_angles[interval_index] = population_vector_angle
                
                # Check if the norm of the resultant vector is zero
        if norm_res_vector_interval == 0:
                    # Handle missing data by interpolation if there are previous intervals
            if interval_index - 2 < 0:  # If there aren't enough previous intervals
                resultant_ang_comp_xy[interval_index, 0] = 0
                resultant_ang_comp_xy[interval_index, 1] = 0
            else:
                        # Calculate the slope for linear interpolation based on previous intervals
                slope_x = resultant_ang_comp_xy[interval_index - 1, 0] - resultant_ang_comp_xy[interval_index - 2, 0]
                slope_y = resultant_ang_comp_xy[interval_index - 1, 1] - resultant_ang_comp_xy[interval_index - 2, 1]
                        
                        # Estimate the missing point by extrapolating the last known trend
                resultant_ang_comp_xy[interval_index, 0] = resultant_ang_comp_xy[interval_index - 1, 0] + slope_x
                resultant_ang_comp_xy[interval_index, 1] = resultant_ang_comp_xy[interval_index - 1, 1] + slope_y
        else:
                    # Normalize and store the resultant vector for the interval
                    #normalizing_factor = 1 #/ norm_res_vector
            if interval_index == 0:
                resultant_ang_comp_xy[interval_index, 0] = resultant_vec_x_interval 
                resultant_ang_comp_xy[interval_index, 1] = resultant_vec_y_interval
                norm_res_vector[interval_index, 0] = np.sqrt(resultant_ang_comp_xy[interval_index, 0]**2 + resultant_ang_comp_xy[interval_index, 1]**2)

            else:
                resultant_ang_comp_xy[interval_index, 0] = alpha * resultant_vec_x_interval + (1-alpha)*resultant_ang_comp_xy[interval_index-1, 0]
                resultant_ang_comp_xy[interval_index, 1] = alpha * resultant_vec_y_interval + (1-alpha)*resultant_ang_comp_xy[interval_index-1, 1]

                norm_res_vector[interval_index, 0] = np.sqrt(resultant_ang_comp_xy[interval_index, 0]**2 + resultant_ang_comp_xy[interval_index, 1]**2)


    #need to exclude the last rows because of the different size of those time bins
    num_bins_to_exclude = decoding_time_interval // time_overlapping

    resultant_ang_comp_xy = resultant_ang_comp_xy[:-num_bins_to_exclude]
    norm_res_vector = norm_res_vector[:-num_bins_to_exclude]
    spike_count = spike_count[:-num_bins_to_exclude]
    population_vector_angles = population_vector_angles[:-num_bins_to_exclude]

    return resultant_ang_comp_xy, norm_res_vector, spike_count, population_vector_angles


def resultant_vector_otbins_expwdistribution(spike_matrix_01,neuro_pref_dir_vector,decoding_time_interval,time_overlapping,exp_weight):

    num_columns = len(spike_matrix_01[0])
    # Initialize the resultant vector array with zeros for each interval
    # Assuming there are ceil(num_columns / decoding_time_interval) intervals

    num_intervals = ((num_columns + time_overlapping - 1) // time_overlapping)

    resultant_ang_comp_xy = np.zeros((num_intervals, 2))
    norm_res_vector = np.zeros((num_intervals, 1))
    spike_count = np.zeros((num_intervals, 1))

    # Iterate over each interval of columns, starting from column 0
    for interval_start in range(0, num_columns, time_overlapping):
        interval_end = min(interval_start + decoding_time_interval, num_columns)
        interval_index = interval_start // time_overlapping

        # Initialize the resultant vector components for the interval
        resultant_vec_x_interval = 0
        resultant_vec_y_interval = 0
                
        # Process each column within the current interval and accumulate
        for col_index in range(interval_start, interval_end):
                    
            # Step 1: Find the positions of all the 1s in the current column
            positions = [row_index for row_index, row in enumerate(spike_matrix_01) if row[col_index] == 1]
            
            spike_count[interval_index] += np.sum(spike_matrix_01[:,col_index])

            # Get the angular positions for the spikes in the current column
            estracted_ang_position = neuro_pref_dir_vector[positions]
                    
            # Sum the x and y components for the current column and add them to the interval totals
            resultant_vec_x_interval += sum(np.cos(estracted_ang_position))
            resultant_vec_y_interval += sum(np.sin(estracted_ang_position))
                
        # Calculate the norm of the resultant vector for the entire interval
        norm_res_vector_interval = np.sqrt(resultant_vec_x_interval**2 + resultant_vec_y_interval**2)
                
                
         # Check if the norm of the resultant vector is zero
        if norm_res_vector_interval == 0:
             # Handle missing data by interpolation if there are previous intervals
            if interval_index - 2 < 0:  # If there aren't enough previous intervals
                resultant_ang_comp_xy[interval_index, 0] = 0
                resultant_ang_comp_xy[interval_index, 1] = 0
            else:
                 # Calculate the slope for linear interpolation based on previous intervals
                slope_x = resultant_ang_comp_xy[interval_index - 1, 0] - resultant_ang_comp_xy[interval_index - 2, 0]
                slope_y = resultant_ang_comp_xy[interval_index - 1, 1] - resultant_ang_comp_xy[interval_index - 2, 1]
                        
                # Estimate the missing point by extrapolating the last known trend
                resultant_ang_comp_xy[interval_index, 0] = resultant_ang_comp_xy[interval_index - 1, 0] + slope_x
                resultant_ang_comp_xy[interval_index, 1] = resultant_ang_comp_xy[interval_index - 1, 1] + slope_y
        else:
             # Normalize and store the resultant vector for the interval
            #normalizing_factor = 1 / norm_res_vector
            if interval_index == 0:
                resultant_ang_comp_xy[interval_index, 0] = resultant_vec_x_interval 
                resultant_ang_comp_xy[interval_index, 1] = resultant_vec_y_interval
                norm_res_vector[interval_index, 0] = np.sqrt(resultant_ang_comp_xy[interval_index, 0]**2 + resultant_ang_comp_xy[interval_index, 1]**2)

            else:
                weight_slice = exp_weight[ -interval_index:,0]  
                x_past_weighted = np.sum(weight_slice * resultant_ang_comp_xy[0:interval_index, 0])
                y_past_weighted = np.sum(weight_slice * resultant_ang_comp_xy[0:interval_index, 1])
            
                resultant_ang_comp_xy[interval_index, 0] = resultant_vec_x_interval + x_past_weighted
                resultant_ang_comp_xy[interval_index, 1] = resultant_vec_y_interval + y_past_weighted

                norm_res_vector[interval_index, 0] = np.sqrt(resultant_ang_comp_xy[interval_index, 0]**2 + resultant_ang_comp_xy[interval_index, 1]**2)


    #need to exclude the last rows because of the different size of those time bins
    num_bins_to_exclude = decoding_time_interval // time_overlapping

    resultant_ang_comp_xy = resultant_ang_comp_xy[:-num_bins_to_exclude]
    norm_res_vector = norm_res_vector[:-num_bins_to_exclude]
    spike_count = spike_count[:-num_bins_to_exclude]

    return resultant_ang_comp_xy, norm_res_vector, spike_count

def resultant_vector_otbins_expw_velocity(spike_matrix_01,neuro_pref_dir_vector,decoding_time_interval,time_overlapping,exp_weight, velocity_vector):

    num_columns = len(spike_matrix_01[0])
    # Initialize the resultant vector array with zeros for each interval
    # Assuming there are ceil(num_columns / decoding_time_interval) intervals

    num_intervals = ((num_columns + time_overlapping - 1) // time_overlapping)
    velocity_vector = velocity_vector.reshape(-1, 1)

    resultant_ang_comp_xy = np.zeros((num_intervals, 2))
    norm_res_vector = np.zeros((num_intervals, 1))
    ang_pos_time_integrated = np.zeros((num_intervals, 1))
        
        
    # Iterate over each interval of columns, starting from column 0
    for interval_start in range(0, num_columns, time_overlapping):
        interval_end = min(interval_start + decoding_time_interval, num_columns)
                
        # Initialize the resultant vector components for the interval
        resultant_vec_x_interval = 0
        resultant_vec_y_interval = 0
                
                # Process each column within the current interval and accumulate
        for col_index in range(interval_start, interval_end):
                    # Step 1: Find the positions of all the 1s in the current column
            positions = [row_index for row_index, row in enumerate(spike_matrix_01) if row[col_index] == 1]
                    
                    # Get the angular positions for the spikes in the current column
            estracted_ang_position = neuro_pref_dir_vector[positions]
                    
                    # Sum the x and y components for the current column and add them to the interval totals
            resultant_vec_x_interval += sum(np.cos(estracted_ang_position))
            resultant_vec_y_interval += sum(np.sin(estracted_ang_position))

        # Calculate the norm of the resultant vector for the entire interval
        norm_res_vector_interval = np.sqrt(resultant_vec_x_interval**2 + resultant_vec_y_interval**2)
                
        # Determine the index in resultant_ang_comp_xy corresponding to this interval
        interval_index = interval_start // time_overlapping

        #predict position with mean velocity over the interval of time_overlapping size
        if interval_index - 1 < 0:
            ang_pos_time_integrated[interval_index,0] =  velocity_vector[interval_index,0]*time_overlapping
        else:
            ang_pos_time_integrated[interval_index,0] = ang_pos_time_integrated[interval_index-1,0] + velocity_vector[interval_index,0]*time_overlapping
        
        # Check if the norm of the resultant vector is zero
        if norm_res_vector_interval == 0:
                    # Handle missing data by interpolation if there are previous intervals
            if interval_index - 2 < 0:  # If there aren't enough previous intervals
                resultant_ang_comp_xy[interval_index, 0] = 0
                resultant_ang_comp_xy[interval_index, 1] = 0
            else:
                        # Calculate the slope for linear interpolation based on previous intervals
                slope_x = resultant_ang_comp_xy[interval_index - 1, 0] - resultant_ang_comp_xy[interval_index - 2, 0]
                slope_y = resultant_ang_comp_xy[interval_index - 1, 1] - resultant_ang_comp_xy[interval_index - 2, 1]
                        
                        # Estimate the missing point by extrapolating the last known trend
                resultant_ang_comp_xy[interval_index, 0] = resultant_ang_comp_xy[interval_index - 1, 0] + slope_x
                resultant_ang_comp_xy[interval_index, 1] = resultant_ang_comp_xy[interval_index - 1, 1] + slope_y
        else:
                    # Normalize and store the resultant vector for the interval
            normalizing_factor = 1 #/ norm_res_vector
            if interval_index == 0:
                resultant_ang_comp_xy[interval_index, 0] = normalizing_factor * resultant_vec_x_interval 
                resultant_ang_comp_xy[interval_index, 1] = normalizing_factor * resultant_vec_y_interval
            else:
                weight_slice = exp_weight[ -interval_index:,0]  
                x_past_weighted = np.sum(weight_slice * resultant_ang_comp_xy[0:interval_index, 0])
                y_past_weighted = np.sum(weight_slice * resultant_ang_comp_xy[0:interval_index, 1])
            
                resultant_ang_comp_xy[interval_index, 0] = (normalizing_factor * resultant_vec_x_interval) + x_past_weighted
                resultant_ang_comp_xy[interval_index, 1] = (normalizing_factor * resultant_vec_y_interval) + y_past_weighted
                norm_res_vector[interval_index, 0] = np.sqrt(resultant_ang_comp_xy[interval_index, 0]**2 + resultant_ang_comp_xy[interval_index, 1]**2)


    #need to exclude the last rows because of the different size of those time bins
    num_bins_to_exclude = decoding_time_interval // time_overlapping

    resultant_ang_comp_xy = resultant_ang_comp_xy[:-num_bins_to_exclude]
    norm_res_vector = norm_res_vector[:-num_bins_to_exclude]

     #= ang_pos_time_integrated[:-num_bins_to_exclude].reshape(-1, 1) % (2*np.pi) - np.pi
    ang_pos_time_integrated_x = np.cos(ang_pos_time_integrated[:-num_bins_to_exclude])
    ang_pos_time_integrated_y = np.sin(ang_pos_time_integrated[:-num_bins_to_exclude])

    ang_pos_time_integrated_xy = np.hstack((ang_pos_time_integrated_x, ang_pos_time_integrated_y))

    resultant_ang_vector_comp_xp_w_velocity = (resultant_ang_comp_xy - ang_pos_time_integrated_xy )
    
    resultant_ang_vector_0 = np.arctan2(resultant_ang_vector_comp_xp_w_velocity[:,1], resultant_ang_vector_comp_xp_w_velocity[:,0])
    resultant_ang_vector = resultant_ang_vector_0.reshape(-1, 1)


    return resultant_ang_vector, norm_res_vector

def covariance_matrix(vector_x, vector_y):

    #check whether the vectors are 1D or not
    if vector_x.ndim != 1: vector_x = vector_x.ravel()
    if vector_y.ndim != 1: vector_y = vector_y.ravel()

    #Default normalization (False) is by (N - 1), where N is the number of observations given (unbiased estimate). 
    # If bias is True, then normalization is by N. 
    cov_matrix = np.cov(vector_x, vector_y,bias=True)

    #this will give the following matrix:

    #cov_matrix = [ Var(X)      Cov(X;Y)
    #               Cov(Y,X)    Var(Y)  ]

    Var_x = cov_matrix[0,0]
    Var_y = cov_matrix[1,1]
    Cov_XY = cov_matrix[0,1] #= cov_matrix[1,0]


    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort the eigenvalues and eigenvectors
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

    # Compute the angle of the ellipse based on the largest eigenvector
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

    # Width and height of the ellipse are proportional to the square root of the eigenvalues
    width, height = 2 * np.sqrt(eigenvalues)  # 2 standard deviations


    return Var_x, Var_y, Cov_XY, width, height, angle

def ewma_circdata_alpha(alpha, data_population_vec, data_past):
    
    ewma_values = []
    cos_data_pop_vec = np.cos(data_population_vec)
    sin_data_pop_vec = np.sin(data_population_vec)
    cos_data_past = np.cos(data_past)
    sin_data_past = np.sin(data_past)

    for t in range(0,len(data_past)): 
        if t == 0:
            ewma_cos_value = cos_data_pop_vec[t]
            ewma_sin_value = sin_data_pop_vec[t]
        else:
            ewma_cos_value = alpha*cos_data_pop_vec[t] + (1-alpha)*cos_data_past[t-1]
            ewma_sin_value = alpha*sin_data_pop_vec[t] + (1-alpha)*sin_data_past[t-1]

        ewma_value = np.arctan2(ewma_sin_value, ewma_cos_value)
        ewma_values.append(ewma_value)

    return np.array(ewma_values)


