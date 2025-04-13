# Project description
This project focuses on building a decoder of spatial location or directionality given the activity of head-direction cells.

#Files explanation
The main experiment consists in the decoding of the activity of HD cells recorded over a period of time T.
The main experiment is decoder_without_AVtuning and running_experiment (both WITHOUT angular velocity tuning). 

functions_file reposrts the fucntions using in the encoding and decoding phases.
experiment_multiple_runs runs the main experiment multiple times (multiple recordings).

alpha_dependance_parameters present in the same graph the curves alpha-MSE given a variation of paramters mu, k, dt_enc.

vel_tuning_experiment reproduces one decoding experiment and applies the different possibilities of AV tuning. 
decoder_with_AVtuning_multiple_runs runs vel_tuning_experiment multiple times and presents some statistics on the results.

decoer_nonuniform_pref_directions presents and optimizes the main experiment in the case of non-uniform distribution of neurons' preferred directions.
