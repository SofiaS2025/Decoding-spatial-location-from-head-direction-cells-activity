import pandas as pd
import matplotlib.pyplot as plt
import os

# Folder containing the data files
data_dir = "alpha_plots_outputs"
os.makedirs(data_dir, exist_ok=True)

# Load data
alpha_df = pd.read_csv(os.path.join(data_dir, "alpha_values.csv"))
mse_df = pd.read_csv(os.path.join(data_dir, "mse_values.csv"))
alpha_fit_df = pd.read_csv(os.path.join(data_dir, "alpha_fit.csv"))
mse_fit_df = pd.read_csv(os.path.join(data_dir, "mse_fit.csv"))
best_alpha_df = pd.read_csv(os.path.join(data_dir, "best_alpha.csv"))
min_mse_df = pd.read_csv(os.path.join(data_dir, "min_mse.csv"))

k_values = [5,3,8]

# Group columns in intervals of 3
columns = alpha_df.columns
group_size = 3
column_groups = [columns[i:i + group_size] for i in range(0, len(columns), group_size)]

# Manually specify which group you want to plot
# For example, to plot the first group (1st, 2nd, and 3rd columns), use group_idx = 0
group_idx = 2  # Change this value to plot other groups (0 for the first group, 1 for the second, etc.)

# Get the selected group
group = column_groups[group_idx]

count_col = 0
# Create the plot
plt.figure(figsize=(6.4, 4.8))

# Plot all columns in the selected group
for col in group:
    # Plot data points
    alpha_vals = alpha_df[col].dropna().values
    mse_vals = mse_df[col].dropna().values

    if len(alpha_vals) != len(mse_vals):
        continue

    plt.scatter(alpha_vals, mse_vals, s=15, zorder=5)

    # Plot fit line if available
    if col in alpha_fit_df.columns and col in mse_fit_df.columns:
        alpha_fit_vals = alpha_fit_df[col].dropna().values
        mse_fit_vals = mse_fit_df[col].dropna().values
        if len(alpha_fit_vals) == len(mse_fit_vals):
            plt.plot(alpha_fit_vals, mse_fit_vals, label=f'k={k_values[count_col]}', zorder=4, linewidth=4)
            count_col += 1

# Now we add arrows pointing between best_alpha, min_mse of k_1 -> k_2 and k_1 -> k_3
best_alpha_k1 = best_alpha_df[group[0]].dropna().values[0]
min_mse_k1 = min_mse_df[group[0]].dropna().values[0]

best_alpha_k2 = best_alpha_df[group[1]].dropna().values[0]
min_mse_k2 = min_mse_df[group[1]].dropna().values[0]

best_alpha_k3 = best_alpha_df[group[2]].dropna().values[0]
min_mse_k3 = min_mse_df[group[2]].dropna().values[0]

# Add arrows from k_1 to k_2 and k_1 to k_3
plt.annotate(
    '', 
    xy=(best_alpha_k2, min_mse_k2), 
    xytext=(best_alpha_k1, min_mse_k1), 
    arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=4),  # Finer and smaller arrows
    zorder=6
)

plt.annotate(
    '', 
    xy=(best_alpha_k3, min_mse_k3), 
    xytext=(best_alpha_k1, min_mse_k1), 
    arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=4),  # Finer and smaller arrows
    zorder=6
)
# Mark the best_alpha, min_mse points with a red dot or small "+"
plt.scatter(best_alpha_k1, min_mse_k1, color='red', zorder=7,  s=150, marker='+')  # Red dot for k_1
plt.scatter(best_alpha_k2, min_mse_k2, color='red', zorder=7,  s=150, marker='+')  # Red dot for k_2
plt.scatter(best_alpha_k3, min_mse_k3, color='red', zorder=7,  s=150, marker='+')  # Red dot for k_3

# Final plot formatting
plt.xlabel(r'$\alpha$', fontsize=18)
plt.ylabel('MSE', fontsize=18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlim(0, 0.4)
plt.ylim(0, 0.3)
plt.legend(fontsize=15)
plt.tight_layout()
plt.show()
