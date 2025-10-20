import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set global font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False  # Prevent minus signs from displaying as boxes

# Load the data from CSV
csv_filename = 'sensitivity_boxplot_data.csv'

if not os.path.exists(csv_filename):
    raise FileNotFoundError(f"CSV file '{csv_filename}' not found in the current directory.")

sensitivity_data = pd.read_csv(csv_filename)

# Extract unique alpha and beta values
alpha_values = sensitivity_data['Alpha'].unique()
beta_values = sensitivity_data['Beta'].unique()

# Sort the values to ensure consistent ordering on x-axis
alpha_values = np.sort(alpha_values)
beta_values = np.sort(beta_values)

# Prepare data for alpha boxplots
so_mild_alpha_box = []
so_mandatory_alpha_box = []
for alpha in alpha_values:
    mild_values = sensitivity_data[sensitivity_data['Alpha'] == alpha]['SO_Mild_Alpha'].values
    mand_values = sensitivity_data[sensitivity_data['Alpha'] == alpha]['SO_Mandatory_Alpha'].values
    so_mild_alpha_box.append(mild_values)
    so_mandatory_alpha_box.append(mand_values)

# Prepare data for beta boxplots
so_mild_beta_box = []
so_mandatory_beta_box = []
for beta in beta_values:
    mild_values = sensitivity_data[sensitivity_data['Beta'] == beta]['SO_Mild_Beta'].values
    mand_values = sensitivity_data[sensitivity_data['Beta'] == beta]['SO_Mandatory_Beta'].values
    so_mild_beta_box.append(mild_values)
    so_mandatory_beta_box.append(mand_values)

# Plot Alpha boxplots
fig, ax = plt.subplots(figsize=(8, 6))
positions_mild = np.arange(len(alpha_values)) - 0.2
positions_mand = np.arange(len(alpha_values)) + 0.2

ax.boxplot(so_mild_alpha_box,
           positions=positions_mild, widths=0.4, patch_artist=True,
           boxprops=dict(facecolor='orange', alpha=0.5), medianprops=dict(color='black'),
           label='Mild Publicity')

ax.boxplot(so_mandatory_alpha_box,
           positions=positions_mand, widths=0.4, patch_artist=True,
           boxprops=dict(facecolor='blue', alpha=0.5), medianprops=dict(color='black'),
           label='Mandatory Training')

ax.set_xticks(np.arange(len(alpha_values)))
ax.set_xticklabels([f'{a:.2f}' for a in alpha_values], rotation=45)
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$S_o$')
ax.set_title(r'Sensitivity of $S_o$ to $\alpha$ (50 Runs per Value)')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()  # Better layout management
plt.savefig('sensitivity_boxplots_alpha.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot Beta boxplots
fig, ax = plt.subplots(figsize=(8, 6))
positions_mild = np.arange(len(beta_values)) - 0.2
positions_mand = np.arange(len(beta_values)) + 0.2

ax.boxplot(so_mild_beta_box,
           positions=positions_mild, widths=0.4, patch_artist=True,
           boxprops=dict(facecolor='orange', alpha=0.5), medianprops=dict(color='black'),
           label='Mild Publicity')

ax.boxplot(so_mandatory_beta_box,
           positions=positions_mand, widths=0.4, patch_artist=True,
           boxprops=dict(facecolor='blue', alpha=0.5), medianprops=dict(color='black'),
           label='Mandatory Training')

ax.set_xticks(np.arange(len(beta_values)))
ax.set_xticklabels([f'{b:.2f}' for b in beta_values], rotation=45)
ax.set_xlabel(r'$\beta$')
ax.set_ylabel(r'$S_o$')
ax.set_title(r'Sensitivity of $S_o$ to $\beta$ (50 Runs per Value)')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('sensitivity_boxplots_beta.png', dpi=300, bbox_inches='tight')
plt.close()

print("Boxplots saved successfully as 'sensitivity_boxplots_alpha.png' and 'sensitivity_boxplots_beta.png'.")
