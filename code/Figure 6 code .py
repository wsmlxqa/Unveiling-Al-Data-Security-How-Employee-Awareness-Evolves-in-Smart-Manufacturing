import pandas as pd
import matplotlib.pyplot as plt

# Set global font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False  # Prevent minus signs from displaying as boxes

# Load the data from CSV
sensitivity_data = pd.read_csv('sensitivity_alpha_beta_line_data.csv')

# Extract data for alpha sensitivity
alpha_values = sensitivity_data['Alpha'].values
so_mild_alpha = sensitivity_data['SO_Mild_Alpha'].values
so_mandatory_alpha = sensitivity_data['SO_Mandatory_Alpha'].values

# Extract data for beta sensitivity
beta_values = sensitivity_data['Beta'].values
so_mild_beta = sensitivity_data['SO_Mild_Beta'].values
so_mandatory_beta = sensitivity_data['SO_Mandatory_Beta'].values

# Create the figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

# --- Left subplot: Sensitivity to alpha ---
ax1.plot(alpha_values, so_mandatory_alpha, 'b-', label='Mandatory Training')
ax1.plot(alpha_values, so_mild_alpha, 'orange', linestyle='--', label='Mild Propaganda')
ax1.set_title('(a) Sensitivity to $\\alpha$')
ax1.set_xlabel('$\\alpha$')
ax1.set_ylabel('$S_o$')
ax1.set_ylim(45000, 70000)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

# --- Right subplot: Sensitivity to beta ---
ax2.plot(beta_values, so_mandatory_beta, 'b-', label='Mandatory Training')
ax2.plot(beta_values, so_mild_beta, 'orange', linestyle='--', label='Mild Propaganda')
ax2.set_title('(b) Sensitivity to $\\beta$')
ax2.set_xlabel('$\\beta$')
ax2.set_ylabel('$S_o$')
ax2.set_ylim(45000, 70000)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()

# --- Main title ---
fig.suptitle('Impact of Dynamic Threshold Parameters $\\alpha$ and $\\beta$ on $S_o$')

# --- Save figure ---
plt.savefig('sensitivity_alpha_beta.png', dpi=300, bbox_inches='tight')
plt.close()
