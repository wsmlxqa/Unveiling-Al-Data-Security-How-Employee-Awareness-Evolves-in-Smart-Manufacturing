import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import make_interp_spline
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
import seaborn as sns
import os

# =============================================================================
# 0. Initial Setup and Verification
# =============================================================================
print("=" * 70)
print("Starting figure generation from pre-existing raw data files...")
print("=" * 70)

# Define the directory where the raw data is stored
DATA_DIR = 'reconstructed_raw_data'
# Define the expected filenames
files_to_load = {
    'No Intervention': os.path.join(DATA_DIR, 'raw_data_no_intervention.csv'),
    'Mild Propaganda': os.path.join(DATA_DIR, 'raw_data_mild_propaganda.csv'),
    'Mandatory Training': os.path.join(DATA_DIR, 'raw_data_mandatory_training.csv')
}

# Verify that all required files exist before proceeding
for name, path in files_to_load.items():
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Error: The required data file '{path}' was not found.\n"
            f"Please ensure this script is run in a directory that contains the '{DATA_DIR}' folder with the necessary CSV files."
        )
print("All required data files found. Proceeding with visualization.")


# =============================================================================
# 1. Setup for Top-Tier Journal Publication Quality
# =============================================================================
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 16
rcParams['axes.linewidth'] = 1.5
rcParams['xtick.major.width'] = 1.5
rcParams['ytick.major.width'] = 1.5
rcParams['xtick.major.size'] = 8
rcParams['ytick.major.size'] = 8
rcParams['xtick.minor.width'] = 1.0
rcParams['ytick.minor.width'] = 1.0
rcParams['xtick.minor.size'] = 4
rcParams['ytick.minor.size'] = 4
rcParams['axes.unicode_minus'] = False
rcParams['mathtext.fontset'] = 'stix'

# =============================================================================
# 2. Load and Process Raw Data from CSV Files
# =============================================================================
print("\nLoading and processing raw data...")

raw_data_dict = {}
for name, path in files_to_load.items():
    # Load data using pandas, set the first column as the index
    df = pd.read_csv(path, index_col=0)
    # Convert the DataFrame to a NumPy array for numerical processing
    raw_data_dict[name] = df.to_numpy()
    print(f"  - Loaded '{name}' data with shape: {raw_data_dict[name].shape}")

# Get parameters from the loaded data shape
n_simulations, iterations = raw_data_dict['No Intervention'].shape

# --- Data Processing and Smoothing Function ---
def compute_stats_and_smooth(data, smooth_factor=300):
    """
    Computes statistics (mean, CI, std) from raw data and smooths them for plotting.
    """
    # Calculate statistics along the time axis (axis=0)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    sem = stats.sem(data, axis=0)
    n_sim = data.shape[0] # Get number of simulations from data
    ci = sem * stats.t.ppf((1 + 0.95) / 2., n_sim - 1)
    
    # Create x-axes for original and smoothed data
    x_orig = np.arange(len(mean))
    x_smooth = np.linspace(x_orig.min(), x_orig.max(), smooth_factor)
    
    # Apply spline interpolation for smoothing
    mean_smooth = make_interp_spline(x_orig, mean, k=3)(x_smooth)
    ci_smooth = make_interp_spline(x_orig, ci, k=3)(x_smooth)
    std_smooth = make_interp_spline(x_orig, std, k=3)(x_smooth)
    
    return x_smooth, mean_smooth, ci_smooth, std_smooth

# Process all loaded scenarios
smooth_results = {}
for name, data in raw_data_dict.items():
    x_s, mean_s, ci_s, std_s = compute_stats_and_smooth(data)
    smooth_results[name] = {'x': x_s, 'mean': mean_s, 'ci': ci_s, 'std': std_s}
print("Statistical processing and data smoothing complete.")


# =============================================================================
# 3. Visualization: Comprehensive 5-Panel Figure
# =============================================================================
print("\nGenerating the 5-panel figure...")

# --- Define a professional color palette and styles ---
colors = {'No Intervention': '#4477AA', 'Mild Propaganda': '#EE6677', 'Mandatory Training': '#228833'}
linestyles = {'No Intervention': '--', 'Mild Propaganda': '-.', 'Mandatory Training': '-'}

# --- Create the figure with GridSpec for complex layout ---
fig = plt.figure(figsize=(18, 12), dpi=300, facecolor='white')
gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

axA = fig.add_subplot(gs[0, :2])
axB = fig.add_subplot(gs[0, 2])
axC = fig.add_subplot(gs[1, 0])
axD = fig.add_subplot(gs[1, 1])
axE = fig.add_subplot(gs[1, 2])

fig.suptitle('Comprehensive Analysis of Interventions on AI Data Security Awareness ($S_o$)', fontsize=22, fontweight='bold', y=0.98)

# --- Panel A: Main Evolutionary Dynamics ---
for name, res in smooth_results.items():
    axA.plot(res['x'], res['mean'], label=name, color=colors[name], linestyle=linestyles[name], lw=3, zorder=3)
    axA.fill_between(res['x'], res['mean'] - res['ci'], res['mean'] + res['ci'], color=colors[name], alpha=0.15, ec='none', zorder=2)
axA.set_title('Main Evolutionary Dynamics', fontsize=18, style='italic')
axA.set_xlabel('Iteration (Time)', fontsize=16, fontweight='bold', labelpad=10)
axA.set_ylabel('Awareness Level ($S_o$)', fontsize=16, fontweight='bold', labelpad=10)
axA.set_xlim(0, iterations)
axA.set_ylim(35000, 65000)
legend = axA.legend(loc='upper left', fontsize=14, frameon=True)
legend.get_frame().set_linewidth(1.2)
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_alpha(0.8)

# --- Panel B: Rate of Change Analysis ---
for name, res in smooth_results.items():
    rate = np.gradient(res['mean'], res['x'][1] - res['x'][0])
    axB.plot(res['x'], rate, color=colors[name], linestyle=linestyles[name], lw=2.5, label=name)
axB.axhline(0, color='black', lw=1.5, ls=':', alpha=0.7)
axB.set_title('Rate of Change ($dS_o/dt$)', fontsize=18, style='italic')
axB.set_xlabel('Iteration (Time)', fontsize=16, fontweight='bold', labelpad=10)
axB.set_ylabel('Growth Rate', fontsize=16, fontweight='bold', labelpad=10)
axB.set_xlim(0, iterations)

# --- Panel C: Intervention Impact Analysis ---
baseline_mean = smooth_results['No Intervention']['mean']
impact_mp = smooth_results['Mild Propaganda']['mean'] - baseline_mean
impact_mt = smooth_results['Mandatory Training']['mean'] - baseline_mean
x_smooth = smooth_results['No Intervention']['x']

axC.plot(x_smooth, impact_mp, color=colors['Mild Propaganda'], lw=2.5, label='Mild Propaganda vs. Baseline')
axC.fill_between(x_smooth, 0, impact_mp, color=colors['Mild Propaganda'], alpha=0.2)
axC.plot(x_smooth, impact_mt, color=colors['Mandatory Training'], lw=2.5, label='Mandatory Training vs. Baseline')
axC.fill_between(x_smooth, 0, impact_mt, color=colors['Mandatory Training'], alpha=0.2)
axC.axhline(0, color='black', lw=1.5, ls=':', alpha=0.7)
axC.set_title('Intervention Impact ($\Delta S_o$ vs Baseline)', fontsize=18, style='italic')
axC.set_xlabel('Iteration (Time)', fontsize=16, fontweight='bold', labelpad=10)
axC.set_ylabel('Added Awareness', fontsize=16, fontweight='bold', labelpad=10)
axC.legend(fontsize=12)
axC.set_xlim(0, iterations)

# --- Panel D: Volatility Analysis ---
for name, res in smooth_results.items():
    axD.plot(res['x'], res['std'], color=colors[name], linestyle=linestyles[name], lw=2.5, label=name)
    axD.fill_between(res['x'], 0, res['std'], color=colors[name], alpha=0.15)
axD.set_title('Outcome Volatility ($\sigma$ of $S_o$)', fontsize=18, style='italic')
axD.set_xlabel('Iteration (Time)', fontsize=16, fontweight='bold', labelpad=10)
axD.set_ylabel('Standard Deviation', fontsize=16, fontweight='bold', labelpad=10)
axD.legend(fontsize=12)
axD.set_xlim(0, iterations)

# --- Panel E: Final State Distribution ---
# Create a DataFrame for the final state data (last column of raw data)
final_states_df = pd.DataFrame({
    name: data[:, -1] for name, data in raw_data_dict.items()
})
sns.violinplot(data=final_states_df, ax=axE, palette=colors, cut=0)
axE.set_title('Final State Distribution (t=100)', fontsize=18, style='italic')
axE.set_ylabel('Final Awareness Level ($S_o$)', fontsize=16, fontweight='bold', labelpad=10)
axE.set_xlabel('Intervention Scenario', fontsize=16, fontweight='bold', labelpad=10)
axE.tick_params(axis='x', rotation=15)

# --- Add panel labels and finalize layout ---
panel_labels = ['A', 'B', 'C', 'D', 'E']
axes = [axA, axB, axC, axD, axE]
for ax, label in zip(axes, panel_labels):
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='left')
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray', alpha=0.6)
    ax.minorticks_on()
    ax.tick_params(direction='in', which='both', top=True, right=True)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# =============================================================================
# 4. Save the Final Figure
# =============================================================================
output_filename_base = 'Figure_5_Reproduced_from_Data'
plt.savefig(f'{output_filename_base}.png', dpi=600, bbox_inches='tight')
plt.savefig(f'{output_filename_base}.pdf', bbox_inches='tight')
plt.savefig(f'{output_filename_base}.tiff', dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})

print("\n" + "=" * 70)
print("✅ Figure successfully generated and saved from raw data files!")
print("   This script is now ready for submission.")
print("\n💾 High-resolution output files:")
print(f"   • {output_filename_base}.png")
print(f"   • {output_filename_base}.pdf")
print(f"   • {output_filename_base}.tiff")
print("=" * 70)

plt.show()

