# --- 1. Load Your Data ---

try:
    df = pd.read_csv('/content/drive/My Drive/Perceived Quality of Life Across Southeast Asian Cities/Datasets/full_bayesian_regression_results_with_marital_status.csv')
except FileNotFoundError:
    print("Error: 'full_bayesian_regression_results_with_marital_status.csv' not found.")
    print("Please upload the file to your Colab session or mount your Google Drive and update the path.")
    # Create a dummy dataframe to prevent the rest of the code from crashing
    df = pd.DataFrame(columns=['term', 'estimate', 'conf.low', 'conf.high', 'City'])
    
# --- 2. Define Policy Levers, Map to Names, and Assign Categories ---
policy_mapping = {
    'q2_4': 'Ample employment opportunities',
    'q4_8': 'I have saved money to invest',
    'q1_2': 'Good place for children',
    'q4_6': 'Neighbours look out for each other',
    'q4_13': 'I am not disadvantaged',
    'q3_1': 'Good quality healthcare facilities',
    'q3_2': 'Effective waste management system',
    'q4_7': 'I feel safe in this neighborhood',
    'q1_4': 'Effective sanitary system'
}

policy_categories = {
    'q2_4': 'Personal-Cohesion',
    'q4_8': 'Personal-Cohesion',
    'q4_13': 'Personal-Cohesion',
    'q4_6': 'Personal-Cohesion',
    'q4_7': 'Personal-Cohesion',
    'q1_2': 'Space-Environment',
    'q3_1': 'Space-Environment',
    'q3_2': 'Space-Environment',
    'q1_4': 'Space-Environment'
}

# Filter for policy levers
df_policies = df[df['term'].isin(policy_mapping.keys())].copy()

# Add new columns for the descriptive names and categories
df_policies['Policy Initiative'] = df_policies['term'].map(policy_mapping)
df_policies['Category'] = df_policies['term'].map(policy_categories)

# --- 3. Final Plotting Function with Right-Aligned LaTeX Labels ---
def create_final_coefficient_plot(city_name, dataframe):
    """Generates and saves a coefficient plot with right-aligned group labels."""
    city_df = dataframe[dataframe['City'] == city_name].copy()

    if city_df.empty:
        print(f"No data found for {city_name}. Skipping plot generation.")
        return

    # Sort by Category first, then by the estimate, for visual grouping
    city_df = city_df.sort_values(by=['Category', 'estimate'], ascending=[False, True])
    city_df = city_df.reset_index(drop=True)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))

    # The point estimates (Odds Ratios) and error bars
    errors = [city_df['estimate'] - city_df['conf.low'], city_df['conf.high'] - city_df['estimate']]
    ax.errorbar(x=city_df['estimate'], y=city_df.index, xerr=errors,
                fmt='o', color='gray', capsize=5, zorder=1, linewidth=1.5,
                markerfacecolor='crimson', markeredgecolor='crimson', markersize=8)

    # --- Add Group Separator Line ---
    space_env_count = city_df['Category'].value_counts()['Space-Environment']
    separator_pos = space_env_count - 0.5
    ax.axhline(separator_pos, color='blue', linestyle='--', linewidth=1.2, alpha=0.7)

    # Add a vertical line at 1.0 for reference
    ax.axvline(x=1, linestyle='--', color='black', linewidth=1.5, zorder=2)

    # --- Add Right-Aligned LaTeX Labels ---
    # Determine a good x-position for the labels on the right
    x_pos = ax.get_xlim()[1] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02

    personal_coh_count = city_df['Category'].value_counts()['Personal-Cohesion']

    # Y-position for the label in the middle of each group
    y_pos_se = (space_env_count - 1) / 2
    y_pos_pc = space_env_count + (personal_coh_count - 1) / 2

    ax.text(x_pos, y_pos_se, r'$K_{SE}$', ha='left', va='center', fontsize=14, color='blue')
    ax.text(x_pos, y_pos_pc, r'$K_{PC}$', ha='left', va='center', fontsize=14, color='blue')

    # --- Formatting and Final Touches ---
    ax.set_yticks(city_df.index)
    ax.set_yticklabels(city_df['Policy Initiative'])
    ax.set_title(f' {city_name}', fontsize=16)
    ax.set_xlabel('Odds Ratio', fontsize=12)
    ax.set_ylabel('') # Y-axis label is redundant
    fig.tight_layout(pad=1.5)

    # Adjust plot limits to make space for the new labels
    plt.subplots_adjust(right=0.88)

    # --- Save the Figure ---
    filename = f'{city_name.lower().replace(" ", "_")}_final_coefficients.png'
    plt.savefig(filename, dpi=300)
    print(f"Final plot for {city_name} saved as '{filename}'")
    plt.show()

# --- 4. Generate the Final Plots ---
print("Generating final plots...")
create_final_coefficient_plot('Jakarta', df_policies)
create_final_coefficient_plot('Phnom Penh', df_policies)
print("Done.")