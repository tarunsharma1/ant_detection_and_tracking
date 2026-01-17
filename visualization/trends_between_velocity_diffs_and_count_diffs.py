import sys
sys.path.append('../')
import pickle
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


def bootstrap(data, n):
    """Randomly resample n times and take the mean each time"""
    bootstrapped_data = np.zeros(n)
    for i in range(0, n):
        sample = np.random.choice(data, size=len(data))
        bootstrapped_data[i] = np.mean(np.array(sample))
    return bootstrapped_data

def confidence_interval(data):
    """Get the 95% confidence interval by getting the 2.5th and 97.5th percentile of the data"""
    conf_interval = np.percentile(data, [2.5, 97.5])
    return conf_interval[0], conf_interval[1]

def load_velocity_data(bout_name):
    """
    Load velocity data from pickle file.
    
    Parameters
    ----------
    bout_name : str
        Name of the bout (e.g., "beer_2024-08-01_2024-08-10")
    
    Returns
    -------
    dict
        Dictionary containing velocities_away_per_hour_across_days, velocities_toward_per_hour_across_days, etc.
    """
    save_dir = '/home/tarun/Desktop/plots_for_committee_meeting/velocity_analysis'
    pickle_path = os.path.join(save_dir, f"velocities_{bout_name}.pkl")
    
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Velocity pickle file not found: {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    return data

def load_count_data(bout_name):
    """
    Load count data from pickle file.
    
    Parameters
    ----------
    bout_name : str
        Name of the bout (e.g., "beer_2024-08-01_2024-08-10")
    
    Returns
    -------
    dict
        Dictionary containing counts_away_per_hour_across_days, counts_toward_per_hour_across_days, etc.
    """
    save_dir = '/home/tarun/Desktop/plots_for_committee_meeting/velocity_analysis'
    pickle_path = os.path.join(save_dir, f"counts_{bout_name}.pkl")
    
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Count pickle file not found: {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    return data

def compute_hourly_differences_per_day(away_per_hour_across_days, toward_per_hour_across_days):
    """
    Compute (away - toward) differences for each hour, for each day.
    Returns a list where each element is a list of 24 hourly differences for that day.
    
    Parameters
    ----------
    away_per_hour_across_days : dict
        {hour: [values_per_day]}
    toward_per_hour_across_days : dict
        {hour: [values_per_day]}
    
    Returns
    -------
    list of lists
        [[day0_hour0_diff, day0_hour1_diff, ..., day0_hour23_diff],
         [day1_hour0_diff, day1_hour1_diff, ..., day1_hour23_diff],
         ...]
    """
    # Determine number of days from the data
    num_days = None
    for hour in range(24):
        if hour in away_per_hour_across_days and len(away_per_hour_across_days[hour]) > 0:
            num_days = len(away_per_hour_across_days[hour])
            break
    
    if num_days is None:
        return []
    
    hourly_differences_per_day = []
    
    for day_idx in range(num_days):
        day_hourly_differences = []
        
        # Get differences for each hour for this day
        for hour in range(24):
            if hour in away_per_hour_across_days and hour in toward_per_hour_across_days:
                away_values = away_per_hour_across_days[hour]
                toward_values = toward_per_hour_across_days[hour]
                
                if day_idx < len(away_values) and day_idx < len(toward_values):
                    diff = away_values[day_idx] - toward_values[day_idx]
                    day_hourly_differences.append(diff)
                else:
                    day_hourly_differences.append(0.0)  # Missing data
            else:
                day_hourly_differences.append(0.0)  # Missing hour
        
        hourly_differences_per_day.append(day_hourly_differences)
    
    return hourly_differences_per_day

def analyze_velocity_count_correlation(bout_name):
    """
    Analyze Spearman correlation between velocity differences and count differences on a day-to-day basis.
    Bootstraps across days to test if correlation is significantly greater than 0.
    
    Parameters
    ----------
    bout_name : str
        Name of the bout (e.g., "beer_2024-08-01_2024-08-10")
    
    Returns
    -------
    dict
        Results containing correlation, p-value, bootstrapped statistics, etc.
    """
    # Load data
    velocity_data = load_velocity_data(bout_name)
    count_data = load_count_data(bout_name)
    
    velocities_away = velocity_data['velocities_away_per_hour_across_days']
    velocities_toward = velocity_data['velocities_toward_per_hour_across_days']
    counts_away = count_data['counts_away_per_hour_across_days']
    counts_toward = count_data['counts_toward_per_hour_across_days']
    
    # Compute hourly differences for each day
    velocity_hourly_diffs_per_day = compute_hourly_differences_per_day(velocities_away, velocities_toward)
    count_hourly_diffs_per_day = compute_hourly_differences_per_day(counts_away, counts_toward)
    
    if len(velocity_hourly_diffs_per_day) != len(count_hourly_diffs_per_day):
        print(f"Warning: Mismatch in number of days. Velocity: {len(velocity_hourly_diffs_per_day)}, Count: {len(count_hourly_diffs_per_day)}")
        min_len = min(len(velocity_hourly_diffs_per_day), len(count_hourly_diffs_per_day))
        velocity_hourly_diffs_per_day = velocity_hourly_diffs_per_day[:min_len]
        count_hourly_diffs_per_day = count_hourly_diffs_per_day[:min_len]
    
    if len(velocity_hourly_diffs_per_day) < 2:
        print("Not enough days for correlation analysis")
        return None
    
    # Compute Spearman correlation for each day
    daily_correlations = []
    daily_p_values = []
    
    for day_idx in range(len(velocity_hourly_diffs_per_day)):
        velocity_diffs = np.array(velocity_hourly_diffs_per_day[day_idx])
        count_diffs = np.array(count_hourly_diffs_per_day[day_idx])
        
        # Check if we have variation in both vectors
        if len(set(velocity_diffs)) > 1 and len(set(count_diffs)) > 1:
            corr, p_val = spearmanr(velocity_diffs, count_diffs)
            if not np.isnan(corr):
                daily_correlations.append(corr)
                daily_p_values.append(p_val)
            else:
                daily_correlations.append(np.nan)
                daily_p_values.append(np.nan)
        else:
            daily_correlations.append(np.nan)
            daily_p_values.append(np.nan)
    
    # Remove NaN values
    valid_corrs = [c for c in daily_correlations if not np.isnan(c)]
    
    if len(valid_corrs) == 0:
        print("No valid daily correlations computed")
        return None
    
    # Compute mean correlation across days
    mean_corr = np.mean(valid_corrs)
    
    print("\n" + "="*80)
    print(f"CORRELATION ANALYSIS: {bout_name}")
    print("="*80)
    print(f"Number of days: {len(velocity_hourly_diffs_per_day)}")
    print(f"Valid daily correlations: {len(valid_corrs)}")
    print(f"Daily correlations: {[f'{c:.4f}' for c in valid_corrs]}")
    print(f"Mean correlation across days: {mean_corr:.4f}")
    print("="*80)
    
    # Bootstrap correlation values across days
    n_bootstrap = 10000
    bootstrapped_corrs = []
    
    for _ in range(n_bootstrap):
        # Resample daily correlations with replacement
        bootstrapped_sample = np.random.choice(valid_corrs, size=len(valid_corrs), replace=True)
        bootstrapped_mean = np.mean(bootstrapped_sample)
        bootstrapped_corrs.append(bootstrapped_mean)
    
    bootstrapped_corrs = np.array(bootstrapped_corrs)
    
    # Compute bootstrapped statistics
    bootstrapped_mean = np.mean(bootstrapped_corrs)
    conf_min, conf_max = confidence_interval(bootstrapped_corrs)
    
    # Test if correlation is significantly different from 0
    p_value_gt_zero = np.mean(bootstrapped_corrs > 0)
    p_value_lt_zero = np.mean(bootstrapped_corrs < 0)
    is_significant = (conf_min > 0) or (conf_max < 0)
    
    print(f"\nBootstrapped mean correlation: {bootstrapped_mean:.4f}")
    print(f"95% CI: [{conf_min:.4f}, {conf_max:.4f}]")
    print(f"P(correlation > 0): {p_value_gt_zero:.4f}")
    print(f"P(correlation < 0): {p_value_lt_zero:.4f}")
    print(f"Significantly different from 0 (CI excludes 0): {is_significant}")
    if is_significant:
        direction = "positive" if bootstrapped_mean > 0 else "negative"
        print(f"Direction: {direction}")
    print("="*80 + "\n")
    
    # ========== SIGN ALIGNMENT ANALYSIS ==========
    # For each day, for each hour, check if signs match
    daily_sign_alignment_proportions = []
    
    for day_idx in range(len(velocity_hourly_diffs_per_day)):
        velocity_diffs = np.array(velocity_hourly_diffs_per_day[day_idx])
        count_diffs = np.array(count_hourly_diffs_per_day[day_idx])
        
        # Check sign alignment for each hour (1 if same sign, 0 if different)
        sign_matches = []
        for hour_idx in range(len(velocity_diffs)):
            vel_sign = np.sign(velocity_diffs[hour_idx])
            count_sign = np.sign(count_diffs[hour_idx])
            
            # Same sign if both positive, both negative, or both zero
            # Note: np.sign(0) = 0, so we need to handle zeros carefully
            if vel_sign == 0 and count_sign == 0:
                # Both zero - consider as matching
                sign_matches.append(1)
            elif vel_sign == count_sign and vel_sign != 0:
                # Same non-zero sign
                sign_matches.append(1)
            else:
                # Different signs or one is zero and other is not
                sign_matches.append(0)
        
        # Compute proportion of hours with matching signs for this day
        alignment_proportion = np.mean(sign_matches)
        daily_sign_alignment_proportions.append(alignment_proportion)
    
    daily_sign_alignment_proportions = np.array(daily_sign_alignment_proportions)
    mean_alignment = np.mean(daily_sign_alignment_proportions)
    
    # Bootstrap mean alignment per day across days
    bootstrapped_alignments = bootstrap(daily_sign_alignment_proportions, 10000)
    bootstrapped_mean_alignment = np.mean(bootstrapped_alignments)
    alignment_conf_min, alignment_conf_max = confidence_interval(bootstrapped_alignments)
    
    # Test if alignment is significantly greater than 0.5 (random chance)
    p_value_gt_half = np.mean(bootstrapped_alignments > 0.5)
    is_alignment_significant = alignment_conf_min > 0.5
    
    print("\n" + "="*80)
    print(f"SIGN ALIGNMENT ANALYSIS: {bout_name}")
    print("="*80)
    print(f"Number of days: {len(daily_sign_alignment_proportions)}")
    print(f"Daily sign alignment proportions: {[f'{p:.4f}' for p in daily_sign_alignment_proportions]}")
    print(f"Mean alignment proportion: {mean_alignment:.4f}")
    print(f"Bootstrapped mean alignment: {bootstrapped_mean_alignment:.4f}")
    print(f"95% CI: [{alignment_conf_min:.4f}, {alignment_conf_max:.4f}]")
    print(f"P(alignment > 0.5): {p_value_gt_half:.4f}")
    print(f"Significantly greater than 0.5 (random chance): {is_alignment_significant}")
    if is_alignment_significant:
        print(f"Interpretation: Signs align more often than random chance (>50%)")
    print("="*80 + "\n")
    
    # Create scatter plot of daily correlations
    plt.figure(figsize=(10, 8))
    days = range(1, len(daily_correlations) + 1)
    valid_days = [d for d, c in zip(days, daily_correlations) if not np.isnan(c)]
    valid_corrs_plot = [c for c in daily_correlations if not np.isnan(c)]
    
    plt.scatter(valid_days, valid_corrs_plot, s=100, alpha=0.6, edgecolors='black', linewidth=1.5, 
                label='Daily correlations')
    
    # Add horizontal line at mean
    plt.axhline(y=mean_corr, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                label=f'Mean: {mean_corr:.4f}')
    
    # Add horizontal line at 0
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    
    plt.xlabel('Day')
    plt.ylabel('Spearman Correlation\n(Velocity diff vs Count diff, 24 hours)')
    plt.title(f'Daily Correlations: Velocity vs Count Differences\n{bout_name}\nMean r={mean_corr:.4f}, Bootstrapped mean={bootstrapped_mean:.4f}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add text box with statistics
    stats_text = f'Mean correlation: {mean_corr:.4f}\nBootstrapped mean: {bootstrapped_mean:.4f}\n95% CI: [{conf_min:.4f}, {conf_max:.4f}]\nP(r > 0): {p_value_gt_zero:.4f}\nSignificant: {is_significant}'
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    save_dir = '/home/tarun/Desktop/plots_for_committee_meeting/velocity_analysis'
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, f"velocity_count_correlation_{bout_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Correlation plot saved to: {plot_path}")
    plt.show()
    
    # Create scatter plot of daily sign alignment proportions
    plt.figure(figsize=(10, 8))
    days = range(1, len(daily_sign_alignment_proportions) + 1)
    
    plt.scatter(days, daily_sign_alignment_proportions, s=100, alpha=0.6, edgecolors='black', 
                linewidth=1.5, label='Daily sign alignment', color='green')
    
    # Add horizontal line at mean
    plt.axhline(y=mean_alignment, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                label=f'Mean: {mean_alignment:.4f}')
    
    # Add horizontal line at 0.5 (random chance)
    plt.axhline(y=0.5, color='black', linestyle='-', linewidth=1, alpha=0.5, 
                label='Random chance (0.5)')
    
    plt.xlabel('Day')
    plt.ylabel('Proportion of Hours with Matching Signs\n(Velocity diff and Count diff)')
    plt.title(f'Daily Sign Alignment: Velocity vs Count Differences\n{bout_name}\nMean={mean_alignment:.4f}, Bootstrapped mean={bootstrapped_mean_alignment:.4f}')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add text box with statistics
    stats_text = f'Mean alignment: {mean_alignment:.4f}\nBootstrapped mean: {bootstrapped_mean_alignment:.4f}\n95% CI: [{alignment_conf_min:.4f}, {alignment_conf_max:.4f}]\nP(>0.5): {p_value_gt_half:.4f}\nSignificant: {is_alignment_significant}'
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    
    # Save sign alignment plot
    alignment_plot_path = os.path.join(save_dir, f"sign_alignment_{bout_name}.png")
    plt.savefig(alignment_plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Sign alignment plot saved to: {alignment_plot_path}")
    plt.show()
    
    return {
        'bout_name': bout_name,
        'n_days': len(velocity_hourly_diffs_per_day),
        'n_valid_days': len(valid_corrs),
        'daily_correlations': daily_correlations,
        'mean_correlation': mean_corr,
        'bootstrapped_mean': bootstrapped_mean,
        'ci_lower': conf_min,
        'ci_upper': conf_max,
        'p_value_gt_zero': p_value_gt_zero,
        'p_value_lt_zero': p_value_lt_zero,
        'is_significant': is_significant,
        'bootstrapped_corrs': bootstrapped_corrs,
        'daily_sign_alignment_proportions': daily_sign_alignment_proportions.tolist(),
        'mean_alignment': mean_alignment,
        'bootstrapped_mean_alignment': bootstrapped_mean_alignment,
        'alignment_ci_lower': alignment_conf_min,
        'alignment_ci_upper': alignment_conf_max,
        'p_value_gt_half': p_value_gt_half,
        'is_alignment_significant': is_alignment_significant,
        'bootstrapped_alignments': bootstrapped_alignments
    }


if __name__ == '__main__':
    # Example usage - update bout_name to match your data
    # bout_name format: "{site_name}_{start_day}_{end_day}"
    # e.g., "beer_2024-08-01_2024-08-10" or "shack_2024-08-26_2024-09-18"
    
    bout_name = "rain_2024-11-15_2024-12-05"  # Update this to match your bout
    
    results = analyze_velocity_count_correlation(bout_name)
    
    if results:
        print("\nSummary:")
        print(f"  Bout: {results['bout_name']}")
        print(f"  Total days: {results['n_days']}")
        print(f"  Valid days: {results['n_valid_days']}")
        print(f"\n  CORRELATION ANALYSIS:")
        print(f"    Mean correlation: {results['mean_correlation']:.4f}")
        print(f"    Bootstrapped mean: {results['bootstrapped_mean']:.4f}")
        print(f"    95% CI: [{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]")
        print(f"    P(correlation > 0): {results['p_value_gt_zero']:.4f}")
        print(f"    Significant: {results['is_significant']}")
        print(f"\n  SIGN ALIGNMENT ANALYSIS:")
        print(f"    Mean alignment proportion: {results['mean_alignment']:.4f}")
        print(f"    Bootstrapped mean alignment: {results['bootstrapped_mean_alignment']:.4f}")
        print(f"    95% CI: [{results['alignment_ci_lower']:.4f}, {results['alignment_ci_upper']:.4f}]")
        print(f"    P(alignment > 0.5): {results['p_value_gt_half']:.4f}")
        print(f"    Significantly > 0.5: {results['is_alignment_significant']}")

