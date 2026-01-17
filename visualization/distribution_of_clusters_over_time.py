#!/usr/bin/env python3
"""
Distribution of Clusters Over Time Analysis

This script analyzes the temporal distribution of trajectory clusters across hours and days.
It creates visualizations showing how cluster composition changes throughout the day and across days.

Usage:
    python distribution_of_clusters_over_time.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

def load_trajectory_data(file_path):
    """
    Load the comprehensive trajectory analysis data.
    
    Parameters
    ----------
    file_path : str
        Path to the comprehensive_trajectory_analysis.csv file
    
    Returns
    -------
    pd.DataFrame
        Loaded trajectory data
    """
    print(f"Loading trajectory data from {file_path}...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Load the data
    df = pd.read_csv(file_path)
    
    print(f"✅ Loaded {len(df):,} trajectories")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Hours: {sorted(df['hour'].unique())}")
    print(f"   Clusters: {sorted(df['cluster_label'].unique())}")
    print(f"   File size: {os.path.getsize(file_path) / (1024*1024):.1f} MB")
    
    return df

def calculate_cluster_distributions(df):
    """
    Calculate cluster distributions for each hour-day combination.
    Combines clusters: 0&3 -> 0, 1&2 -> 1
    
    Parameters
    ----------
    df : pd.DataFrame
        Trajectory data with hour, day, and cluster_label columns
    
    Returns
    -------
    pd.DataFrame
        Distribution data with fractions for each cluster per hour-day
    """
    print("Calculating cluster distributions...")
    print("Combining clusters: 0&3 -> 0, 1&2 -> 1")
    
    # Create a copy of the dataframe to modify
    df_combined = df.copy()
    
    # Combine clusters: 0&3 -> 0, 1&2 -> 1
    df_combined['cluster_label'] = df_combined['cluster_label'].replace({3: 0, 2: 1})
    
    # Group by hour and day, count trajectories per cluster
    cluster_counts = df_combined.groupby(['hour', 'day', 'cluster_label']).size().reset_index(name='count')
    
    # Calculate total trajectories per hour-day
    total_per_hour_day = df_combined.groupby(['hour', 'day']).size().reset_index(name='total')
    
    # Merge to get fractions
    distribution_data = cluster_counts.merge(total_per_hour_day, on=['hour', 'day'])
    distribution_data['fraction'] = distribution_data['count'] / distribution_data['total']
    
    # Create hour-day index for plotting
    distribution_data['hour_day'] = distribution_data['hour'] + distribution_data['day'] * 24
    
    print(f"✅ Calculated distributions for {len(distribution_data)} hour-day-cluster combinations")
    print(f"   Combined clusters: {sorted(distribution_data['cluster_label'].unique())}")
    
    return distribution_data

def plot_cluster_distributions(distribution_data, output_dir='plots'):
    """
    Create visualizations of cluster distributions over time.
    
    Parameters
    ----------
    distribution_data : pd.DataFrame
        Distribution data with fractions per cluster
    output_dir : str
        Directory to save plots
    """
    print("Creating cluster distribution plots...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique clusters and colors
    clusters = sorted(distribution_data['cluster_label'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))
    
    # Create the main plot
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Plot each cluster as a separate line
    for i, cluster in enumerate(clusters):
        cluster_data = distribution_data[distribution_data['cluster_label'] == cluster]
        
        # Sort by hour_day for proper line plotting
        cluster_data = cluster_data.sort_values('hour_day')
        
        # Create appropriate label for combined clusters
        if cluster == 0:
            label = 'Cluster 0 (0+3)'
        elif cluster == 1:
            label = 'Cluster 1 (1+2)'
        else:
            label = f'Cluster {cluster}'
        
        ax.plot(cluster_data['hour_day'], cluster_data['fraction'], 
                color=colors[i], linewidth=2, marker='o', markersize=4,
                label=label, alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel('Time (Hours × Days)', fontsize=14)
    ax.set_ylabel('Fraction of Trajectories', fontsize=14)
    ax.set_title('Distribution of Trajectory Clusters Over Time', fontsize=16, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Format x-axis to show hours and days
    max_hour_day = distribution_data['hour_day'].max()
    n_days = int(max_hour_day / 24) + 1
    
    # Add day separators
    for day in range(n_days):
        ax.axvline(x=day * 24, color='gray', linestyle='--', alpha=0.5)
    
    # Set x-axis ticks to show hours
    hour_ticks = range(0, max_hour_day + 1, 6)  # Every 6 hours
    ax.set_xticks(hour_ticks)
    ax.set_xticklabels([f'D{int(h/24)} H{h%24}' for h in hour_ticks], rotation=45)
    
    # Ensure y-axis shows fractions properly
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(output_dir, 'cluster_distribution_over_time.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Main plot saved to {output_file}")
    
    # Create a heatmap version
    create_heatmap_visualization(distribution_data, output_dir)
    
    # Create hourly average plot
    create_hourly_average_plot(distribution_data, output_dir)
    
    plt.show()

def create_heatmap_visualization(distribution_data, output_dir):
    """
    Create a heatmap showing cluster distributions.
    """
    print("Creating heatmap visualization...")
    
    # Pivot data for heatmap
    heatmap_data = distribution_data.pivot_table(
        index='hour', 
        columns='cluster_label', 
        values='fraction', 
        aggfunc='mean'
    ).fillna(0)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.heatmap(heatmap_data.T, annot=True, fmt='.2f', cmap='viridis', 
                ax=ax, cbar_kws={'label': 'Fraction of Trajectories'})
    
    ax.set_title('Cluster Distribution Heatmap (Average Across Days)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Hour of Day', fontsize=14)
    ax.set_ylabel('Cluster', fontsize=14)
    
    plt.tight_layout()
    
    # Save heatmap
    heatmap_file = os.path.join(output_dir, 'cluster_distribution_heatmap.png')
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    print(f"✅ Heatmap saved to {heatmap_file}")
    
    plt.show()

def create_hourly_average_plot(distribution_data, output_dir):
    """
    Create a plot showing average cluster distributions by hour of day.
    """
    print("Creating hourly average plot...")
    
    # Calculate hourly averages
    hourly_avg = distribution_data.groupby(['hour', 'cluster_label'])['fraction'].mean().reset_index()
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 8))
    
    clusters = sorted(hourly_avg['cluster_label'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))
    
    for i, cluster in enumerate(clusters):
        cluster_data = hourly_avg[hourly_avg['cluster_label'] == cluster]
        
        # Create appropriate label for combined clusters
        if cluster == 0:
            label = 'Cluster 0 (0+3)'
        elif cluster == 1:
            label = 'Cluster 1 (1+2)'
        else:
            label = f'Cluster {cluster}'
        
        ax.plot(cluster_data['hour'], cluster_data['fraction'], 
                color=colors[i], linewidth=3, marker='o', markersize=6,
                label=label, alpha=0.8)
    
    ax.set_xlabel('Hour of Day', fontsize=14)
    ax.set_ylabel('Average Fraction of Trajectories', fontsize=14)
    ax.set_title('Average Cluster Distribution by Hour of Day', fontsize=16, fontweight='bold')
    
    ax.set_xlim(0, 23)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add day/night shading
    ax.axvspan(0, 6, alpha=0.1, color='blue', label='Night')
    ax.axvspan(6, 18, alpha=0.1, color='yellow', label='Day')
    ax.axvspan(18, 24, alpha=0.1, color='orange', label='Evening')
    
    plt.tight_layout()
    
    # Save hourly average plot
    hourly_file = os.path.join(output_dir, 'hourly_average_cluster_distribution.png')
    plt.savefig(hourly_file, dpi=300, bbox_inches='tight')
    print(f"✅ Hourly average plot saved to {hourly_file}")
    
    plt.show()

def print_summary_statistics(distribution_data, df):
    """
    Print summary statistics about cluster distributions and feature values.
    Note: This function works with the original df (before cluster combining) for feature analysis.
    """
    print("\n📊 CLUSTER DISTRIBUTION SUMMARY")
    print("=" * 50)
    print("Note: Clusters 0&3 combined, 1&2 combined for distribution analysis")
    
    # Overall cluster distribution
    overall_dist = distribution_data.groupby('cluster_label')['fraction'].mean()
    print(f"\nOverall cluster distribution (average fractions):")
    for cluster, fraction in overall_dist.items():
        if cluster == 0:
            print(f"  Cluster {cluster} (0+3): {fraction:.3f} ({fraction*100:.1f}%)")
        elif cluster == 1:
            print(f"  Cluster {cluster} (1+2): {fraction:.3f} ({fraction*100:.1f}%)")
        else:
            print(f"  Cluster {cluster}: {fraction:.3f} ({fraction*100:.1f}%)")
    
    # Hourly patterns
    print(f"\nHourly patterns:")
    hourly_stats = distribution_data.groupby('hour')['fraction'].agg(['mean', 'std'])
    print(f"  Average fraction per hour: {hourly_stats['mean'].mean():.3f} ± {hourly_stats['std'].mean():.3f}")
    
    # Peak hours for each cluster with feature analysis
    print(f"\nPeak hours and feature analysis for each cluster:")

    feature_columns = ['efficiency', 'straight_distance', 'avg_angle_change', 'total_length']
    
    for cluster in sorted(distribution_data['cluster_label'].unique()):
        cluster_data = distribution_data[distribution_data['cluster_label'] == cluster]
        peak_hour = cluster_data.groupby('hour')['fraction'].mean().idxmax()
        peak_fraction = cluster_data.groupby('hour')['fraction'].mean().max()
        
        cluster_name = f"{cluster} (0+3)" if cluster == 0 else f"{cluster} (1+2)" if cluster == 1 else str(cluster)
        print(f"\n  🔍 Cluster {cluster_name}: Peak at hour {peak_hour} ({peak_fraction:.3f})")
        
        # Get feature data for this cluster at peak hour (need to check original clusters)
        if cluster == 0:
            # Cluster 0+3: check both original clusters 0 and 3
            cluster_df_0 = df[df['cluster_label'] == 0]
            cluster_df_3 = df[df['cluster_label'] == 3]
            cluster_df = pd.concat([cluster_df_0, cluster_df_3], ignore_index=True)
        elif cluster == 1:
            # Cluster 1+2: check both original clusters 1 and 2
            cluster_df_1 = df[df['cluster_label'] == 1]
            cluster_df_2 = df[df['cluster_label'] == 2]
            cluster_df = pd.concat([cluster_df_1, cluster_df_2], ignore_index=True)
        else:
            cluster_df = df[df['cluster_label'] == cluster]
        
        peak_hour_data = cluster_df[cluster_df['hour'] == peak_hour]
        
        if len(peak_hour_data) > 0:
            print(f"     📈 Feature means at peak hour {peak_hour}:")
            for feature in feature_columns:
                if feature in peak_hour_data.columns:
                    mean_val = peak_hour_data[feature].mean()
                    std_val = peak_hour_data[feature].std()
                    print(f"       {feature:20s}: {mean_val:.3f} ± {std_val:.3f}")
        
        # Also show overall cluster feature means
        print(f"     📊 Overall cluster {cluster_name} feature means:")
        for feature in feature_columns:
            if feature in cluster_df.columns:
                mean_val = cluster_df[feature].mean()
                std_val = cluster_df[feature].std()
                print(f"       {feature:20s}: {mean_val:.3f} ± {std_val:.3f}")
    
    # Hourly feature analysis for all clusters
    print(f"\n🕐 HOURLY FEATURE ANALYSIS (averaged over all days)")
    print("=" * 60)
    
    for hour in sorted(df['hour'].unique()):
        hour_data = df[df['hour'] == hour]
        print(f"\n  Hour {hour:2d}: {len(hour_data):,} trajectories")
        
        # Show combined clusters
        for cluster in sorted(distribution_data['cluster_label'].unique()):
            if cluster == 0:
                # Cluster 0+3
                cluster_hour_data_0 = hour_data[hour_data['cluster_label'] == 0]
                cluster_hour_data_3 = hour_data[hour_data['cluster_label'] == 3]
                cluster_hour_data = pd.concat([cluster_hour_data_0, cluster_hour_data_3], ignore_index=True)
                cluster_name = "0+3"
            elif cluster == 1:
                # Cluster 1+2
                cluster_hour_data_1 = hour_data[hour_data['cluster_label'] == 1]
                cluster_hour_data_2 = hour_data[hour_data['cluster_label'] == 2]
                cluster_hour_data = pd.concat([cluster_hour_data_1, cluster_hour_data_2], ignore_index=True)
                cluster_name = "1+2"
            else:
                cluster_hour_data = hour_data[hour_data['cluster_label'] == cluster]
                cluster_name = str(cluster)
            
            if len(cluster_hour_data) > 0:
                print(f"    Cluster {cluster_name} ({len(cluster_hour_data):,} trajectories):")
                for feature in feature_columns:
                    if feature in cluster_hour_data.columns:
                        mean_val = cluster_hour_data[feature].mean()
                        print(f"      {feature:20s}: {mean_val:.3f}")

def main():
    """
    Main function to analyze cluster distributions over time.
    """
    print("🔍 CLUSTER DISTRIBUTION OVER TIME ANALYSIS")
    print("=" * 60)
    
    # Configuration
    data_file = '/home/tarun/Desktop/plots_for_committee_meeting/trajectory_clustering/beer-tree-08-01-2024_to_08-10-2024/comprehensive_trajectory_analysis.csv'
    output_dir = 'cluster_distribution_plots'
    
    try:
        # Load data
        df = load_trajectory_data(data_file)
        
        # Calculate distributions
        distribution_data = calculate_cluster_distributions(df)
        
        # Print summary statistics
        print_summary_statistics(distribution_data, df)
        
        # Create visualizations
        plot_cluster_distributions(distribution_data, output_dir)
        
        print(f"\n🎉 Analysis complete! Plots saved to {output_dir}/")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return

if __name__ == "__main__":
    main()