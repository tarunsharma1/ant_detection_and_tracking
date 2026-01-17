"""
Analyze excavating ant trajectories using behavioral clustering features.

This script loads the specific excavating ant IDs and analyzes their trajectories
using the same feature extraction methods from clustering_trajectories.py.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mysql_dataset import database_helper
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

class ExcavationFeatureAnalyzer:
    """
    Analyzes excavating ant trajectories using behavioral clustering features.
    """
    
    def __init__(self, site_id=1, bin_size=20):
        self.site_id = site_id
        self.bin_size = bin_size
        self.excavating_ant_ids = [141690, 141996, 141841, 143040, 141807, 141960]
        
    def load_excavating_trajectories(self, target_date, target_hour):
        """
        Load trajectories for the specific excavating ant IDs.
        """
        print(f"Loading excavating ant trajectories for site {self.site_id}, {target_date} at hour {target_hour}...")
        
        # Convert target_date to datetime if string
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
        
        # Connect to database
        connection = database_helper.create_connection("localhost", "root", "master", "ant_colony_db")
        cursor = connection.cursor()
        
        query = """
        SELECT Counts.video_id, Counts.herdnet_tracking_with_direction_closest_boundary_method_csv,
        Videos.temperature, Videos.humidity, Videos.LUX, Videos.time_stamp, Videos.site_id 
        FROM Counts INNER JOIN Videos ON Counts.video_id=Videos.video_id
        WHERE Videos.site_id = %s
        """
        cursor.execute(query, (self.site_id,))
        table_rows = cursor.fetchall()
        
        df = pd.DataFrame(table_rows, columns=cursor.column_names)
        df["time_stamp"] = pd.to_datetime(df["time_stamp"])
        
        # Filter by site, date, and hour
        df_filtered = df.loc[
            (df.site_id == self.site_id) &
            (df.time_stamp.dt.date == target_date.date()) &
            (df.time_stamp.dt.hour == target_hour)
        ]
        
        if len(df_filtered) == 0:
            print(f"No videos found for site {self.site_id} on {target_date} at hour {target_hour}")
            return []
        
        print(f"Found {len(df_filtered)} videos for the specified hour")
        
        excavating_trajectories = []
        
        for _, video in df_filtered.iterrows():
            tracking_csv = video['herdnet_tracking_with_direction_closest_boundary_method_csv']
            try:
                data = pd.read_csv(tracking_csv)
                if len(data) == 0:
                    continue
                
                # Filter for excavating ant IDs only
                excavating_data = data[data['ant_id'].isin(self.excavating_ant_ids)]
                
                if len(excavating_data) == 0:
                    continue
                
                # Extract trajectories for each excavating ant
                for ant_id in excavating_data['ant_id'].unique():
                    ant_track = excavating_data[excavating_data['ant_id'] == ant_id].copy()
                    if len(ant_track) < 20:  # Skip very short tracks
                        continue
                        
                    # Calculate center points
                    ant_track['center_x'] = (ant_track['x1'] + ant_track['x2']) / 2
                    ant_track['center_y'] = (ant_track['y1'] + ant_track['y2']) / 2
                    
                    # Add metadata
                    ant_track['video_id'] = video['video_id']
                    ant_track['hour'] = video['time_stamp'].hour
                    ant_track['day'] = target_date.day
                    ant_track['date'] = target_date.date()
                    
                    excavating_trajectories.append(ant_track)
                    
            except Exception as e:
                print(f"Error loading {tracking_csv}: {e}")
                continue
        
        print(f"Loaded {len(excavating_trajectories)} excavating ant trajectories")
        return excavating_trajectories
    
    def extract_trajectory_features(self, trajectories):
        """
        Extract the same features used in clustering_trajectories.py
        """
        print("Extracting behavioral features for excavating ants...")
        features = []
        feature_names = []
        
        for i, traj in enumerate(trajectories):
            if len(traj) < 5:
                continue
                
            # Sort by frame number
            traj = traj.sort_values('frame_number').reset_index(drop=True)
            
            # Basic geometric features
            coords = traj[['center_x', 'center_y']].values
            
            # 1. Trajectory length
            total_length = np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1))
            
            # 2. Straight-line distance (efficiency)
            straight_distance = np.linalg.norm(coords[-1] - coords[0])
            efficiency = straight_distance / (total_length + 1e-6)
            
            # 3. Curvature (average angle change)
            if len(coords) > 2:
                vectors = np.diff(coords, axis=0)
                angles = np.arctan2(vectors[:, 1], vectors[:, 0])
                angle_changes = np.abs(np.diff(angles))
                angle_changes = np.minimum(angle_changes, 2*np.pi - angle_changes)
                avg_curvature = np.mean(angle_changes)
                max_curvature = np.max(angle_changes)
            else:
                avg_curvature = 0
                max_curvature = 0
            
            # 4. Loop detection (return to starting area)
            start_pos = coords[0]
            end_pos = coords[-1]
            loop_distance = np.linalg.norm(end_pos - start_pos)
            is_loop = loop_distance < 50  # Within 50 pixels of start
            
            # 5. Speed features
            if 'velocity' in traj.columns:
                avg_speed = traj['velocity'].mean()
                speed_std = traj['velocity'].std()
                max_speed = traj['velocity'].max()
            else:
                # Calculate speed from position changes
                frame_diff = traj['frame_number'].diff()
                distances = np.linalg.norm(np.diff(coords, axis=0), axis=1)
                speeds = distances / (frame_diff[1:] + 1e-6) * 20  # 20 fps
                avg_speed = np.mean(speeds)
                speed_std = np.std(speeds)
                max_speed = np.max(speeds)
            
            # 6. Direction consistency
            if len(coords) > 1:
                vectors = np.diff(coords, axis=0)
                if len(vectors) > 1:
                    # Calculate direction consistency
                    unit_vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-6)
                    direction_consistency = np.mean(np.dot(unit_vectors[:-1], unit_vectors[1:].T).diagonal())
                else:
                    direction_consistency = 0
            else:
                direction_consistency = 0
            
            # 7. Spatial coverage (bounding box area)
            bbox_area = (coords[:, 0].max() - coords[:, 0].min()) * (coords[:, 1].max() - coords[:, 1].min())
            
            # 8. Start/end positions (normalized)
            start_x_norm = coords[0, 0] / 1920
            start_y_norm = coords[0, 1] / 1080
            end_x_norm = coords[-1, 0] / 1920
            end_y_norm = coords[-1, 1] / 1080
            
            # 9. Duration
            duration = len(traj)
            
            # 10. Direction changes (zigzag behavior)
            if len(vectors) > 2:
                direction_changes = np.sum(np.abs(np.diff(np.arctan2(vectors[:, 1], vectors[:, 0]))) > np.pi/4)
            else:
                direction_changes = 0
            
            # 11. Advanced features
            # Fractal dimension (simplified)
            if len(coords) > 10:
                fractal_dim = self._calculate_fractal_dimension(coords)
            else:
                fractal_dim = 1.0
            
            # Entropy of movement directions
            if len(vectors) > 1:
                angles = np.arctan2(vectors[:, 1], vectors[:, 0])
                # Discretize angles into bins
                angle_bins = np.digitize(angles, np.linspace(-np.pi, np.pi, 9))
                direction_entropy = entropy(np.bincount(angle_bins))
            else:
                direction_entropy = 0
            
            # Pause detection (low speed periods)
            if len(speeds) > 1:
                pause_ratio = np.sum(speeds < 1.0) / len(speeds)  # Speeds < 1 pixel/frame
            else:
                pause_ratio = 0
            
            # Acceleration features
            if len(speeds) > 2:
                accelerations = np.diff(speeds)
                avg_acceleration = np.mean(np.abs(accelerations))
                acceleration_std = np.std(accelerations)
            else:
                avg_acceleration = 0
                acceleration_std = 0
            
            # Spatial distribution (how spread out the trajectory is)
            spatial_std_x = np.std(coords[:, 0])
            spatial_std_y = np.std(coords[:, 1])
            
            # Return frequency (how often the ant returns to previously visited areas)
            return_frequency = self._calculate_return_frequency(coords)
            
            feature_vector = [
                total_length,
                efficiency,
                avg_curvature,
                max_curvature,
                float(is_loop),
                avg_speed,
                speed_std,
                max_speed,
                direction_consistency,
                bbox_area,
                start_x_norm,
                start_y_norm,
                end_x_norm,
                end_y_norm,
                duration,
                direction_changes,
                fractal_dim,
                direction_entropy,
                pause_ratio,
                avg_acceleration,
                acceleration_std,
                spatial_std_x,
                spatial_std_y,
                return_frequency
            ]
            
            features.append(feature_vector)
        
        # Define feature names
        feature_names = [
            'Total Length', 'Efficiency', 'Avg Curvature', 'Max Curvature', 'Is Loop',
            'Avg Speed', 'Speed Std', 'Max Speed', 'Direction Consistency', 'Bbox Area',
            'Start X Norm', 'Start Y Norm', 'End X Norm', 'End Y Norm', 'Duration',
            'Direction Changes', 'Fractal Dimension', 'Direction Entropy', 'Pause Ratio',
            'Avg Acceleration', 'Acceleration Std', 'Spatial Std X', 'Spatial Std Y',
            'Return Frequency'
        ]
        
        self.features = np.array(features)
        self.feature_names = feature_names
        print(f"Extracted {len(features)} feature vectors with {len(feature_names)} features each")
        return self.features, feature_names
    
    def _calculate_fractal_dimension(self, coords, max_box_size=50):
        """Calculate fractal dimension using box-counting method."""
        if len(coords) < 10:
            return 1.0
        
        # Normalize coordinates
        coords_norm = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0) + 1e-6)
        
        box_sizes = np.logspace(0, np.log10(max_box_size), 5)
        counts = []
        
        for box_size in box_sizes:
            # Count boxes that contain trajectory points
            grid_size = 1.0 / box_size
            grid_x = np.floor(coords_norm[:, 0] / grid_size).astype(int)
            grid_y = np.floor(coords_norm[:, 1] / grid_size).astype(int)
            
            unique_boxes = len(set(zip(grid_x, grid_y)))
            counts.append(unique_boxes)
        
        # Fit line to log-log plot
        if len(counts) > 1 and np.all(np.array(counts) > 0):
            log_box_sizes = np.log(1.0 / box_sizes)
            log_counts = np.log(counts)
            slope = np.polyfit(log_box_sizes, log_counts, 1)[0]
            return max(1.0, min(2.0, -slope))  # Clamp between 1 and 2
        else:
            return 1.0
    
    def _calculate_return_frequency(self, coords, radius=30):
        """Calculate how often the ant returns to previously visited areas."""
        if len(coords) < 10:
            return 0
        
        returns = 0
        for i in range(10, len(coords)):  # Start checking after 10 points
            current_pos = coords[i]
            # Check if current position is close to any previous position
            distances = np.linalg.norm(coords[:i] - current_pos, axis=1)
            if np.any(distances < radius):
                returns += 1
        
        return returns / (len(coords) - 10) if len(coords) > 10 else 0
    
    def analyze_excavation_features(self, trajectories):
        """
        Analyze the behavioral features of excavating ants.
        """
        print("\n🔍 EXCAVATION BEHAVIORAL FEATURE ANALYSIS")
        print("=" * 60)
        
        # Extract features
        features, feature_names = self.extract_trajectory_features(trajectories)
        
        if len(features) == 0:
            print("No features extracted. Check trajectory data.")
            return
        
        # Calculate statistics for each feature
        print("\n📊 Feature Statistics for Excavating Ants:")
        print("-" * 50)
        
        for i, feature_name in enumerate(feature_names):
            feature_values = features[:, i]
            print(f"{feature_name:20s}: Mean={np.mean(feature_values):8.3f}, "
                  f"Std={np.std(feature_values):8.3f}, "
                  f"Min={np.min(feature_values):8.3f}, "
                  f"Max={np.max(feature_values):8.3f}")
        
        # Analyze key excavation indicators
        print("\n🎯 Key Excavation Indicators:")
        print("-" * 40)
        
        # Loop detection
        loop_ratio = np.mean(features[:, 4])  # Is Loop feature
        print(f"Loop Ratio: {loop_ratio:.3f} ({loop_ratio*100:.1f}% of trajectories are loops)")
        
        # Efficiency (low efficiency indicates complex paths)
        efficiency = np.mean(features[:, 1])
        print(f"Average Efficiency: {efficiency:.3f} (lower = more complex paths)")
        
        # Curvature (high curvature indicates turning behavior)
        avg_curvature = np.mean(features[:, 2])
        print(f"Average Curvature: {avg_curvature:.3f} (higher = more turning)")
        
        # Return frequency (high return frequency indicates excavation behavior)
        return_freq = np.mean(features[:, 23])
        print(f"Return Frequency: {return_freq:.3f} (higher = more excavation-like)")
        
        # Pause ratio (excavating ants might pause more)
        pause_ratio = np.mean(features[:, 18])
        print(f"Pause Ratio: {pause_ratio:.3f} (higher = more pausing)")
        
        # Create visualizations
        self._create_feature_visualizations(features, feature_names, trajectories)
        
        return features, feature_names
    
    def _create_feature_visualizations(self, features, feature_names, trajectories):
        """
        Create visualizations of the excavation features.
        """
        print("\n📈 Creating feature visualizations...")
        
        # Create a comprehensive figure
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Feature correlation heatmap
        ax1 = plt.subplot(3, 3, 1)
        corr_matrix = np.corrcoef(features.T)
        im = ax1.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax1.set_title('Feature Correlation Matrix')
        ax1.set_xlabel('Features')
        ax1.set_ylabel('Features')
        plt.colorbar(im, ax=ax1)
        
        # 2. Key excavation features box plot
        ax2 = plt.subplot(3, 3, 2)
        key_features = ['Efficiency', 'Is Loop', 'Return Frequency', 'Pause Ratio']
        key_indices = [1, 4, 23, 18]
        key_values = features[:, key_indices]
        
        bp = ax2.boxplot(key_values, labels=key_features, patch_artist=True)
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax2.set_title('Key Excavation Features')
        ax2.set_ylabel('Feature Value')
        plt.xticks(rotation=45)
        
        # 3. Trajectory length vs efficiency
        ax3 = plt.subplot(3, 3, 3)
        ax3.scatter(features[:, 0], features[:, 1], alpha=0.7, s=50)
        ax3.set_xlabel('Total Length')
        ax3.set_ylabel('Efficiency')
        ax3.set_title('Length vs Efficiency')
        ax3.grid(True, alpha=0.3)
        
        # 4. Loop detection visualization
        ax4 = plt.subplot(3, 3, 4)
        loop_trajectories = [trajectories[i] for i in range(len(trajectories)) if features[i, 4] == 1]
        non_loop_trajectories = [trajectories[i] for i in range(len(trajectories)) if features[i, 4] == 0]
        
        # Plot non-loop trajectories in light gray
        for traj in non_loop_trajectories[:5]:
            traj = traj.sort_values('frame_number')
            ax4.plot(traj['center_x'], traj['center_y'], 
                    color='lightgray', alpha=0.5, linewidth=1)
        
        # Plot loop trajectories in bright colors
        colors = plt.cm.tab10(np.linspace(0, 1, len(loop_trajectories)))
        for i, traj in enumerate(loop_trajectories[:10]):
            traj = traj.sort_values('frame_number')
            ax4.plot(traj['center_x'], traj['center_y'], 
                    color=colors[i], alpha=0.8, linewidth=2,
                    label=f'Loop {i+1}' if i < 5 else '')
            
            # Mark start and end points
            ax4.scatter(traj['center_x'].iloc[0], traj['center_y'].iloc[0], 
                       color='green', s=50, marker='o', zorder=5)
            ax4.scatter(traj['center_x'].iloc[-1], traj['center_y'].iloc[-1], 
                       color='red', s=50, marker='s', zorder=5)
        
        ax4.set_title(f'Loop Trajectories ({len(loop_trajectories)}/{len(trajectories)})')
        ax4.set_xlabel('X (pixels)')
        ax4.set_ylabel('Y (pixels)')
        ax4.set_aspect('equal')
        ax4.grid(True, alpha=0.3)
        if len(loop_trajectories) <= 5:
            ax4.legend()
        
        # 5. Speed analysis
        ax5 = plt.subplot(3, 3, 5)
        ax5.hist(features[:, 5], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax5.set_xlabel('Average Speed')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Speed Distribution')
        ax5.grid(True, alpha=0.3)
        
        # 6. Curvature analysis
        ax6 = plt.subplot(3, 3, 6)
        ax6.hist(features[:, 2], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax6.set_xlabel('Average Curvature')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Curvature Distribution')
        ax6.grid(True, alpha=0.3)
        
        # 7. PCA visualization
        ax7 = plt.subplot(3, 3, 7)
        if len(features) > 1:
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Apply PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(features_scaled)
            
            # Color by loop detection
            loop_mask = features[:, 4] == 1
            ax7.scatter(pca_result[~loop_mask, 0], pca_result[~loop_mask, 1], 
                       c='lightgray', alpha=0.6, s=50, label='Non-loop')
            ax7.scatter(pca_result[loop_mask, 0], pca_result[loop_mask, 1], 
                       c='red', alpha=0.8, s=50, label='Loop')
            
            ax7.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            ax7.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            ax7.set_title('PCA: Loop vs Non-loop')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        # 8. Feature importance (variance)
        ax8 = plt.subplot(3, 3, 8)
        feature_variance = np.var(features, axis=0)
        top_features = np.argsort(feature_variance)[-10:]  # Top 10 most variable features
        
        ax8.barh(range(len(top_features)), feature_variance[top_features])
        ax8.set_yticks(range(len(top_features)))
        ax8.set_yticklabels([feature_names[i] for i in top_features])
        ax8.set_xlabel('Variance')
        ax8.set_title('Top 10 Most Variable Features')
        ax8.grid(True, alpha=0.3)
        
        # 9. Summary statistics
        ax9 = plt.subplot(3, 3, 9)
        summary_text = f"Excavation Analysis Summary\n\n"
        summary_text += f"Total Trajectories: {len(trajectories)}\n"
        summary_text += f"Loop Trajectories: {np.sum(features[:, 4])}\n"
        summary_text += f"Loop Percentage: {np.mean(features[:, 4])*100:.1f}%\n"
        summary_text += f"Avg Efficiency: {np.mean(features[:, 1]):.3f}\n"
        summary_text += f"Avg Curvature: {np.mean(features[:, 2]):.3f}\n"
        summary_text += f"Avg Return Freq: {np.mean(features[:, 23]):.3f}\n"
        summary_text += f"Avg Pause Ratio: {np.mean(features[:, 18]):.3f}\n"
        
        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax9.set_xlim(0, 1)
        ax9.set_ylim(0, 1)
        ax9.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Save the analysis
        self._save_analysis_results(features, feature_names, trajectories)
    
    def _save_analysis_results(self, features, feature_names, trajectories):
        """
        Save the analysis results to files.
        """
        output_dir = '/home/tarun/Desktop/plots_for_committee_meeting/excavation_analysis/'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save feature data
        feature_df = pd.DataFrame(features, columns=feature_names)
        feature_df['ant_id'] = [traj['ant_id'].iloc[0] for traj in trajectories]
        feature_df['trajectory_length'] = [len(traj) for traj in trajectories]
        
        feature_df.to_csv(os.path.join(output_dir, 'excavation_features.csv'), index=False)
        
        # Save trajectory summary
        trajectory_summary = []
        for i, traj in enumerate(trajectories):
            trajectory_summary.append({
                'ant_id': traj['ant_id'].iloc[0],
                'trajectory_length': len(traj),
                'is_loop': features[i, 4],
                'efficiency': features[i, 1],
                'curvature': features[i, 2],
                'return_frequency': features[i, 23],
                'pause_ratio': features[i, 18]
            })
        
        summary_df = pd.DataFrame(trajectory_summary)
        summary_df.to_csv(os.path.join(output_dir, 'excavation_summary.csv'), index=False)
        
        print(f"\n💾 Analysis results saved to {output_dir}")
        print(f"   - excavation_features.csv: Full feature matrix")
        print(f"   - excavation_summary.csv: Key metrics per trajectory")

def main():
    """
    Main function to analyze excavating ant trajectories.
    """
    print("🔍 EXCAVATION BEHAVIORAL FEATURE ANALYSIS")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = ExcavationFeatureAnalyzer(site_id=1, bin_size=20)
    
    # Configuration
    target_date = '2024-08-02'
    target_hour = 12
    
    print(f"Analyzing excavating ants for site {analyzer.site_id} on {target_date} at hour {target_hour}")
    print(f"Excavating ant IDs: {analyzer.excavating_ant_ids}")
    
    # Load excavating trajectories
    trajectories = analyzer.load_excavating_trajectories(target_date, target_hour)
    
    if len(trajectories) == 0:
        print("❌ No excavating ant trajectories found")
        return
    
    print(f"✅ Loaded {len(trajectories)} excavating ant trajectories")
    
    # Analyze features
    features, feature_names = analyzer.analyze_excavation_features(trajectories)
    
    print("\n🎉 Excavation analysis complete!")
    print(f"\n📊 Summary:")
    print(f"   - Trajectories analyzed: {len(trajectories)}")
    print(f"   - Features extracted: {len(feature_names)}")
    print(f"   - Loop trajectories: {np.sum(features[:, 4])}")
    print(f"   - Loop percentage: {np.mean(features[:, 4])*100:.1f}%")

if __name__ == "__main__":
    main()
