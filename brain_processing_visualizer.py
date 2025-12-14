#!/usr/bin/env python3
"""
Comprehensive Brain Processing Visualization Tool
Creates visualizations for pattern recognition, memory, and multi-region coordination
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec
from final_enhanced_brain import FinalEnhancedBrain
import time

class BrainProcessingVisualizer:
    """Visualization tool for brain processing systems"""
    
    def __init__(self):
        self.brain = FinalEnhancedBrain(total_neurons=10000)
    
    def visualize_pattern_recognition_pipeline(self, pattern, pattern_name="Test Pattern"):
        """Visualize pattern recognition processing pipeline"""
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
        
        # Step 1: Input Pattern
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(pattern[:200] if len(pattern) > 200 else pattern, 'b-', linewidth=1.5)
        ax1.set_title(f'Input Pattern: {pattern_name}', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Element Index')
        ax1.set_ylabel('Value')
        ax1.grid(True, alpha=0.3)
        
        # Analyze pattern
        result = self.brain.enhanced_pattern_recognition(pattern)
        
        # Step 2: Density Analysis
        ax2 = fig.add_subplot(gs[0, 1])
        threshold = np.median(pattern) if len(pattern) > 0 else 0.0
        density = np.sum(np.abs(pattern) > threshold) / len(pattern)
        is_sparse = density < 0.3
        
        categories = ['Dense\n(≥0.3)', 'Sparse\n(<0.3)']
        values = [1.0 - density, density] if is_sparse else [density, 1.0 - density]
        colors = ['#3498DB', '#E74C3C']
        bars = ax2.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_title(f'Pattern Density: {density:.3f}', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Proportion')
        ax2.set_ylim([0, 1.0])
        
        # Add text
        ax2.text(0.5, 0.5, f'Type: {"Sparse" if is_sparse else "Dense"}', 
                ha='center', va='center', fontsize=14, fontweight='bold',
                transform=ax2.transAxes)
        
        # Step 3: Feature Extraction Visualization
        ax3 = fig.add_subplot(gs[0, 2])
        if is_sparse:
            # Show density features
            chunk_size = max(1, len(pattern) // 20)
            features = []
            for i in range(0, len(pattern), chunk_size):
                chunk = pattern[i:i+chunk_size]
                chunk_density = np.sum(np.abs(chunk) > threshold) / len(chunk) if len(chunk) > 0 else 0.0
                features.append(chunk_density)
            ax3.bar(range(len(features)), features, color='#E74C3C', alpha=0.7)
            ax3.set_title('Density Features', fontsize=12, fontweight='bold')
        else:
            # Show edge features
            features = []
            for i in range(0, len(pattern) - 5, 5):
                window = pattern[i:i+5]
                edge_strength = np.std(window)
                features.append(edge_strength)
            ax3.bar(range(len(features)), features, color='#3498DB', alpha=0.7)
            ax3.set_title('Edge Features', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Feature Index')
        ax3.set_ylabel('Feature Strength')
        ax3.grid(True, alpha=0.3)
        
        # Step 4: Recognition Results
        ax4 = fig.add_subplot(gs[1, :])
        metrics = ['Recognition\nScore', 'Confidence', 'Features\nDetected', 'Pattern\nRecognized']
        values = [
            result['recognition_score'],
            result['confidence'],
            result['features_detected'] / 100.0,  # Normalize
            1.0 if result['pattern_recognized'] else 0.0
        ]
        colors_metrics = ['#9B59B6', '#1ABC9C', '#F39C12', '#E74C3C' if not result['pattern_recognized'] else '#27AE60']
        
        bars = ax4.bar(metrics, values, color=colors_metrics, alpha=0.8, edgecolor='black', linewidth=2)
        ax4.set_title('Recognition Results', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Score', fontsize=12)
        ax4.set_ylim([0, 1.1])
        ax4.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val, metric in zip(bars, values, metrics):
            height = bar.get_height()
            if metric == 'Pattern\nRecognized':
                label = 'Yes' if val > 0.5 else 'No'
            elif metric == 'Features\nDetected':
                label = f'{result["features_detected"]}'
            else:
                label = f'{val:.3f}'
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    label, ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Step 5: Processing Pipeline Diagram
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        # Draw pipeline
        stages = [
            ('Input\nNormalization', '#3498DB'),
            ('Sparsity\nDetection', '#9B59B6'),
            ('Feature\nExtraction', '#1ABC9C'),
            ('Pattern\nIntegration', '#F39C12'),
            ('Recognition\nScoring', '#E74C3C'),
            ('Confidence\nCalculation', '#27AE60')
        ]
        
        x_positions = np.linspace(0.1, 0.9, len(stages))
        y_center = 0.5
        
        for i, (stage_name, color) in enumerate(stages):
            # Draw box
            box = FancyBboxPatch((x_positions[i] - 0.08, y_center - 0.15),
                               0.16, 0.3, boxstyle="round,pad=0.01",
                               facecolor=color, edgecolor='black', linewidth=2,
                               transform=ax5.transAxes)
            ax5.add_patch(box)
            
            # Add text
            ax5.text(x_positions[i], y_center, stage_name,
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    transform=ax5.transAxes, color='white')
            
            # Draw arrow
            if i < len(stages) - 1:
                arrow = FancyArrowPatch((x_positions[i] + 0.08, y_center),
                                        (x_positions[i+1] - 0.08, y_center),
                                        arrowstyle='->', mutation_scale=20,
                                        linewidth=2, color='black',
                                        transform=ax5.transAxes)
                ax5.add_patch(arrow)
        
        ax5.set_xlim([0, 1])
        ax5.set_ylim([0, 1])
        ax5.set_title('Pattern Recognition Processing Pipeline', fontsize=14, fontweight='bold', pad=20)
        
        plt.suptitle(f'Pattern Recognition Analysis: {pattern_name}', fontsize=16, fontweight='bold', y=0.98)
        return fig
    
    def visualize_memory_system(self, test_patterns):
        """Visualize memory system storage and recall"""
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Clear memory for clean test
        self.brain.memory_system['working_memory'] = []
        self.brain.memory_system['long_term_memory'] = []
        
        # Storage visualization
        ax1 = fig.add_subplot(gs[0, 0])
        storage_results = []
        for i, pattern in enumerate(test_patterns):
            result = self.brain.enhanced_memory_operations('store', pattern)
            storage_results.append(result.get('stored', False))
        
        colors_storage = ['#27AE60' if r else '#E74C3C' for r in storage_results]
        bars = ax1.bar(range(len(test_patterns)), [1 if r else 0 for r in storage_results],
                      color=colors_storage, alpha=0.8, edgecolor='black')
        ax1.set_title('Memory Storage Results', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Pattern Index')
        ax1.set_ylabel('Stored (1) / Not Stored (0)')
        ax1.set_xticks(range(len(test_patterns)))
        ax1.set_ylim([0, 1.2])
        ax1.grid(axis='y', alpha=0.3)
        
        # Add labels
        for i, (bar, stored) in enumerate(zip(bars, storage_results)):
            label = '✓' if stored else '✗'
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                    label, ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # Memory capacity visualization
        ax2 = fig.add_subplot(gs[0, 1])
        status = self.brain.enhanced_memory_operations('capacity_status')
        memory_data = {
            'Working\nMemory': status['working_memory_items'],
            'Long-Term\nMemory': status['long_term_memory_items'],
            'Capacity\nUsed': status['total_capacity_used'] * 7  # Convert to items
        }
        
        bars = ax2.bar(memory_data.keys(), list(memory_data.values()),
                      color=['#3498DB', '#9B59B6', '#F39C12'], alpha=0.8, edgecolor='black')
        ax2.set_title('Memory Status', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Items')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add capacity line
        ax2.axhline(y=7, color='r', linestyle='--', linewidth=2, label='WM Capacity')
        ax2.legend()
        
        # Recall visualization
        ax3 = fig.add_subplot(gs[0, 2])
        recall_results = []
        similarities = []
        for i, pattern in enumerate(test_patterns):
            # Add noise for recall test
            noisy_pattern = pattern + np.random.normal(0, 0.1, len(pattern))
            result = self.brain.enhanced_memory_operations('recall', noisy_pattern)
            recall_results.append(result.get('recalled', False))
            similarities.append(result.get('similarity', 0.0))
        
        x_pos = np.arange(len(test_patterns))
        colors_recall = ['#27AE60' if r else '#E74C3C' for r in recall_results]
        bars = ax3.bar(x_pos, similarities, color=colors_recall, alpha=0.8, edgecolor='black')
        ax3.set_title('Memory Recall Results', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Pattern Index')
        ax3.set_ylabel('Similarity Score')
        ax3.set_xticks(x_pos)
        ax3.set_ylim([0, 1.1])
        ax3.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, label='Recall Threshold')
        ax3.grid(axis='y', alpha=0.3)
        ax3.legend()
        
        # Memory system architecture
        ax4 = fig.add_subplot(gs[1, :])
        ax4.axis('off')
        
        # Draw memory architecture
        stages = [
            ('Input\nPattern', '#3498DB'),
            ('Pattern\nAnalysis', '#9B59B6'),
            ('Threshold\nCheck', '#1ABC9C'),
            ('Working\nMemory', '#F39C12'),
            ('Long-Term\nMemory', '#E74C3C')
        ]
        
        x_positions = np.linspace(0.1, 0.9, len(stages))
        y_store = 0.7
        y_recall = 0.3
        
        # Storage path
        for i, (stage_name, color) in enumerate(stages):
            box = FancyBboxPatch((x_positions[i] - 0.08, y_store - 0.1),
                               0.16, 0.2, boxstyle="round,pad=0.01",
                               facecolor=color, edgecolor='black', linewidth=2,
                               transform=ax4.transAxes)
            ax4.add_patch(box)
            ax4.text(x_positions[i], y_store, stage_name,
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    transform=ax4.transAxes, color='white')
            
            if i < len(stages) - 1:
                arrow = FancyArrowPatch((x_positions[i] + 0.08, y_store),
                                        (x_positions[i+1] - 0.08, y_store),
                                        arrowstyle='->', mutation_scale=15,
                                        linewidth=2, color='black',
                                        transform=ax4.transAxes)
                ax4.add_patch(arrow)
        
        # Recall path (reverse)
        recall_stages = stages[::-1]
        for i, (stage_name, color) in enumerate(recall_stages):
            box = FancyBboxPatch((x_positions[len(stages)-1-i] - 0.08, y_recall - 0.1),
                               0.16, 0.2, boxstyle="round,pad=0.01",
                               facecolor=color, edgecolor='black', linewidth=2,
                               transform=ax4.transAxes)
            ax4.add_patch(box)
            ax4.text(x_positions[len(stages)-1-i], y_recall, stage_name,
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    transform=ax4.transAxes, color='white')
            
            if i < len(recall_stages) - 1:
                arrow = FancyArrowPatch((x_positions[len(stages)-1-i] - 0.08, y_recall),
                                        (x_positions[len(stages)-2-i] + 0.08, y_recall),
                                        arrowstyle='->', mutation_scale=15,
                                        linewidth=2, color='blue',
                                        transform=ax4.transAxes)
                ax4.add_patch(arrow)
        
        ax4.text(0.5, 0.85, 'STORAGE PATH', ha='center', fontsize=12, fontweight='bold',
                transform=ax4.transAxes)
        ax4.text(0.5, 0.15, 'RECALL PATH', ha='center', fontsize=12, fontweight='bold',
                transform=ax4.transAxes, color='blue')
        
        ax4.set_xlim([0, 1])
        ax4.set_ylim([0, 1])
        ax4.set_title('Memory System Architecture', fontsize=14, fontweight='bold', pad=20)
        
        plt.suptitle('Memory System Analysis', fontsize=16, fontweight='bold', y=0.98)
        return fig
    
    def visualize_multi_region_coordination(self, stimulus):
        """Visualize multi-region brain coordination"""
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Process stimulus
        result = self.brain.multi_region_processing(stimulus)
        region_activities = result['region_activities']
        
        # Region activity bars
        ax1 = fig.add_subplot(gs[0, 0])
        regions = list(region_activities.keys())
        activities = list(region_activities.values())
        colors_regions = ['#E74C3C', '#3498DB', '#9B59B6', '#F39C12', '#1ABC9C']
        
        bars = ax1.barh(regions, activities, color=colors_regions[:len(regions)], 
                       alpha=0.8, edgecolor='black', linewidth=2)
        ax1.set_title('Region Activity Levels', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Activity Level')
        ax1.set_xlim([0, 1.0])
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, activity in zip(bars, activities):
            width = bar.get_width()
            ax1.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                    f'{activity:.3f}', ha='left', va='center', fontsize=10, fontweight='bold')
        
        # Activity propagation timeline
        ax2 = fig.add_subplot(gs[0, 1])
        # Simulate propagation steps
        steps = ['Sensory', 'Association', 'Memory', 'Executive', 'Motor']
        step_activities = [
            region_activities.get('sensory_cortex', 0),
            region_activities.get('association_cortex', 0),
            region_activities.get('memory_hippocampus', 0),
            region_activities.get('executive_cortex', 0),
            region_activities.get('motor_cortex', 0)
        ]
        
        ax2.plot(steps, step_activities, 'o-', linewidth=3, markersize=10, color='#3498DB')
        ax2.fill_between(steps, step_activities, alpha=0.3, color='#3498DB')
        ax2.set_title('Activity Propagation', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Activity Level')
        ax2.set_ylim([0, max(step_activities) * 1.2 if max(step_activities) > 0 else 1.0])
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # Coordination metrics
        ax3 = fig.add_subplot(gs[0, 2])
        metrics = {
            'Coordination\nScore': result['coordination_score'],
            'Active\nRegions': result['active_regions'] / 5.0,
            'Total\nActivity': min(1.0, result['total_activity'] / 5.0)
        }
        
        bars = ax3.bar(metrics.keys(), list(metrics.values()),
                      color=['#27AE60', '#3498DB', '#F39C12'], alpha=0.8, edgecolor='black')
        ax3.set_title('Coordination Metrics', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Score')
        ax3.set_ylim([0, 1.1])
        ax3.grid(axis='y', alpha=0.3)
        
        # Add labels
        for bar, (key, val) in zip(bars, metrics.items()):
            height = bar.get_height()
            if key == 'Active\nRegions':
                label = f'{result["active_regions"]}/5'
            else:
                label = f'{val:.3f}'
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    label, ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Brain region architecture diagram
        ax4 = fig.add_subplot(gs[1, :])
        ax4.axis('off')
        
        # Draw brain regions
        region_positions = {
            'sensory_cortex': (0.2, 0.8),
            'association_cortex': (0.5, 0.8),
            'memory_hippocampus': (0.35, 0.5),
            'executive_cortex': (0.65, 0.5),
            'motor_cortex': (0.8, 0.2)
        }
        
        region_labels = {
            'sensory_cortex': 'Sensory\nCortex',
            'association_cortex': 'Association\nCortex',
            'memory_hippocampus': 'Memory\nHippocampus',
            'executive_cortex': 'Executive\nCortex',
            'motor_cortex': 'Motor\nCortex'
        }
        
        # Draw connections
        connections = [
            ('sensory_cortex', 'association_cortex'),
            ('association_cortex', 'memory_hippocampus'),
            ('memory_hippocampus', 'executive_cortex'),
            ('executive_cortex', 'motor_cortex'),
            ('sensory_cortex', 'executive_cortex')
        ]
        
        for source, target in connections:
            x1, y1 = region_positions[source]
            x2, y2 = region_positions[target]
            activity = region_activities.get(source, 0)
            if activity > 0.1:
                arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                       arrowstyle='->', mutation_scale=20,
                                       linewidth=2 + activity * 3,
                                       color=plt.cm.RdYlGn(activity),
                                       transform=ax4.transAxes)
                ax4.add_patch(arrow)
        
        # Draw regions
        for region_name, (x, y) in region_positions.items():
            activity = region_activities.get(region_name, 0)
            color = plt.cm.RdYlGn(activity)
            
            circle = plt.Circle((x, y), 0.08, color=color, 
                              edgecolor='black', linewidth=2,
                              transform=ax4.transAxes)
            ax4.add_patch(circle)
            
            ax4.text(x, y, region_labels[region_name],
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    transform=ax4.transAxes, color='white' if activity > 0.3 else 'black')
            
            # Activity label
            ax4.text(x, y - 0.12, f'{activity:.2f}',
                    ha='center', va='top', fontsize=8, fontweight='bold',
                    transform=ax4.transAxes)
        
        ax4.set_xlim([0, 1])
        ax4.set_ylim([0, 1])
        ax4.set_title('Multi-Region Brain Architecture', fontsize=14, fontweight='bold', pad=20)
        
        plt.suptitle('Multi-Region Coordination Analysis', fontsize=16, fontweight='bold', y=0.98)
        return fig

def main():
    """Main visualization function"""
    print("🎨 Creating Brain Processing Visualizations...")
    
    visualizer = BrainProcessingVisualizer()
    
    # Test patterns
    test_patterns = [
        np.sin(np.linspace(0, 4*np.pi, 1000)),  # Sine wave
        np.random.random(1000) > 0.7,  # Sparse random
        np.array([1, 0, 1, 0, 1] * 200),  # Alternating
        np.random.random(1000)  # Dense random
    ]
    
    pattern_names = ['Sine Wave', 'Sparse Random', 'Alternating', 'Dense Random']
    
    print("\n1. Visualizing Pattern Recognition Pipeline...")
    for pattern, name in zip(test_patterns, pattern_names):
        fig = visualizer.visualize_pattern_recognition_pipeline(pattern.astype(float), name)
        filename = f'pattern_recognition_{name.lower().replace(" ", "_")}.png'
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {filename}")
        plt.close(fig)
    
    print("\n2. Visualizing Memory System...")
    memory_patterns = [
        np.random.random(50) > 0.3,
        np.sin(np.linspace(0, 4*np.pi, 50)),
        np.random.random(50) > 0.7,
        np.array([1, 0, 1, 0, 1] * 10),
        np.random.random(50) * 0.5 + 0.5
    ]
    fig = visualizer.visualize_memory_system([p.astype(float) for p in memory_patterns])
    fig.savefig('memory_system_analysis.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: memory_system_analysis.png")
    plt.close(fig)
    
    print("\n3. Visualizing Multi-Region Coordination...")
    stimulus = {
        'sensory_input': np.random.random(500),
        'store_memory': np.random.random(100)
    }
    fig = visualizer.visualize_multi_region_coordination(stimulus)
    fig.savefig('multi_region_coordination.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: multi_region_coordination.png")
    plt.close(fig)
    
    print("\n✅ All visualizations created successfully!")

if __name__ == "__main__":
    main()

