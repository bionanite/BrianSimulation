#!/usr/bin/env python3
"""
Interactive Multi-Region Coordination Exploration Tool
Demonstrates how different brain regions coordinate during processing
"""

import numpy as np
from final_enhanced_brain import FinalEnhancedBrain
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

class MultiRegionExplorer:
    """Interactive explorer for multi-region coordination"""
    
    def __init__(self):
        self.brain = FinalEnhancedBrain(total_neurons=10000)
        self.processing_history = []
    
    def test_stimulus(self, stimulus, stimulus_name):
        """Test processing a stimulus through all regions"""
        print(f"\n{'='*70}")
        print(f"Testing Stimulus: {stimulus_name}")
        print(f"{'='*70}")
        
        # Process stimulus
        result = self.brain.multi_region_processing(stimulus)
        
        # Display results
        print(f"\n🧠 Region Activities:")
        region_activities = result['region_activities']
        for region_name, activity in region_activities.items():
            status = "✅ ACTIVE" if activity > 0.1 else "❌ INACTIVE"
            print(f"   {region_name.replace('_', ' ').title():20s}: {activity:.3f} {status}")
        
        print(f"\n📊 Processing Results:")
        processing_results = result.get('processing_results', {})
        for process_name, process_result in processing_results.items():
            if isinstance(process_result, dict):
                if 'decision_made' in process_result:
                    print(f"   {process_name}: Decision = {process_result['decision_made']}")
                elif 'confidence' in process_result:
                    print(f"   {process_name}: Confidence = {process_result['confidence']:.3f}")
                else:
                    print(f"   {process_name}: {process_result}")
            else:
                print(f"   {process_name}: {process_result:.3f}")
        
        print(f"\n🎯 Coordination Metrics:")
        print(f"   Coordination Score: {result['coordination_score']:.3f}")
        print(f"   Active Regions: {result['active_regions']}/5")
        print(f"   Total Activity: {result['total_activity']:.3f}")
        
        # Store history
        self.processing_history.append({
            'name': stimulus_name,
            'stimulus': stimulus,
            'result': result
        })
        
        return result
    
    def visualize_region_coordination(self, result, stimulus_name):
        """Visualize region coordination for a stimulus"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Multi-Region Coordination: {stimulus_name}', fontsize=16, fontweight='bold')
        
        region_activities = result['region_activities']
        regions = list(region_activities.keys())
        activities = list(region_activities.values())
        
        # Plot 1: Region activity bars
        ax1 = axes[0, 0]
        colors = ['#E74C3C', '#3498DB', '#9B59B6', '#F39C12', '#1ABC9C']
        bars = ax1.barh(regions, activities, color=colors[:len(regions)], 
                       alpha=0.8, edgecolor='black', linewidth=2)
        ax1.set_title('Region Activity Levels', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Activity Level')
        ax1.set_xlim([0, 1.0])
        ax1.grid(axis='x', alpha=0.3)
        
        # Add threshold line
        ax1.axvline(x=0.1, color='orange', linestyle='--', linewidth=2, label='Activation Threshold')
        ax1.legend()
        
        # Add value labels
        for bar, activity in zip(bars, activities):
            width = bar.get_width()
            status = "✓" if activity > 0.1 else "✗"
            ax1.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                    f'{activity:.3f} {status}', ha='left', va='center', 
                    fontsize=10, fontweight='bold')
        
        # Plot 2: Activity propagation timeline
        ax2 = axes[0, 1]
        step_order = ['sensory_cortex', 'association_cortex', 'memory_hippocampus', 
                     'executive_cortex', 'motor_cortex']
        step_activities = [region_activities.get(step, 0) for step in step_order]
        step_labels = [s.replace('_', ' ').title() for s in step_order]
        
        ax2.plot(step_labels, step_activities, 'o-', linewidth=3, markersize=12, color='#3498DB')
        ax2.fill_between(step_labels, step_activities, alpha=0.3, color='#3498DB')
        ax2.set_title('Activity Propagation Through Regions', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Activity Level')
        ax2.set_ylim([0, max(step_activities) * 1.2 if max(step_activities) > 0 else 1.0])
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add threshold lines
        thresholds = [0.1, 0.15, 0.15, 0.25, 0.3]
        for i, (label, threshold) in enumerate(zip(step_labels, thresholds)):
            ax2.axhline(y=threshold, xmin=i/len(step_labels), xmax=(i+1)/len(step_labels),
                       color='r', linestyle='--', alpha=0.5, linewidth=1)
        
        # Plot 3: Coordination metrics
        ax3 = axes[1, 0]
        metrics = {
            'Coordination\nScore': result['coordination_score'],
            'Active\nRegions': result['active_regions'] / 5.0,
            'Total\nActivity': min(1.0, result['total_activity'] / 5.0)
        }
        
        bars = ax3.bar(metrics.keys(), list(metrics.values()),
                      color=['#27AE60', '#3498DB', '#F39C12'], alpha=0.8, edgecolor='black', linewidth=2)
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
                    label, ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Plot 4: Brain architecture diagram
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Region positions
        region_positions = {
            'sensory_cortex': (0.2, 0.8),
            'association_cortex': (0.5, 0.8),
            'memory_hippocampus': (0.35, 0.5),
            'executive_cortex': (0.65, 0.5),
            'motor_cortex': (0.8, 0.2)
        }
        
        region_labels = {
            'sensory_cortex': 'Sensory',
            'association_cortex': 'Association',
            'memory_hippocampus': 'Memory',
            'executive_cortex': 'Executive',
            'motor_cortex': 'Motor'
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
                # Color and width based on activity
                color = plt.cm.RdYlGn(activity)
                width = 1 + activity * 3
                arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                       arrowstyle='->', mutation_scale=20,
                                       linewidth=width, color=color,
                                       transform=ax4.transAxes)
                ax4.add_patch(arrow)
        
        # Draw regions
        for region_name, (x, y) in region_positions.items():
            activity = region_activities.get(region_name, 0)
            color = plt.cm.RdYlGn(activity)
            
            circle = Circle((x, y), 0.08, color=color, 
                          edgecolor='black', linewidth=2,
                          transform=ax4.transAxes)
            ax4.add_patch(circle)
            
            ax4.text(x, y, region_labels[region_name],
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    transform=ax4.transAxes, color='white' if activity > 0.3 else 'black')
            
            # Activity label
            ax4.text(x, y - 0.12, f'{activity:.2f}',
                    ha='center', va='top', fontsize=9, fontweight='bold',
                    transform=ax4.transAxes)
        
        ax4.set_xlim([0, 1])
        ax4.set_ylim([0, 1])
        ax4.set_title('Brain Region Architecture', fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        filename = f'multi_region_{stimulus_name.lower().replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n💾 Visualization saved: {filename}")
        plt.close()
    
    def run_comprehensive_test(self):
        """Run comprehensive multi-region test"""
        print("\n" + "="*70)
        print("MULTI-REGION COORDINATION COMPREHENSIVE TEST SUITE")
        print("="*70)
        
        # Test stimuli
        test_stimuli = [
            ({
                'sensory_input': np.random.random(500),
                'store_memory': np.random.random(100)
            }, "Random Sensory + Memory"),
            ({
                'sensory_input': np.sin(np.linspace(0, 2*np.pi, 300))
            }, "Sine Wave Sensory"),
            ({
                'sensory_input': np.ones(300) * 0.8,
                'store_memory': np.random.random(100)
            }, "Uniform Sensory + Memory"),
            ({
                'sensory_input': np.array([1, 0, 1, 0, 1] * 100)
            }, "Alternating Pattern"),
            ({
                'sensory_input': np.random.random(200) > 0.5
            }, "Binary Random Pattern"),
            ({
                'sensory_input': np.zeros(200)
            }, "Zero Input (Baseline)")
        ]
        
        # Test each stimulus
        for stimulus, name in test_stimuli:
            result = self.test_stimulus(stimulus, name)
            self.visualize_region_coordination(result, name)
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)
        
        avg_coordination = np.mean([h['result']['coordination_score'] for h in self.processing_history])
        avg_active_regions = np.mean([h['result']['active_regions'] for h in self.processing_history])
        avg_total_activity = np.mean([h['result']['total_activity'] for h in self.processing_history])
        
        print(f"\n📈 Overall Performance:")
        print(f"   Stimuli Tested: {len(self.processing_history)}")
        print(f"   Average Coordination Score: {avg_coordination:.3f}")
        print(f"   Average Active Regions: {avg_active_regions:.1f}/5")
        print(f"   Average Total Activity: {avg_total_activity:.3f}")
        
        # Create comparison chart
        self.create_comparison_chart()
    
    def create_comparison_chart(self):
        """Create comparison chart of all stimuli"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('Multi-Region Coordination Comparison', fontsize=16, fontweight='bold')
        
        stimulus_names = [h['name'] for h in self.processing_history]
        coordination_scores = [h['result']['coordination_score'] for h in self.processing_history]
        active_regions = [h['result']['active_regions'] for h in self.processing_history]
        
        # Plot 1: Coordination scores
        ax1 = axes[0]
        colors = plt.cm.RdYlGn(coordination_scores)
        bars = ax1.barh(stimulus_names, coordination_scores, color=colors, 
                       alpha=0.8, edgecolor='black', linewidth=2)
        ax1.set_title('Coordination Scores by Stimulus', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Coordination Score')
        ax1.set_xlim([0, 1.1])
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars, coordination_scores):
            width = bar.get_width()
            ax1.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                    f'{score:.3f}', ha='left', va='center', fontsize=10, fontweight='bold')
        
        # Plot 2: Active regions
        ax2 = axes[1]
        bars = ax2.barh(stimulus_names, active_regions, color='#3498DB', 
                       alpha=0.8, edgecolor='black', linewidth=2)
        ax2.set_title('Active Regions by Stimulus', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Number of Active Regions')
        ax2.set_xlim([0, 5.5])
        ax2.axvline(x=5, color='r', linestyle='--', linewidth=2, label='Maximum (5)')
        ax2.legend()
        ax2.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, count in zip(bars, active_regions):
            width = bar.get_width()
            ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                    f'{count}/5', ha='left', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('multi_region_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n💾 Comparison chart saved: multi_region_comparison.png")
        plt.close()

def main():
    """Main function"""
    explorer = MultiRegionExplorer()
    explorer.run_comprehensive_test()
    print("\n✅ Multi-region coordination exploration complete!")

if __name__ == "__main__":
    main()

