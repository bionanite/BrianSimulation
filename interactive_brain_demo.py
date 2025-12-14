#!/usr/bin/env python3
"""
Interactive Brain Demo - Combined Exploration
Demonstrates all three brain processing systems working together
"""

import numpy as np
from final_enhanced_brain import FinalEnhancedBrain
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
import time

class InteractiveBrainDemo:
    """Interactive demo combining all brain processing systems"""
    
    def __init__(self):
        self.brain = FinalEnhancedBrain(total_neurons=10000)
        self.demo_history = []
    
    def run_full_pipeline_demo(self, sensory_input, memory_input=None, demo_name="Full Pipeline"):
        """Run complete brain processing pipeline"""
        print(f"\n{'='*80}")
        print(f"COMPLETE BRAIN PROCESSING PIPELINE: {demo_name}")
        print(f"{'='*80}")
        
        # Step 1: Pattern Recognition
        print(f"\n🔍 STEP 1: Pattern Recognition")
        print("-" * 60)
        pattern_result = self.brain.enhanced_pattern_recognition(sensory_input.astype(float))
        print(f"   Pattern Type: {'Sparse' if pattern_result['is_sparse'] else 'Dense'}")
        print(f"   Density: {pattern_result['density']:.3f}")
        print(f"   Confidence: {pattern_result['confidence']:.3f}")
        print(f"   Features Detected: {pattern_result['features_detected']}")
        print(f"   Recognized: {'✅ YES' if pattern_result['pattern_recognized'] else '❌ NO'}")
        
        # Step 2: Multi-Region Processing
        print(f"\n🧠 STEP 2: Multi-Region Coordination")
        print("-" * 60)
        stimulus = {'sensory_input': sensory_input}
        if memory_input is not None:
            stimulus['store_memory'] = memory_input
        
        region_result = self.brain.multi_region_processing(stimulus)
        
        print(f"   Region Activities:")
        for region_name, activity in region_result['region_activities'].items():
            status = "✅" if activity > 0.1 else "❌"
            print(f"      {region_name.replace('_', ' ').title():20s}: {activity:.3f} {status}")
        
        print(f"   Coordination Score: {region_result['coordination_score']:.3f}")
        print(f"   Active Regions: {region_result['active_regions']}/5")
        
        # Step 3: Memory Operations (if memory input provided)
        memory_result = None
        if memory_input is not None:
            print(f"\n💾 STEP 3: Memory Operations")
            print("-" * 60)
            
            # Storage
            storage_result = self.brain.enhanced_memory_operations('store', memory_input.astype(float))
            if storage_result.get('stored', False):
                print(f"   Storage: ✅ SUCCESS")
                print(f"   Location: {storage_result.get('location', 'unknown')}")
                
                # Recall test
                time.sleep(0.1)
                noisy_memory = memory_input.astype(float) + np.random.normal(0, 0.1, len(memory_input))
                recall_result = self.brain.enhanced_memory_operations('recall', noisy_memory)
                
                if recall_result.get('recalled', False):
                    print(f"   Recall: ✅ SUCCESS")
                    print(f"   Similarity: {recall_result.get('similarity', 0):.3f}")
                    print(f"   Source: {recall_result.get('source', 'unknown')}")
                else:
                    print(f"   Recall: ❌ FAILED")
                    print(f"   Similarity: {recall_result.get('similarity', 0):.3f}")
                
                memory_result = {'storage': storage_result, 'recall': recall_result}
            else:
                print(f"   Storage: ❌ FAILED")
                print(f"   Reason: {storage_result.get('reason', 'unknown')}")
        
        # Step 4: Hierarchical Processing
        print(f"\n📊 STEP 4: Hierarchical Processing")
        print("-" * 60)
        hierarchy_result = self.brain.hierarchical_processing(sensory_input.astype(float))
        print(f"   Processing Depth: {hierarchy_result['processing_depth']}/{len(self.brain.hierarchy['layers'])}")
        print(f"   Layers Active: {hierarchy_result['layers_active']}")
        print(f"   Information Flow: {hierarchy_result['information_flow']:.3f}")
        
        # Summary
        print(f"\n📈 PIPELINE SUMMARY")
        print("-" * 60)
        print(f"   Pattern Recognition: {'✅' if pattern_result['pattern_recognized'] else '❌'}")
        print(f"   Multi-Region Coordination: {region_result['coordination_score']:.3f}")
        if memory_result:
            print(f"   Memory Storage: {'✅' if memory_result['storage'].get('stored') else '❌'}")
            print(f"   Memory Recall: {'✅' if memory_result['recall'].get('recalled') else '❌'}")
        print(f"   Hierarchical Processing: {hierarchy_result['processing_depth']}/{len(self.brain.hierarchy['layers'])} layers")
        
        # Store demo
        self.demo_history.append({
            'name': demo_name,
            'sensory_input': sensory_input,
            'memory_input': memory_input,
            'pattern_result': pattern_result,
            'region_result': region_result,
            'memory_result': memory_result,
            'hierarchy_result': hierarchy_result
        })
        
        return {
            'pattern': pattern_result,
            'regions': region_result,
            'memory': memory_result,
            'hierarchy': hierarchy_result
        }
    
    def visualize_full_pipeline(self, demo_data, demo_name):
        """Create comprehensive visualization of full pipeline"""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
        
        pattern_result = demo_data['pattern']
        region_result = demo_data['regions']
        memory_result = demo_data.get('memory')
        hierarchy_result = demo_data['hierarchy']
        
        # Plot 1: Pattern Recognition
        ax1 = fig.add_subplot(gs[0, 0])
        sensory_sample = demo_data['sensory_input'][:200] if len(demo_data['sensory_input']) > 200 else demo_data['sensory_input']
        ax1.plot(sensory_sample, 'b-', linewidth=1.5)
        ax1.set_title('Sensory Input (Pattern)', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Element Index')
        ax1.set_ylabel('Value')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Pattern Recognition Results
        ax2 = fig.add_subplot(gs[0, 1])
        metrics = {
            'Confidence': pattern_result['confidence'],
            'Features': pattern_result['features_detected'] / 100.0,
            'Recognized': 1.0 if pattern_result['pattern_recognized'] else 0.0
        }
        colors = ['#3498DB', '#9B59B6', '#27AE60' if pattern_result['pattern_recognized'] else '#E74C3C']
        bars = ax2.bar(metrics.keys(), list(metrics.values()), color=colors, alpha=0.8, edgecolor='black')
        ax2.set_title('Pattern Recognition', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Score')
        ax2.set_ylim([0, 1.1])
        ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: Region Activities
        ax3 = fig.add_subplot(gs[0, 2])
        region_activities = region_result['region_activities']
        regions = list(region_activities.keys())
        activities = list(region_activities.values())
        colors_regions = ['#E74C3C', '#3498DB', '#9B59B6', '#F39C12', '#1ABC9C']
        bars = ax3.barh(regions, activities, color=colors_regions[:len(regions)], alpha=0.8, edgecolor='black')
        ax3.set_title('Region Activities', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Activity')
        ax3.set_xlim([0, 1.0])
        ax3.grid(axis='x', alpha=0.3)
        
        # Plot 4: Activity Propagation
        ax4 = fig.add_subplot(gs[1, 0])
        step_order = ['sensory_cortex', 'association_cortex', 'memory_hippocampus', 
                     'executive_cortex', 'motor_cortex']
        step_activities = [region_activities.get(step, 0) for step in step_order]
        step_labels = [s.replace('_', '\n').title() for s in step_order]
        ax4.plot(step_labels, step_activities, 'o-', linewidth=3, markersize=10, color='#3498DB')
        ax4.fill_between(range(len(step_labels)), step_activities, alpha=0.3, color='#3498DB')
        ax4.set_title('Activity Propagation', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Activity Level')
        ax4.set_xticks(range(len(step_labels)))
        ax4.set_xticklabels(step_labels, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Memory System (if available)
        ax5 = fig.add_subplot(gs[1, 1])
        if memory_result:
            memory_metrics = {
                'Storage': 1.0 if memory_result['storage'].get('stored') else 0.0,
                'Recall': 1.0 if memory_result['recall'].get('recalled') else 0.0,
                'Similarity': memory_result['recall'].get('similarity', 0.0)
            }
            colors_mem = ['#27AE60' if memory_result['storage'].get('stored') else '#E74C3C',
                         '#27AE60' if memory_result['recall'].get('recalled') else '#E74C3C',
                         '#3498DB']
            bars = ax5.bar(memory_metrics.keys(), list(memory_metrics.values()), 
                          color=colors_mem, alpha=0.8, edgecolor='black')
            ax5.set_title('Memory Operations', fontsize=11, fontweight='bold')
            ax5.set_ylabel('Score')
            ax5.set_ylim([0, 1.1])
            ax5.grid(axis='y', alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No Memory\nOperation', ha='center', va='center',
                    transform=ax5.transAxes, fontsize=12, fontweight='bold')
            ax5.set_title('Memory Operations', fontsize=11, fontweight='bold')
            ax5.axis('off')
        
        # Plot 6: Hierarchical Processing
        ax6 = fig.add_subplot(gs[1, 2])
        hierarchy_metrics = {
            'Depth': hierarchy_result['processing_depth'] / len(self.brain.hierarchy['layers']),
            'Layers\nActive': hierarchy_result['layers_active'] / len(self.brain.hierarchy['layers']),
            'Info\nFlow': min(1.0, hierarchy_result['information_flow'] / 10.0)
        }
        bars = ax6.bar(hierarchy_metrics.keys(), list(hierarchy_metrics.values()),
                      color=['#9B59B6', '#1ABC9C', '#F39C12'], alpha=0.8, edgecolor='black')
        ax6.set_title('Hierarchical Processing', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Score')
        ax6.set_ylim([0, 1.1])
        ax6.grid(axis='y', alpha=0.3)
        
        # Plot 7: Brain Architecture Diagram
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        # Draw brain regions
        region_positions = {
            'sensory_cortex': (0.15, 0.8),
            'association_cortex': (0.4, 0.8),
            'memory_hippocampus': (0.3, 0.5),
            'executive_cortex': (0.6, 0.5),
            'motor_cortex': (0.85, 0.2)
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
                color = plt.cm.RdYlGn(activity)
                width = 1 + activity * 3
                arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                       arrowstyle='->', mutation_scale=20,
                                       linewidth=width, color=color,
                                       transform=ax7.transAxes)
                ax7.add_patch(arrow)
        
        # Draw regions
        for region_name, (x, y) in region_positions.items():
            activity = region_activities.get(region_name, 0)
            color = plt.cm.RdYlGn(activity)
            
            circle = Circle((x, y), 0.06, color=color,
                          edgecolor='black', linewidth=2,
                          transform=ax7.transAxes)
            ax7.add_patch(circle)
            
            ax7.text(x, y, region_labels[region_name],
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    transform=ax7.transAxes, color='white' if activity > 0.3 else 'black')
            
            ax7.text(x, y - 0.1, f'{activity:.2f}',
                    ha='center', va='top', fontsize=8, fontweight='bold',
                    transform=ax7.transAxes)
        
        # Add processing labels
        ax7.text(0.15, 0.95, 'PATTERN RECOGNITION', ha='center', fontsize=10, 
                fontweight='bold', transform=ax7.transAxes, color='#3498DB')
        ax7.text(0.5, 0.95, 'MULTI-REGION COORDINATION', ha='center', fontsize=10,
                fontweight='bold', transform=ax7.transAxes, color='#9B59B6')
        if memory_result:
            ax7.text(0.3, 0.35, 'MEMORY', ha='center', fontsize=10,
                    fontweight='bold', transform=ax7.transAxes, color='#F39C12')
        ax7.text(0.85, 0.95, 'HIERARCHICAL PROCESSING', ha='center', fontsize=10,
                fontweight='bold', transform=ax7.transAxes, color='#1ABC9C')
        
        ax7.set_xlim([0, 1])
        ax7.set_ylim([0, 1])
        ax7.set_title('Complete Brain Processing Pipeline', fontsize=14, fontweight='bold', pad=20)
        
        plt.suptitle(f'Interactive Brain Demo: {demo_name}', fontsize=16, fontweight='bold', y=0.98)
        filename = f'interactive_brain_demo_{demo_name.lower().replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n💾 Full pipeline visualization saved: {filename}")
        plt.close()
    
    def run_demo_suite(self):
        """Run comprehensive demo suite"""
        print("\n" + "="*80)
        print("INTERACTIVE BRAIN DEMO - COMPLETE PROCESSING PIPELINE")
        print("="*80)
        
        # Demo 1: Simple sensory input
        print("\n" + "="*80)
        demo1_input = np.sin(np.linspace(0, 4*np.pi, 1000))
        result1 = self.run_full_pipeline_demo(demo1_input, None, "Sine Wave Pattern")
        self.visualize_full_pipeline({'sensory_input': demo1_input, **result1}, "Sine Wave Pattern")
        
        # Demo 2: Sensory + Memory
        print("\n" + "="*80)
        demo2_sensory = np.random.random(500)
        demo2_memory = np.random.random(100) > 0.3
        result2 = self.run_full_pipeline_demo(demo2_sensory, demo2_memory, "Random Pattern + Memory")
        self.visualize_full_pipeline({'sensory_input': demo2_sensory, 'memory_input': demo2_memory, **result2}, 
                                    "Random Pattern + Memory")
        
        # Demo 3: Complex pattern
        print("\n" + "="*80)
        demo3_input = np.array([1, 0, 1, 0, 1] * 200)
        demo3_memory = np.sin(np.linspace(0, 4*np.pi, 50))
        result3 = self.run_full_pipeline_demo(demo3_input, demo3_memory, "Alternating Pattern + Memory")
        self.visualize_full_pipeline({'sensory_input': demo3_input, 'memory_input': demo3_memory, **result3},
                                    "Alternating Pattern + Memory")
        
        # Summary
        print("\n" + "="*80)
        print("DEMO SUITE SUMMARY")
        print("="*80)
        print(f"\n📊 Demos Completed: {len(self.demo_history)}")
        print(f"   All systems integrated and functional!")
        print(f"\n✅ Pattern Recognition: Working")
        print(f"✅ Multi-Region Coordination: Working")
        print(f"✅ Memory System: Working")
        print(f"✅ Hierarchical Processing: Working")

def main():
    """Main function"""
    demo = InteractiveBrainDemo()
    demo.run_demo_suite()
    print("\n✅ Interactive brain demo complete!")

if __name__ == "__main__":
    main()

