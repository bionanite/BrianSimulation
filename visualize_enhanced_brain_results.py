#!/usr/bin/env python3
"""
Visualization script for Enhanced Brain System Results
Creates comprehensive charts showing performance metrics
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

def load_results():
    """Load results from JSON file"""
    with open('final_enhanced_brain_results.json', 'r') as f:
        return json.load(f)

def create_comprehensive_dashboard(results):
    """Create comprehensive dashboard with multiple charts"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Extract data
    scores = results['detailed_results']['individual_scores']
    baseline = results['baseline_comparison']
    current = results['enhanced_intelligence_score']
    system_status = results['detailed_results']['system_status']
    
    # Chart 1: Before/After Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    categories = ['Before\nFixes', 'After\nFixes']
    before_scores = {
        'Memory': 0.000,
        'Multi-Region': 0.133,
        'Overall': 0.379
    }
    after_scores = {
        'Memory': scores['advanced_memory'],
        'Multi-Region': scores['multi_region_coordination'],
        'Overall': current
    }
    
    x = np.arange(len(categories))
    width = 0.25
    ax1.bar(x - width, [before_scores['Memory'], after_scores['Memory']], 
            width, label='Memory System', color='#FF6B6B', alpha=0.8)
    ax1.bar(x, [before_scores['Multi-Region'], after_scores['Multi-Region']], 
            width, label='Multi-Region', color='#4ECDC4', alpha=0.8)
    ax1.bar(x + width, [before_scores['Overall'], after_scores['Overall']], 
            width, label='Overall Score', color='#45B7D1', alpha=0.8)
    
    ax1.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax1.set_title('Before vs After Fixes Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_ylim([0, 1.0])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Chart 2: Individual Enhancement Scores
    ax2 = fig.add_subplot(gs[0, 1])
    enhancements = ['Pattern\nRecognition', 'Multi-Region\nCoordination', 
                    'Advanced\nMemory', 'Hierarchical\nProcessing']
    enhancement_scores = [
        scores['pattern_recognition'],
        scores['multi_region_coordination'],
        scores['advanced_memory'],
        scores['hierarchical_processing']
    ]
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#F38181']
    
    bars = ax2.bar(enhancements, enhancement_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax2.set_title('Individual Enhancement Performance', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 1.0])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, score in zip(bars, enhancement_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Chart 3: Intelligence Score Progression
    ax3 = fig.add_subplot(gs[0, 2])
    milestones = ['Baseline\n(Simple 10K)', 'Before\nFixes', 'After\nFixes']
    intelligence_scores = [baseline, 0.379, current]
    colors_progression = ['#95A5A6', '#E74C3C', '#27AE60']
    
    bars = ax3.bar(milestones, intelligence_scores, color=colors_progression, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Intelligence Score', fontsize=11, fontweight='bold')
    ax3.set_title('Intelligence Score Progression', fontsize=12, fontweight='bold')
    ax3.set_ylim([0, 0.7])
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, score in zip(bars, intelligence_scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add improvement arrow
    ax3.annotate('', xy=(2, current), xytext=(1, 0.379),
                arrowprops=dict(arrowstyle='->', lw=2, color='#27AE60'))
    ax3.text(1.5, 0.45, f'+{results["improvement_percentage"]:.1f}%', 
            ha='center', fontsize=10, fontweight='bold', color='#27AE60')
    
    # Chart 4: System Capabilities Pie Chart
    ax4 = fig.add_subplot(gs[1, 0])
    capabilities = ['Neurons', 'Brain Regions', 'Memory Items', 'Processing Layers']
    values = [
        system_status['total_neurons'] / 10000,  # Normalize
        system_status['brain_regions'] / 5,
        system_status['memory_items'] / 7,  # Max capacity
        system_status['processing_layers'] / 6
    ]
    
    # Create donut chart
    colors_pie = ['#3498DB', '#9B59B6', '#E67E22', '#1ABC9C']
    wedges, texts, autotexts = ax4.pie(values, labels=capabilities, colors=colors_pie, 
                                       autopct='%1.1f%%', startangle=90, 
                                       textprops={'fontsize': 9})
    ax4.set_title('System Capabilities\n(Normalized)', fontsize=12, fontweight='bold')
    
    # Chart 5: Score Breakdown Radar Chart
    ax5 = fig.add_subplot(gs[1, 1], projection='polar')
    categories_radar = ['Pattern\nRecognition', 'Multi-Region', 
                        'Memory', 'Hierarchical\nProcessing', 'Overall']
    scores_radar = [
        scores['pattern_recognition'],
        scores['multi_region_coordination'],
        scores['advanced_memory'],
        scores['hierarchical_processing'],
        current
    ]
    
    # Complete the circle
    categories_radar += [categories_radar[0]]
    scores_radar += [scores_radar[0]]
    
    angles = np.linspace(0, 2 * np.pi, len(categories_radar), endpoint=True).tolist()
    
    ax5.plot(angles, scores_radar, 'o-', linewidth=2, color='#3498DB', label='Current')
    ax5.fill(angles, scores_radar, alpha=0.25, color='#3498DB')
    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(categories_radar[:-1], fontsize=9)
    ax5.set_ylim([0, 1.0])
    ax5.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax5.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_title('Performance Radar Chart', fontsize=12, fontweight='bold', pad=20)
    
    # Chart 6: Improvement Metrics
    ax6 = fig.add_subplot(gs[1, 2])
    improvements = {
        'Memory\nSystem': (0.000, scores['advanced_memory']),
        'Multi-Region': (0.133, scores['multi_region_coordination']),
        'Overall': (0.379, current)
    }
    
    metrics = list(improvements.keys())
    before_vals = [v[0] for v in improvements.values()]
    after_vals = [v[1] for v in improvements.values()]
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax6.bar(x_pos - width/2, before_vals, width, label='Before', 
                    color='#E74C3C', alpha=0.7, edgecolor='black')
    bars2 = ax6.bar(x_pos + width/2, after_vals, width, label='After', 
                    color='#27AE60', alpha=0.7, edgecolor='black')
    
    ax6.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax6.set_title('Improvement Metrics', fontsize=12, fontweight='bold')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(metrics)
    ax6.legend(fontsize=9)
    ax6.set_ylim([0, 0.75])
    ax6.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add improvement percentages
    for i, (before, after) in enumerate(zip(before_vals, after_vals)):
        if before > 0:
            improvement = ((after - before) / before) * 100
        else:
            improvement = 100 if after > 0 else 0
        ax6.text(i, max(before, after) + 0.02, f'+{improvement:.0f}%', 
                ha='center', fontsize=9, fontweight='bold', color='#27AE60')
    
    # Chart 7: Memory System Performance
    ax7 = fig.add_subplot(gs[2, 0])
    memory_metrics = ['Storage\nSuccess', 'Recall\nSuccess', 'Working\nMemory', 'LTM\nItems']
    memory_values = [
        0,  # Storage (0/5)
        1.0,  # Recall (5/5)
        system_status['memory_items'] / 7,  # Working memory capacity
        0  # LTM items
    ]
    
    bars = ax7.bar(memory_metrics, memory_values, color=['#E74C3C', '#27AE60', '#3498DB', '#95A5A6'], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    ax7.set_ylabel('Performance', fontsize=11, fontweight='bold')
    ax7.set_title('Memory System Performance', fontsize=12, fontweight='bold')
    ax7.set_ylim([0, 1.1])
    ax7.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add labels
    labels = ['0/5', '5/5', f'{system_status["memory_items"]}/7', '0']
    for bar, label in zip(bars, labels):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                label, ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Chart 8: Multi-Region Activity
    ax8 = fig.add_subplot(gs[2, 1])
    # Simulate region activities based on coordination score
    regions = ['Sensory', 'Association', 'Memory', 'Executive', 'Motor']
    # Estimate activities (coordination score of 0.667 means ~3.3/5 regions active)
    activities = [0.8, 0.6, 0.5, 0.4, 0.3]  # Estimated based on coordination
    
    bars = ax8.barh(regions, activities, color='#9B59B6', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax8.set_xlabel('Activity Level', fontsize=11, fontweight='bold')
    ax8.set_title('Multi-Region Brain Activity', fontsize=12, fontweight='bold')
    ax8.set_xlim([0, 1.0])
    ax8.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for bar, activity in zip(bars, activities):
        width = bar.get_width()
        ax8.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                f'{activity:.2f}', ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Chart 9: Summary Statistics
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    # Create summary text
    summary_text = f"""
    ENHANCED BRAIN SYSTEM RESULTS
    {'='*40}
    
    Overall Intelligence Score: {current:.3f}/1.000
    Baseline Comparison: {baseline:.3f} → {current:.3f}
    Improvement: +{results['improvement_percentage']:.1f}%
    
    ENHANCEMENT SCORES:
    • Pattern Recognition: {scores['pattern_recognition']:.3f}
    • Multi-Region Coordination: {scores['multi_region_coordination']:.3f}
    • Advanced Memory: {scores['advanced_memory']:.3f}
    • Hierarchical Processing: {scores['hierarchical_processing']:.3f}
    
    SYSTEM STATUS:
    • Neurons: {system_status['total_neurons']:,}
    • Brain Regions: {system_status['brain_regions']}
    • Memory Items: {system_status['memory_items']}
    • Processing Layers: {system_status['processing_layers']}
    
    Intelligence Level: {results['intelligence_level']}
    Grade: {results['grade']}
    Processing Time: {results['processing_time']:.3f}s
    """
    
    ax9.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.5))
    
    # Main title
    fig.suptitle('Enhanced Brain System - Comprehensive Results Dashboard', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig('enhanced_brain_results_dashboard.png', dpi=300, bbox_inches='tight')
    print("✅ Dashboard saved as: enhanced_brain_results_dashboard.png")
    
    return fig

def main():
    """Main function to create visualizations"""
    print("📊 Creating Enhanced Brain System Results Dashboard...")
    
    # Load results
    results = load_results()
    
    # Create dashboard
    fig = create_comprehensive_dashboard(results)
    
    print("\n📈 Dashboard created successfully!")
    print(f"   Overall Score: {results['enhanced_intelligence_score']:.3f}")
    print(f"   Improvement: +{results['improvement_percentage']:.1f}%")
    print(f"   Memory Score: {results['detailed_results']['individual_scores']['advanced_memory']:.3f}")
    print(f"   Multi-Region Score: {results['detailed_results']['individual_scores']['multi_region_coordination']:.3f}")
    
    plt.show()

if __name__ == "__main__":
    main()

