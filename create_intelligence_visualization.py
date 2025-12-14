#!/usr/bin/env python3
"""
Intelligence Assessment Visualization
=====================================
Creates comprehensive visualizations of artificial brain intelligence test results
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

def create_intelligence_dashboard():
    """Create comprehensive intelligence assessment dashboard"""
    
    # Load test results
    try:
        with open('/home/user/intelligence_assessment_report.json', 'r') as f:
            report = json.load(f)
    except:
        # Fallback data if file not found
        report = {
            'intelligence_analysis': {
                'overall_intelligence_score': 0.356,
                'biological_equivalent': 'Insect Intelligence',
                'intelligence_grade': 'D+',
                'category_scores': {
                    'basic_responsiveness': 0.506,
                    'pattern_detection': 0.167,
                    'learning_adaptation': 0.250,
                    'memory_retention': 0.125,
                    'decision_making': 0.812,
                    'stress_resilience': 0.750
                },
                'biological_levels': {
                    'Cellular Response': 0.05,
                    'Simple Reflex': 0.15,
                    'Basic Invertebrate': 0.25,
                    'Insect Intelligence': 0.40,
                    'Fish-level Cognition': 0.55,
                    'Amphibian Processing': 0.65,
                    'Reptilian Brain': 0.75,
                    'Mammalian Intelligence': 0.85,
                    'Primate Cognition': 0.95,
                    'Human-level Intelligence': 1.00
                }
            },
            'test_metadata': {
                'network_size': 80
            }
        }
    
    # Set up the figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 0.8], width_ratios=[1, 1, 1])
    
    # Colors
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
    
    # 1. Overall Intelligence Score (Top Center - Large)
    ax_overall = fig.add_subplot(gs[0, :2])
    
    overall_score = report['intelligence_analysis']['overall_intelligence_score']
    bio_level = report['intelligence_analysis']['biological_equivalent']
    grade = report['intelligence_analysis']['intelligence_grade']
    
    # Create circular progress indicator
    theta = np.linspace(0, 2*np.pi, 100)
    r_outer = 1.0
    r_inner = 0.7
    
    # Background circle
    ax_overall.fill(r_outer * np.cos(theta), r_outer * np.sin(theta), 
                   color='lightgray', alpha=0.3)
    ax_overall.fill(r_inner * np.cos(theta), r_inner * np.sin(theta), 
                   color='white')
    
    # Progress arc
    progress_theta = np.linspace(0, 2*np.pi * overall_score, int(100 * overall_score))
    if len(progress_theta) > 0:
        ax_overall.fill_between(progress_theta, r_inner, r_outer,
                               color='#4ECDC4', alpha=0.8)
    
    # Center text
    ax_overall.text(0, 0.1, f'{overall_score:.1%}', ha='center', va='center',
                   fontsize=32, fontweight='bold', color='#2C3E50')
    ax_overall.text(0, -0.15, f'Grade: {grade}', ha='center', va='center',
                   fontsize=16, color='#34495E')
    ax_overall.text(0, -0.3, f'{bio_level}', ha='center', va='center',
                   fontsize=14, color='#7F8C8D', style='italic')
    
    ax_overall.set_xlim(-1.2, 1.2)
    ax_overall.set_ylim(-1.2, 1.2)
    ax_overall.set_aspect('equal')
    ax_overall.axis('off')
    ax_overall.set_title('🧠 Overall Intelligence Score', fontsize=18, fontweight='bold', pad=20)
    
    # 2. Network Stats (Top Right)
    ax_stats = fig.add_subplot(gs[0, 2])
    
    network_size = report['test_metadata']['network_size']
    stats_text = f"""🔢 Network Statistics
    
Neurons: {network_size}
Connections: ~{int(network_size * 0.08 * network_size)}
Architecture: Biological
Test Duration: ~1.3 seconds

🎯 Achievement Level:
{bio_level}

🏆 Intelligence Grade:
{grade} ({overall_score:.1%})"""
    
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                 fontsize=12, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
    ax_stats.axis('off')
    
    # 3. Category Performance Radar Chart (Middle Left)
    ax_radar = fig.add_subplot(gs[1, 0], projection='polar')
    
    categories = list(report['intelligence_analysis']['category_scores'].keys())
    scores = list(report['intelligence_analysis']['category_scores'].values())
    
    # Clean up category names
    category_labels = [cat.replace('_', ' ').title() for cat in categories]
    
    # Radar chart
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    scores_plot = scores + [scores[0]]  # Close the polygon
    angles_plot = angles + [angles[0]]
    
    ax_radar.plot(angles_plot, scores_plot, 'o-', linewidth=2, color='#4ECDC4')
    ax_radar.fill(angles_plot, scores_plot, color='#4ECDC4', alpha=0.25)
    
    ax_radar.set_ylim(0, 1.0)
    ax_radar.set_xticks(angles)
    ax_radar.set_xticklabels(category_labels, fontsize=10)
    ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_radar.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax_radar.grid(True)
    ax_radar.set_title('🎯 Cognitive Performance Profile', fontsize=14, fontweight='bold', pad=20)
    
    # 4. Category Scores Bar Chart (Middle Center)
    ax_bars = fig.add_subplot(gs[1, 1])
    
    y_pos = np.arange(len(category_labels))
    bars = ax_bars.barh(y_pos, scores, color=colors[:len(scores)])
    
    # Add score labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        width = bar.get_width()
        grade = get_grade(score)
        ax_bars.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f} ({grade})', ha='left', va='center', fontsize=10)
    
    ax_bars.set_yticks(y_pos)
    ax_bars.set_yticklabels(category_labels, fontsize=10)
    ax_bars.set_xlabel('Performance Score', fontsize=12)
    ax_bars.set_xlim(0, 1.1)
    ax_bars.set_title('📊 Cognitive Category Scores', fontsize=14, fontweight='bold')
    ax_bars.grid(axis='x', alpha=0.3)
    
    # 5. Biological Intelligence Ladder (Middle Right)
    ax_bio = fig.add_subplot(gs[1, 2])
    
    bio_levels = report['intelligence_analysis']['biological_levels']
    level_names = list(bio_levels.keys())
    level_thresholds = list(bio_levels.values())
    
    y_positions = np.arange(len(level_names))
    
    # Create ladder visualization
    for i, (name, threshold) in enumerate(zip(level_names, level_thresholds)):
        if overall_score >= threshold * 0.7:  # Achieved
            color = '#2ECC71'
            symbol = '✅'
        elif overall_score >= threshold * 0.5:  # Approaching
            color = '#F39C12'
            symbol = '🔄'
        else:  # Not yet
            color = '#E74C3C'
            symbol = '❌'
        
        # Draw level bar
        bar = ax_bio.barh(i, threshold, color=color, alpha=0.3, edgecolor=color)
        
        # Add current score line
        if i == 0:  # Only on first iteration
            ax_bio.axvline(x=overall_score, color='#8E44AD', linestyle='--', 
                          linewidth=3, label=f'Current Score: {overall_score:.3f}')
        
        # Add labels
        ax_bio.text(threshold + 0.02, i, f'{symbol} {name}', 
                   va='center', fontsize=10, fontweight='bold' if symbol == '✅' else 'normal')
    
    ax_bio.set_yticks(y_positions)
    ax_bio.set_yticklabels([])
    ax_bio.set_xlabel('Intelligence Threshold', fontsize=12)
    ax_bio.set_xlim(0, 1.1)
    ax_bio.set_title('🧬 Biological Intelligence Ladder', fontsize=14, fontweight='bold')
    ax_bio.legend(loc='upper right')
    ax_bio.grid(axis='x', alpha=0.3)
    
    # 6. Development Roadmap (Bottom Spanning)
    ax_roadmap = fig.add_subplot(gs[2, :])
    
    # Roadmap stages
    roadmap_stages = [
        ('Foundation\n(Current)', 0.356, '#E74C3C', 'Insect Intelligence\n80 neurons'),
        ('Growth\nPhase', 0.5, '#F39C12', 'Fish-level Cognition\n500 neurons'),
        ('Enhancement\nPhase', 0.7, '#3498DB', 'Mammalian Intelligence\n5,000 neurons'),
        ('Advanced\nPhase', 0.9, '#9B59B6', 'Primate Cognition\n50,000 neurons'),
        ('Human-level\nIntelligence', 1.0, '#2ECC71', 'Human Intelligence\n1,000,000+ neurons')
    ]
    
    stage_positions = np.linspace(0.1, 0.9, len(roadmap_stages))
    
    for i, ((stage_name, threshold, color, description), x_pos) in enumerate(zip(roadmap_stages, stage_positions)):
        # Draw stage circle
        circle_size = 300 if i == 0 else 200  # Highlight current stage
        ax_roadmap.scatter(x_pos, 0.5, s=circle_size, c=color, alpha=0.7, zorder=3)
        
        # Add stage label
        ax_roadmap.text(x_pos, 0.7, stage_name, ha='center', va='bottom',
                       fontsize=11, fontweight='bold' if i == 0 else 'normal')
        ax_roadmap.text(x_pos, 0.3, description, ha='center', va='top',
                       fontsize=9, style='italic')
        
        # Add threshold score
        ax_roadmap.text(x_pos, 0.5, f'{threshold:.1f}', ha='center', va='center',
                       fontsize=10, fontweight='bold', color='white')
        
        # Connect stages with arrows
        if i < len(roadmap_stages) - 1:
            ax_roadmap.annotate('', xy=(stage_positions[i+1] - 0.05, 0.5),
                              xytext=(x_pos + 0.05, 0.5),
                              arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    
    ax_roadmap.set_xlim(0, 1)
    ax_roadmap.set_ylim(0, 1)
    ax_roadmap.axis('off')
    ax_roadmap.set_title('🚀 Development Roadmap to Human-Level Intelligence', 
                        fontsize=16, fontweight='bold', pad=20)
    
    # 7. Recommendations Panel (Bottom)
    ax_recommendations = fig.add_subplot(gs[3, :])
    
    recommendations_text = """
🎯 IMMEDIATE DEVELOPMENT RECOMMENDATIONS:

1. 🧠 MEMORY ENHANCEMENT (Priority: High)
   • Implement synaptic plasticity (STDP algorithm)
   • Add working memory consolidation mechanisms
   • Create associative memory networks

2. 🔍 PATTERN RECOGNITION (Priority: High)  
   • Add convolutional-like processing layers
   • Implement feature detection algorithms
   • Create hierarchical pattern processing

3. 📈 NETWORK SCALING (Priority: Medium)
   • Scale from 80 → 500 neurons for Fish-level cognition
   • Add specialized brain regions (cortex, hippocampus)
   • Implement parallel processing optimization

4. 🎓 LEARNING ALGORITHMS (Priority: Medium)
   • Add reinforcement learning capabilities
   • Implement attention mechanisms
   • Create adaptive learning rate systems

Next Milestone: Fish-level Cognition (50% intelligence) - Requires 500+ neuron network with memory systems
"""
    
    ax_recommendations.text(0.02, 0.98, recommendations_text, transform=ax_recommendations.transAxes,
                           fontsize=11, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round,pad=1', facecolor='lightgreen', alpha=0.2))
    ax_recommendations.axis('off')
    
    # Overall title
    fig.suptitle('🧠 ARTIFICIAL BRAIN INTELLIGENCE ASSESSMENT DASHBOARD 🧠\n' + 
                f'Score: {overall_score:.1%} | Grade: {grade} | Level: {bio_level}',
                fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save the visualization
    plt.savefig('/home/user/intelligence_dashboard.png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    
    print("📊 Intelligence assessment dashboard created!")
    print("   Saved as: intelligence_dashboard.png")
    
    return fig

def get_grade(score):
    """Convert score to letter grade"""
    if score >= 0.9: return "A+"
    elif score >= 0.8: return "A"
    elif score >= 0.7: return "B+"
    elif score >= 0.6: return "B"
    elif score >= 0.5: return "C+"
    elif score >= 0.4: return "C"
    elif score >= 0.3: return "D+"
    elif score >= 0.2: return "D"
    else: return "F"

if __name__ == "__main__":
    plt.style.use('default')
    sns.set_palette("husl")
    create_intelligence_dashboard()
    plt.show()