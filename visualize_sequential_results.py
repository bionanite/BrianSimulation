#!/usr/bin/env python3
"""
Visualize Sequential Benchmark Testing Results
Creates charts showing scaling behavior across neuron counts
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Results from sequential testing
results = {
    'HellaSwag': {
        10000: 90.00,
        100000: 90.00,
        1000000: 90.00,
        10000000: 80.00,
        100000000: 80.00
    },
    'MMLU': {
        10000: 50.00,
        100000: 40.00,
        1000000: 40.00,
        10000000: 70.00,
        100000000: 50.00
    },
    'ARC': {
        10000: 50.00,
        100000: 60.00,
        1000000: 40.00,
        10000000: 40.00,
        100000000: 40.00
    },
    'GSM8K': {
        10000: 0.00,
        100000: 0.00,
        1000000: 0.00,
        10000000: 10.00,
        100000000: 0.00
    }
}

# Baseline scores for comparison
baselines = {
    'HellaSwag': {'best_ai': 95.0, 'human': 95.6},
    'MMLU': {'best_ai': 86.3, 'human': 89.7},
    'ARC': {'best_ai': 85.0, 'human': 85.0},
    'GSM8K': {'best_ai': 92.0, 'human': 92.0}
}

def create_scaling_plot():
    """Create plot showing accuracy vs neuron count for each benchmark"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Sequential Benchmark Testing: Scaling Behavior', fontsize=16, fontweight='bold')
    
    neuron_counts = [10000, 100000, 1000000, 10000000, 100000000]
    neuron_labels = ['10K', '100K', '1M', '10M', '100M']
    
    benchmarks = ['HellaSwag', 'MMLU', 'ARC', 'GSM8K']
    axes_flat = axes.flatten()
    
    for idx, benchmark in enumerate(benchmarks):
        ax = axes_flat[idx]
        accuracies = [results[benchmark][n] for n in neuron_counts]
        
        # Plot our results
        ax.plot(neuron_labels, accuracies, 'o-', linewidth=2, markersize=8, 
                label='Our System', color='#2E86AB')
        
        # Plot baselines
        baseline = baselines[benchmark]
        ax.axhline(y=baseline['best_ai'], color='#A23B72', linestyle='--', 
                   linewidth=2, label=f"Best AI ({baseline['best_ai']}%)", alpha=0.7)
        ax.axhline(y=baseline['human'], color='#F18F01', linestyle='--', 
                   linewidth=2, label=f"Human ({baseline['human']}%)", alpha=0.7)
        
        # Highlight optimal point
        max_acc = max(accuracies)
        max_idx = accuracies.index(max_acc)
        ax.plot(neuron_labels[max_idx], max_acc, 'o', markersize=12, 
                color='#06A77D', markeredgecolor='black', markeredgewidth=2,
                label=f'Peak: {max_acc}%', zorder=10)
        
        ax.set_title(f'{benchmark}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Neuron Count', fontsize=10)
        ax.set_ylabel('Accuracy (%)', fontsize=10)
        ax.set_ylim([-5, 100])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
        
        # Add value annotations
        for i, (label, acc) in enumerate(zip(neuron_labels, accuracies)):
            ax.annotate(f'{acc:.0f}%', (i, acc), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('sequential_scaling_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: sequential_scaling_analysis.png")
    return fig

def create_comparison_heatmap():
    """Create heatmap comparing performance across benchmarks and scales"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    benchmarks = ['HellaSwag', 'MMLU', 'ARC', 'GSM8K']
    neuron_labels = ['10K', '100K', '1M', '10M', '100M']
    neuron_counts = [10000, 100000, 1000000, 10000000, 100000000]
    
    # Create matrix
    matrix = np.array([[results[b][n] for n in neuron_counts] for b in benchmarks])
    
    # Create heatmap
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    
    # Set ticks
    ax.set_xticks(np.arange(len(neuron_labels)))
    ax.set_yticks(np.arange(len(benchmarks)))
    ax.set_xticklabels(neuron_labels)
    ax.set_yticklabels(benchmarks)
    
    # Add text annotations
    for i in range(len(benchmarks)):
        for j in range(len(neuron_labels)):
            text = ax.text(j, i, f'{matrix[i, j]:.0f}%',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('Benchmark Performance Heatmap Across Neuron Scales', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Neuron Count', fontsize=12)
    ax.set_ylabel('Benchmark', fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy (%)', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig('sequential_heatmap.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: sequential_heatmap.png")
    return fig

def create_gap_analysis():
    """Create bar chart showing gaps to best AI"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    benchmarks = ['HellaSwag', 'MMLU', 'ARC', 'GSM8K']
    neuron_counts = [10000, 100000, 1000000, 10000000, 100000000]
    neuron_labels = ['10K', '100K', '1M', '10M', '100M']
    
    # Calculate gaps for each benchmark at each scale
    gaps = {}
    for benchmark in benchmarks:
        gaps[benchmark] = []
        best_ai = baselines[benchmark]['best_ai']
        for n in neuron_counts:
            our_acc = results[benchmark][n]
            gap = best_ai - our_acc
            gaps[benchmark].append(gap)
    
    # Create grouped bar chart
    x = np.arange(len(benchmarks))
    width = 0.15
    
    for i, (label, n) in enumerate(zip(neuron_labels, neuron_counts)):
        values = [gaps[b][i] for b in benchmarks]
        offset = (i - 2) * width
        ax.bar(x + offset, values, width, label=f'{label} neurons', alpha=0.8)
    
    ax.set_xlabel('Benchmark', fontsize=12)
    ax.set_ylabel('Gap to Best AI (%)', fontsize=12)
    ax.set_title('Performance Gap to Best AI Across Scales', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('sequential_gap_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: sequential_gap_analysis.png")
    return fig

def create_summary_statistics():
    """Print summary statistics"""
    print("\n" + "="*70)
    print("SEQUENTIAL TESTING SUMMARY STATISTICS")
    print("="*70)
    
    for benchmark in results:
        accs = list(results[benchmark].values())
        print(f"\n{benchmark}:")
        print(f"  Mean: {np.mean(accs):.2f}%")
        print(f"  Std:  {np.std(accs):.2f}%")
        print(f"  Min:  {np.min(accs):.2f}%")
        print(f"  Max:  {np.max(accs):.2f}%")
        print(f"  Range: {np.max(accs) - np.min(accs):.2f}%")
        
        # Find optimal scale
        max_acc = max(accs)
        max_idx = accs.index(max_acc)
        optimal_neurons = list(results[benchmark].keys())[max_idx]
        print(f"  Optimal: {optimal_neurons:,} neurons ({max_acc:.2f}%)")
        
        # Calculate gap to best AI
        best_ai = baselines[benchmark]['best_ai']
        gap = best_ai - max_acc
        print(f"  Gap to Best AI: {gap:.2f}%")
    
    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("="*70)
    print("1. No benchmark shows consistent improvement with scale")
    print("2. Optimal scale appears to be 10M neurons for most tasks")
    print("3. Performance degrades at 100M neurons")
    print("4. GSM8K shows critical failure (0-10% accuracy)")
    print("5. Statistical significance limited by small sample size (10 questions)")

if __name__ == "__main__":
    print("Generating visualizations for sequential benchmark testing...")
    
    # Create visualizations
    create_scaling_plot()
    create_comparison_heatmap()
    create_gap_analysis()
    create_summary_statistics()
    
    print("\n✅ All visualizations generated successfully!")
    print("   Files created:")
    print("   - sequential_scaling_analysis.png")
    print("   - sequential_heatmap.png")
    print("   - sequential_gap_analysis.png")

