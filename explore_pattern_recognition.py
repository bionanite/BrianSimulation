#!/usr/bin/env python3
"""
Interactive Pattern Recognition Exploration Tool
Demonstrates how the pattern recognition system works with different pattern types
"""

import numpy as np
from final_enhanced_brain import FinalEnhancedBrain
import matplotlib.pyplot as plt

class PatternRecognitionExplorer:
    """Interactive explorer for pattern recognition system"""
    
    def __init__(self):
        self.brain = FinalEnhancedBrain(total_neurons=10000)
        self.results = []
    
    def test_pattern(self, pattern, pattern_name, visualize=True):
        """Test a pattern and show results"""
        print(f"\n{'='*60}")
        print(f"Testing Pattern: {pattern_name}")
        print(f"{'='*60}")
        
        # Analyze pattern
        result = self.brain.enhanced_pattern_recognition(pattern.astype(float))
        
        # Display results
        print(f"\n📊 Pattern Analysis:")
        print(f"   Pattern Length: {len(pattern)}")
        print(f"   Pattern Density: {result['density']:.3f}")
        print(f"   Pattern Type: {'Sparse' if result['is_sparse'] else 'Dense'}")
        print(f"   Features Detected: {result['features_detected']}")
        
        print(f"\n🎯 Recognition Results:")
        print(f"   Recognition Score: {result['recognition_score']:.3f}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Pattern Recognized: {'✅ YES' if result['pattern_recognized'] else '❌ NO'}")
        
        # Store result
        self.results.append({
            'name': pattern_name,
            'pattern': pattern,
            'result': result
        })
        
        # Visualize if requested
        if visualize:
            self.visualize_pattern(pattern, pattern_name, result)
        
        return result
    
    def visualize_pattern(self, pattern, pattern_name, result):
        """Create visualization for pattern"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Pattern Recognition: {pattern_name}', fontsize=16, fontweight='bold')
        
        # Plot 1: Original pattern
        ax1 = axes[0, 0]
        sample_size = min(200, len(pattern))
        ax1.plot(pattern[:sample_size], 'b-', linewidth=1.5)
        ax1.set_title('Input Pattern (Sample)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Element Index')
        ax1.set_ylabel('Value')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Pattern statistics
        ax2 = axes[0, 1]
        stats = {
            'Mean': np.mean(pattern),
            'Std Dev': np.std(pattern),
            'Min': np.min(pattern),
            'Max': np.max(pattern),
            'Density': result['density']
        }
        bars = ax2.bar(stats.keys(), list(stats.values()), 
                      color=['#3498DB', '#9B59B6', '#1ABC9C', '#E74C3C', '#F39C12'],
                      alpha=0.7, edgecolor='black')
        ax2.set_title('Pattern Statistics', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Value')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: Recognition metrics
        ax3 = axes[1, 0]
        metrics = {
            'Recognition\nScore': result['recognition_score'],
            'Confidence': result['confidence'],
            'Features': result['features_detected'] / 100.0  # Normalize
        }
        colors = ['#27AE60' if result['pattern_recognized'] else '#E74C3C', 
                 '#3498DB', '#F39C12']
        bars = ax3.bar(metrics.keys(), list(metrics.values()), 
                      color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax3.set_title('Recognition Metrics', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Score')
        ax3.set_ylim([0, 1.1])
        ax3.grid(axis='y', alpha=0.3)
        
        # Add threshold line
        ax3.axhline(y=0.7, color='r', linestyle='--', linewidth=2, label='Threshold (0.7)')
        ax3.legend()
        
        # Plot 4: Processing path
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Draw processing steps
        steps = [
            ('Input', result['density']),
            ('Feature\nExtraction', result['features_detected'] / 100.0),
            ('Pattern\nIntegration', result['recognition_score']),
            ('Confidence\nCalc', result['confidence']),
            ('Recognition', 1.0 if result['pattern_recognized'] else 0.0)
        ]
        
        x_pos = np.linspace(0.1, 0.9, len(steps))
        for i, (step_name, value) in enumerate(steps):
            color = plt.cm.RdYlGn(value)
            circle = plt.Circle((x_pos[i], 0.5), 0.08, color=color,
                              edgecolor='black', linewidth=2, transform=ax4.transAxes)
            ax4.add_patch(circle)
            ax4.text(x_pos[i], 0.5, step_name, ha='center', va='center',
                    fontsize=9, fontweight='bold', transform=ax4.transAxes,
                    color='white' if value > 0.5 else 'black')
            ax4.text(x_pos[i], 0.35, f'{value:.2f}', ha='center', va='top',
                    fontsize=8, transform=ax4.transAxes)
            
            if i < len(steps) - 1:
                ax4.plot([x_pos[i] + 0.08, x_pos[i+1] - 0.08], [0.5, 0.5],
                        'k-', linewidth=2, transform=ax4.transAxes)
                ax4.plot(x_pos[i+1] - 0.08, 0.5, 'k>', markersize=10,
                        transform=ax4.transAxes)
        
        ax4.set_xlim([0, 1])
        ax4.set_ylim([0, 1])
        ax4.set_title('Processing Pipeline', fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        filename = f'pattern_exploration_{pattern_name.lower().replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n💾 Visualization saved: {filename}")
        plt.close()
    
    def run_comprehensive_test(self):
        """Run comprehensive test suite"""
        print("\n" + "="*60)
        print("PATTERN RECOGNITION COMPREHENSIVE TEST SUITE")
        print("="*60)
        
        # Define test patterns
        test_patterns = [
            (np.sin(np.linspace(0, 4*np.pi, 1000)), "Sine Wave Pattern"),
            (np.array([1, 0, 1, 0, 1] * 200), "Alternating Pattern"),
            (np.random.random(1000) > 0.7, "Sparse Random Pattern"),
            (np.random.random(1000), "Dense Random Pattern"),
            (np.array([1, 1, 1, 0, 0, 0] * 167), "Block Pattern"),
            (np.cos(np.linspace(0, 8*np.pi, 1000)), "Cosine Wave Pattern"),
            (np.random.random(1000) * 0.5 + 0.5, "Uniform Random Pattern"),
            (np.tanh(np.linspace(-3, 3, 1000)), "Tanh Pattern")
        ]
        
        # Test each pattern
        for pattern, name in test_patterns:
            self.test_pattern(pattern, name, visualize=True)
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        recognized_count = sum(1 for r in self.results if r['result']['pattern_recognized'])
        avg_confidence = np.mean([r['result']['confidence'] for r in self.results])
        avg_features = np.mean([r['result']['features_detected'] for r in self.results])
        
        print(f"\n📈 Overall Performance:")
        print(f"   Patterns Tested: {len(self.results)}")
        print(f"   Patterns Recognized: {recognized_count} ({recognized_count/len(self.results)*100:.1f}%)")
        print(f"   Average Confidence: {avg_confidence:.3f}")
        print(f"   Average Features Detected: {avg_features:.1f}")
        
        print(f"\n🎯 Recognition by Pattern Type:")
        sparse_results = [r for r in self.results if r['result']['is_sparse']]
        dense_results = [r for r in self.results if not r['result']['is_sparse']]
        
        if sparse_results:
            sparse_recognized = sum(1 for r in sparse_results if r['result']['pattern_recognized'])
            print(f"   Sparse Patterns: {sparse_recognized}/{len(sparse_results)} recognized")
        
        if dense_results:
            dense_recognized = sum(1 for r in dense_results if r['result']['pattern_recognized'])
            print(f"   Dense Patterns: {dense_recognized}/{len(dense_results)} recognized")
        
        # Create comparison chart
        self.create_comparison_chart()
    
    def create_comparison_chart(self):
        """Create comparison chart of all patterns"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        pattern_names = [r['name'] for r in self.results]
        confidences = [r['result']['confidence'] for r in self.results]
        recognized = [r['result']['pattern_recognized'] for r in self.results]
        
        colors = ['#27AE60' if r else '#E74C3C' for r in recognized]
        
        bars = ax.barh(pattern_names, confidences, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax.axvline(x=0.7, color='orange', linestyle='--', linewidth=2, label='Recognition Threshold (0.7)')
        
        ax.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
        ax.set_title('Pattern Recognition Comparison', fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1.0])
        ax.grid(axis='x', alpha=0.3)
        ax.legend()
        
        # Add value labels
        for bar, conf, rec in zip(bars, confidences, recognized):
            width = bar.get_width()
            label = f'{conf:.3f} {"✓" if rec else "✗"}'
            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                   label, ha='left', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('pattern_recognition_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n💾 Comparison chart saved: pattern_recognition_comparison.png")
        plt.close()

def main():
    """Main function"""
    explorer = PatternRecognitionExplorer()
    explorer.run_comprehensive_test()
    print("\n✅ Pattern recognition exploration complete!")

if __name__ == "__main__":
    main()

