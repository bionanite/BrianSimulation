#!/usr/bin/env python3
"""
Interactive Memory System Exploration Tool
Demonstrates memory storage, recall, and capacity management
"""

import numpy as np
from final_enhanced_brain import FinalEnhancedBrain
import matplotlib.pyplot as plt
import time

class MemorySystemExplorer:
    """Interactive explorer for memory system"""
    
    def __init__(self):
        self.brain = FinalEnhancedBrain(total_neurons=10000)
        self.storage_history = []
        self.recall_history = []
    
    def test_storage(self, pattern, pattern_name):
        """Test storing a pattern"""
        print(f"\n{'='*60}")
        print(f"Testing Storage: {pattern_name}")
        print(f"{'='*60}")
        
        # Get initial memory status
        status_before = self.brain.enhanced_memory_operations('capacity_status')
        
        # Attempt storage
        result = self.brain.enhanced_memory_operations('store', pattern.astype(float), debug=False)
        
        # Get memory status after
        status_after = self.brain.enhanced_memory_operations('capacity_status')
        
        # Display results
        print(f"\n📊 Pattern Analysis:")
        pattern_analysis = self.brain.enhanced_pattern_recognition(pattern.astype(float))
        print(f"   Pattern Density: {pattern_analysis['density']:.3f}")
        print(f"   Pattern Type: {'Sparse' if pattern_analysis['is_sparse'] else 'Dense'}")
        print(f"   Confidence: {pattern_analysis['confidence']:.3f}")
        print(f"   Features Detected: {pattern_analysis['features_detected']}")
        
        print(f"\n💾 Storage Results:")
        if result.get('stored', False):
            print(f"   Status: ✅ STORED")
            print(f"   Location: {result.get('location', 'unknown')}")
            print(f"   Strength: {result.get('strength', 0):.3f}")
        else:
            print(f"   Status: ❌ NOT STORED")
            print(f"   Reason: {result.get('reason', 'unknown')}")
            if 'confidence' in result:
                print(f"   Confidence: {result['confidence']:.3f}")
            if 'threshold' in result:
                print(f"   Threshold: {result['threshold']:.3f}")
        
        print(f"\n📈 Memory Status:")
        print(f"   Working Memory: {status_before['working_memory_items']} → {status_after['working_memory_items']}")
        print(f"   Long-Term Memory: {status_before['long_term_memory_items']} → {status_after['long_term_memory_items']}")
        print(f"   Capacity Used: {status_before['total_capacity_used']:.1%} → {status_after['total_capacity_used']:.1%}")
        
        # Store history
        self.storage_history.append({
            'name': pattern_name,
            'pattern': pattern,
            'result': result,
            'status_before': status_before,
            'status_after': status_after
        })
        
        return result
    
    def test_recall(self, query_pattern, query_name, original_pattern=None):
        """Test recalling a pattern"""
        print(f"\n{'='*60}")
        print(f"Testing Recall: {query_name}")
        print(f"{'='*60}")
        
        # Attempt recall
        result = self.brain.enhanced_memory_operations('recall', query_pattern.astype(float), debug=False)
        
        # Display results
        print(f"\n🔍 Recall Results:")
        if result.get('recalled', False):
            print(f"   Status: ✅ RECALLED")
            print(f"   Similarity: {result.get('similarity', 0):.3f}")
            print(f"   Source: {result.get('source', 'unknown')}")
            if result.get('memory_item'):
                mem_item = result['memory_item']
                print(f"   Stored Confidence: {mem_item.get('confidence', 0):.3f}")
                print(f"   Stored Features: {mem_item.get('features', 0)}")
        else:
            print(f"   Status: ❌ NOT RECALLED")
            print(f"   Similarity: {result.get('similarity', 0):.3f}")
            print(f"   (Similarity below threshold)")
        
        # Compare with original if provided
        if original_pattern is not None:
            query_analysis = self.brain.enhanced_pattern_recognition(query_pattern.astype(float))
            original_analysis = self.brain.enhanced_pattern_recognition(original_pattern.astype(float))
            
            print(f"\n📊 Pattern Comparison:")
            print(f"   Original Confidence: {original_analysis['confidence']:.3f}")
            print(f"   Query Confidence: {query_analysis['confidence']:.3f}")
            print(f"   Confidence Difference: {abs(original_analysis['confidence'] - query_analysis['confidence']):.3f}")
        
        # Store history
        self.recall_history.append({
            'name': query_name,
            'query_pattern': query_pattern,
            'result': result
        })
        
        return result
    
    def test_storage_and_recall(self, pattern, pattern_name, noise_level=0.1):
        """Test both storage and recall with noise"""
        print(f"\n{'='*70}")
        print(f"COMPLETE TEST: Storage + Recall ({pattern_name})")
        print(f"{'='*70}")
        
        # Store pattern
        storage_result = self.test_storage(pattern, pattern_name)
        
        if storage_result.get('stored', False):
            # Wait a bit
            time.sleep(0.1)
            
            # Create noisy version for recall
            noisy_pattern = pattern.astype(float) + np.random.normal(0, noise_level, len(pattern))
            # Normalize
            if noisy_pattern.max() > noisy_pattern.min():
                noisy_pattern = (noisy_pattern - noisy_pattern.min()) / (noisy_pattern.max() - noisy_pattern.min())
            
            # Test recall
            recall_result = self.test_recall(noisy_pattern, f"{pattern_name} (with noise)", pattern)
            
            return storage_result, recall_result
        else:
            print(f"\n⚠️  Cannot test recall - pattern was not stored")
            return storage_result, None
    
    def visualize_memory_system(self):
        """Create comprehensive memory system visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Memory System Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Storage success rate
        ax1 = axes[0, 0]
        if self.storage_history:
            storage_success = [1 if h['result'].get('stored', False) else 0 for h in self.storage_history]
            pattern_names = [h['name'] for h in self.storage_history]
            
            colors = ['#27AE60' if s else '#E74C3C' for s in storage_success]
            bars = ax1.barh(pattern_names, storage_success, color=colors, alpha=0.8, edgecolor='black')
            ax1.set_title('Storage Success Rate', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Stored (1) / Not Stored (0)')
            ax1.set_xlim([0, 1.2])
            ax1.grid(axis='x', alpha=0.3)
            
            # Add labels
            for bar, success in zip(bars, storage_success):
                label = '✓' if success else '✗'
                ax1.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2.,
                        label, ha='left', va='center', fontsize=14, fontweight='bold')
        
        # Plot 2: Memory capacity over time
        ax2 = axes[0, 1]
        if self.storage_history:
            wm_counts = [h['status_after']['working_memory_items'] for h in self.storage_history]
            ltm_counts = [h['status_after']['long_term_memory_items'] for h in self.storage_history]
            
            x = range(len(self.storage_history))
            ax2.plot(x, wm_counts, 'o-', label='Working Memory', linewidth=2, markersize=8, color='#3498DB')
            ax2.plot(x, ltm_counts, 's-', label='Long-Term Memory', linewidth=2, markersize=8, color='#9B59B6')
            ax2.axhline(y=7, color='r', linestyle='--', linewidth=2, label='WM Capacity (7)')
            ax2.set_title('Memory Capacity Over Time', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Storage Attempt')
            ax2.set_ylabel('Items')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Recall similarity scores
        ax3 = axes[1, 0]
        if self.recall_history:
            similarities = [h['result'].get('similarity', 0) for h in self.recall_history]
            recalled = [h['result'].get('recalled', False) for h in self.recall_history]
            query_names = [h['name'] for h in self.recall_history]
            
            colors = ['#27AE60' if r else '#E74C3C' for r in recalled]
            bars = ax3.barh(query_names, similarities, color=colors, alpha=0.8, edgecolor='black')
            ax3.axvline(x=0.5, color='orange', linestyle='--', linewidth=2, label='Recall Threshold (0.5)')
            ax3.set_title('Recall Similarity Scores', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Similarity Score')
            ax3.set_xlim([0, 1.1])
            ax3.legend()
            ax3.grid(axis='x', alpha=0.3)
            
            # Add labels
            for bar, sim, rec in zip(bars, similarities, recalled):
                label = f'{sim:.3f} {"✓" if rec else "✗"}'
                ax3.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2.,
                        label, ha='left', va='center', fontsize=9, fontweight='bold')
        
        # Plot 4: Memory system architecture
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Draw memory stores
        status = self.brain.enhanced_memory_operations('capacity_status')
        
        # Working Memory
        wm_box = plt.Rectangle((0.1, 0.5), 0.3, 0.3, 
                              facecolor='#3498DB', edgecolor='black', linewidth=2)
        ax4.add_patch(wm_box)
        ax4.text(0.25, 0.65, 'Working\nMemory', ha='center', va='center',
                fontsize=12, fontweight='bold', color='white', transform=ax4.transAxes)
        ax4.text(0.25, 0.55, f'{status["working_memory_items"]}/7', ha='center', va='center',
                fontsize=14, fontweight='bold', color='white', transform=ax4.transAxes)
        
        # Long-Term Memory
        ltm_box = plt.Rectangle((0.6, 0.5), 0.3, 0.3,
                               facecolor='#9B59B6', edgecolor='black', linewidth=2)
        ax4.add_patch(ltm_box)
        ax4.text(0.75, 0.65, 'Long-Term\nMemory', ha='center', va='center',
                fontsize=12, fontweight='bold', color='white', transform=ax4.transAxes)
        ax4.text(0.75, 0.55, f'{status["long_term_memory_items"]}', ha='center', va='center',
                fontsize=14, fontweight='bold', color='white', transform=ax4.transAxes)
        
        # Arrows
        ax4.arrow(0.4, 0.65, 0.15, 0, head_width=0.03, head_length=0.03,
                 fc='black', ec='black', transform=ax4.transAxes, linewidth=2)
        ax4.text(0.475, 0.7, 'Consolidation', ha='center', fontsize=9,
                transform=ax4.transAxes, fontweight='bold')
        
        # Storage/Recall labels
        ax4.text(0.25, 0.4, 'STORAGE', ha='center', fontsize=10, fontweight='bold',
                transform=ax4.transAxes, color='#27AE60')
        ax4.text(0.75, 0.4, 'RECALL', ha='center', fontsize=10, fontweight='bold',
                transform=ax4.transAxes, color='#E74C3C')
        
        ax4.set_xlim([0, 1])
        ax4.set_ylim([0, 1])
        ax4.set_title('Memory System Architecture', fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig('memory_system_exploration.png', dpi=300, bbox_inches='tight')
        print(f"\n💾 Visualization saved: memory_system_exploration.png")
        plt.close()
    
    def run_comprehensive_test(self):
        """Run comprehensive memory system test"""
        print("\n" + "="*70)
        print("MEMORY SYSTEM COMPREHENSIVE TEST SUITE")
        print("="*70)
        
        # Test patterns
        test_patterns = [
            (np.random.random(50) > 0.3, "Dense Random Pattern"),
            (np.sin(np.linspace(0, 4*np.pi, 50)), "Sine Wave Pattern"),
            (np.random.random(50) > 0.7, "Sparse Random Pattern"),
            (np.array([1, 0, 1, 0, 1] * 10), "Alternating Pattern"),
            (np.random.random(50) * 0.5 + 0.5, "Uniform Random Pattern"),
            (np.cos(np.linspace(0, 8*np.pi, 50)), "Cosine Pattern"),
            (np.tanh(np.linspace(-2, 2, 50)), "Tanh Pattern"),
            (np.array([1, 1, 1, 0, 0, 0] * 8), "Block Pattern")
        ]
        
        # Test storage and recall for each
        storage_results = []
        recall_results = []
        
        for pattern, name in test_patterns:
            storage_result, recall_result = self.test_storage_and_recall(pattern, name, noise_level=0.1)
            storage_results.append(storage_result)
            if recall_result:
                recall_results.append(recall_result)
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)
        
        storage_success = sum(1 for r in storage_results if r.get('stored', False))
        recall_success = sum(1 for r in recall_results if r.get('recalled', False)) if recall_results else 0
        
        print(f"\n📈 Overall Performance:")
        print(f"   Patterns Tested: {len(test_patterns)}")
        print(f"   Storage Success: {storage_success}/{len(test_patterns)} ({storage_success/len(test_patterns)*100:.1f}%)")
        print(f"   Recall Success: {recall_success}/{len(recall_results)} ({recall_success/len(recall_results)*100:.1f}%)" if recall_results else "   Recall Success: N/A (no stored patterns)")
        
        if recall_results:
            avg_similarity = np.mean([r.get('similarity', 0) for r in recall_results])
            print(f"   Average Similarity: {avg_similarity:.3f}")
        
        # Final memory status
        final_status = self.brain.enhanced_memory_operations('capacity_status')
        print(f"\n💾 Final Memory Status:")
        print(f"   Working Memory: {final_status['working_memory_items']}/7 items")
        print(f"   Long-Term Memory: {final_status['long_term_memory_items']} items")
        print(f"   Capacity Used: {final_status['total_capacity_used']:.1%}")
        print(f"   Avg Synaptic Strength: {final_status['synaptic_strength_avg']:.3f}")
        
        # Create visualization
        self.visualize_memory_system()

def main():
    """Main function"""
    explorer = MemorySystemExplorer()
    explorer.run_comprehensive_test()
    print("\n✅ Memory system exploration complete!")

if __name__ == "__main__":
    main()

