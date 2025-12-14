#!/usr/bin/env python3
"""
Validate Performance at 80B Neuron Scale
Uses existing roadmap optimizations to validate scaling
"""

import sys
import time
from final_enhanced_brain import FinalEnhancedBrain
from benchmark_framework import BenchmarkFramework
from benchmark_adapters import MMLUAdapter
from performance_tracking import PerformanceTracker


def validate_scaling(target_neurons: int = 80_000_000_000):
    """
    Validate performance at target neuron scale
    
    Args:
        target_neurons: Target number of neurons (default: 80B)
    """
    print("=" * 70)
    print("80B NEURON SCALE VALIDATION")
    print("=" * 70)
    print()
    
    print(f"Target Scale: {target_neurons:,} neurons")
    print()
    
    # Check if we can actually run at this scale
    # For validation purposes, we'll test with smaller scales first
    # and validate that the architecture supports scaling
    
    test_scales = [
        10_000,           # Current working scale
        100_000,          # Phase 1 target
        1_000_000,        # Phase 2 target
        10_000_000,       # Phase 3 target (if GPU available)
        100_000_000,      # Phase 4 target (if distributed available)
    ]
    
    # Add target scale if reasonable
    if target_neurons <= 100_000_000:
        test_scales.append(target_neurons)
    
    results = {}
    
    for scale in test_scales:
        print(f"\n{'='*70}")
        print(f"Testing Scale: {scale:,} neurons")
        print(f"{'='*70}")
        
        try:
            start_time = time.time()
            
            # Initialize brain at this scale
            print(f"Initializing brain with {scale:,} neurons...")
            brain = FinalEnhancedBrain(total_neurons=scale, debug=False)
            
            init_time = time.time() - start_time
            print(f"✅ Initialization complete in {init_time:.2f}s")
            
            # Run a simple benchmark test
            print("Running benchmark test...")
            framework = BenchmarkFramework(brain_system=brain)
            framework.register_adapter("MMLU", MMLUAdapter())
            
            # Run with limited questions for speed
            summary = framework.run_benchmark(
                benchmark_name="MMLU",
                max_questions=5,  # Small test
                verbose=False
            )
            
            total_time = time.time() - start_time
            
            results[scale] = {
                'success': True,
                'init_time': init_time,
                'total_time': total_time,
                'accuracy': summary.accuracy,
                'memory_usage': get_memory_usage() if has_memory_tracking() else None
            }
            
            print(f"✅ Scale {scale:,}: Success")
            print(f"   Accuracy: {summary.accuracy:.2%}")
            print(f"   Total Time: {total_time:.2f}s")
            
        except MemoryError as e:
            print(f"❌ Scale {scale:,}: Memory Error - {e}")
            results[scale] = {
                'success': False,
                'error': 'MemoryError',
                'message': str(e)
            }
            break  # Stop if we hit memory limits
            
        except Exception as e:
            print(f"❌ Scale {scale:,}: Error - {e}")
            results[scale] = {
                'success': False,
                'error': type(e).__name__,
                'message': str(e)
            }
            # Continue to next scale
    
    # Generate scaling report
    print("\n" + "=" * 70)
    print("SCALING VALIDATION REPORT")
    print("=" * 70)
    
    successful_scales = [s for s, r in results.items() if r.get('success')]
    
    if successful_scales:
        max_scale = max(successful_scales)
        print(f"\n✅ Maximum Successful Scale: {max_scale:,} neurons")
        
        if max_scale >= target_neurons:
            print(f"🎉 TARGET SCALE ACHIEVED: {target_neurons:,} neurons")
        else:
            scale_factor = target_neurons / max_scale
            print(f"📊 Scale Factor Needed: {scale_factor:.1f}x")
            print(f"   Current: {max_scale:,}")
            print(f"   Target: {target_neurons:,}")
        
        print("\nScaling Performance:")
        for scale in sorted(successful_scales):
            r = results[scale]
            print(f"  {scale:>15,} neurons: {r['total_time']:.2f}s, Accuracy: {r['accuracy']:.2%}")
    else:
        print("\n❌ No scales tested successfully")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    if successful_scales:
        max_scale = max(successful_scales)
        if max_scale < 1_000_000:
            print("  - Implement Phase 1 optimizations (memory optimization)")
        elif max_scale < 10_000_000:
            print("  - Implement Phase 2 optimizations (computational optimization)")
        elif max_scale < 100_000_000:
            print("  - Implement Phase 3 optimizations (GPU acceleration)")
        else:
            print("  - Implement Phase 4 optimizations (distributed computing)")
        
        if max_scale < target_neurons:
            print(f"  - Need {target_neurons / max_scale:.1f}x scaling to reach target")
    else:
        print("  - Start with Phase 1 optimizations")
        print("  - Check memory availability")
    
    return results


def get_memory_usage():
    """Get current memory usage"""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    except:
        return None


def has_memory_tracking():
    """Check if memory tracking is available"""
    try:
        import psutil
        return True
    except:
        return False


def main():
    """Main function"""
    target = 80_000_000_000  # 80B
    
    if len(sys.argv) > 1:
        try:
            target = int(sys.argv[1])
        except ValueError:
            print(f"⚠️  Invalid target, using default: {target:,}")
    
    results = validate_scaling(target)
    
    print("\n✅ Validation complete!")


if __name__ == "__main__":
    main()

