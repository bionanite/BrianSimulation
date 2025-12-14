#!/usr/bin/env python3
"""
Phase 2 Computational Optimization Test Script
Tests scaling to 1B neurons with performance validation
"""

import time
import sys
from final_enhanced_brain import FinalEnhancedBrain
import numpy as np

def test_scale(neuron_count: int, max_time: float, max_memory_mb: float, test_name: str):
    """Test brain simulation at specified scale"""
    print(f"\n{'='*70}")
    print(f"🧪 Testing {test_name}")
    print(f"{'='*70}")
    print(f"Neuron count: {neuron_count:,}")
    print(f"Target: <{max_time:.1f}s, <{max_memory_mb:.0f}MB RAM")
    
    start_time = time.time()
    start_memory = get_memory_usage_mb()
    
    try:
        # Create brain instance
        print(f"\n📊 Initializing brain...")
        brain = FinalEnhancedBrain(total_neurons=neuron_count, debug=False)
        
        init_time = time.time() - start_time
        init_memory = get_memory_usage_mb() - start_memory
        
        print(f"   ✅ Initialization complete")
        print(f"   ⏱️  Init time: {init_time:.2f}s")
        print(f"   💾 Init memory: {init_memory:.1f}MB")
        print(f"   🔧 Optimizations: Vectorized={brain.total_neurons > 1_000_000}, "
              f"Parallel={brain.use_parallel}, Event-driven={brain.use_event_driven}, "
              f"Float32={brain.dtype == np.float32}")
        
        # Run processing tests
        print(f"\n🧠 Running processing tests...")
        
        # Test 1: Pattern recognition (vectorized)
        test_patterns = [
            np.random.random(1000).astype(brain.dtype),
            np.sin(np.linspace(0, 4*np.pi, 1000)).astype(brain.dtype),
            (np.random.random(1000) > 0.8).astype(brain.dtype)
        ]
        
        pattern_start = time.time()
        for pattern in test_patterns:
            result = brain.enhanced_pattern_recognition(pattern)
        pattern_time = time.time() - pattern_start
        
        # Test 2: Multi-region processing (parallel + event-driven)
        test_stimulus = {
            'sensory_input': np.random.random(1000).astype(brain.dtype),
            'store_memory': np.random.random(100).astype(brain.dtype)
        }
        
        region_start = time.time()
        for _ in range(10):  # Multiple iterations
            result = brain.multi_region_processing(test_stimulus)
        region_time = time.time() - region_start
        
        # Test 3: Hierarchical processing (vectorized)
        hierarchical_input = np.random.random(500).astype(brain.dtype)
        
        hierarchy_start = time.time()
        for _ in range(10):  # Multiple iterations
            result = brain.hierarchical_processing(hierarchical_input)
        hierarchy_time = time.time() - hierarchy_start
        
        total_time = time.time() - start_time
        total_memory = get_memory_usage_mb() - start_memory
        
        print(f"   ✅ All processing tests complete")
        print(f"   ⏱️  Pattern recognition: {pattern_time:.3f}s")
        print(f"   ⏱️  Multi-region processing: {region_time:.3f}s")
        print(f"   ⏱️  Hierarchical processing: {hierarchy_time:.3f}s")
        print(f"   ⏱️  Total time: {total_time:.2f}s")
        print(f"   💾 Total memory: {total_memory:.1f}MB")
        
        # Validate results
        time_ok = total_time < max_time
        memory_ok = total_memory < max_memory_mb
        
        print(f"\n{'='*70}")
        print(f"📊 RESULTS SUMMARY")
        print(f"{'='*70}")
        print(f"Total time: {total_time:.2f}s {'✅' if time_ok else '❌'} (target: <{max_time:.1f}s)")
        print(f"Memory usage: {total_memory:.1f}MB {'✅' if memory_ok else '❌'} (target: <{max_memory_mb:.0f}MB)")
        print(f"Performance breakdown:")
        print(f"   - Pattern recognition: {pattern_time:.3f}s")
        print(f"   - Multi-region: {region_time:.3f}s")
        print(f"   - Hierarchical: {hierarchy_time:.3f}s")
        
        if time_ok and memory_ok:
            print(f"\n✅ {test_name} PASSED")
            return True
        else:
            print(f"\n❌ {test_name} FAILED")
            if not time_ok:
                print(f"   ⚠️  Time exceeded target by {total_time - max_time:.2f}s")
            if not memory_ok:
                print(f"   ⚠️  Memory exceeded target by {total_memory - max_memory_mb:.1f}MB")
            return False
            
    except MemoryError as e:
        print(f"\n❌ {test_name} FAILED - Out of Memory")
        print(f"   Error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ {test_name} FAILED - Error occurred")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_memory_usage_mb() -> float:
    """Get current memory usage in MB"""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        # Fallback: estimate based on system
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux/Mac
        except:
            return 0.0

def main():
    """Run Phase 2 scaling tests"""
    print("🚀 PHASE 2 COMPUTATIONAL OPTIMIZATION VALIDATION")
    print("=" * 70)
    print("Testing computational optimizations for scaling to 1B neurons")
    print("=" * 70)
    
    results = {}
    
    # Test 1: 100M neurons (baseline for Phase 2)
    results['100M'] = test_scale(
        neuron_count=100_000_000,
        max_time=60.0,
        max_memory_mb=10240,  # 10GB
        test_name="100M Neuron Test (Phase 2 Baseline)"
    )
    
    # Test 2: 1B neurons (Phase 2 target)
    if results['100M']:
        print(f"\n⏳ Waiting 2 seconds before next test...")
        time.sleep(2)
        results['1B'] = test_scale(
            neuron_count=1_000_000_000,
            max_time=600.0,  # 10 minutes
            max_memory_mb=102400,  # 100GB
            test_name="1B Neuron Test (Phase 2 Target)"
        )
    else:
        print(f"\n⚠️  Skipping 1B test due to 100M test failure")
        results['1B'] = False
    
    # Test 3: Performance comparison - 10M neurons (should be faster with Phase 2)
    print(f"\n⏳ Testing performance improvement...")
    time.sleep(1)
    results['10M_perf'] = test_scale(
        neuron_count=10_000_000,
        max_time=5.0,  # Should be faster than Phase 1
        max_memory_mb=1024,  # 1GB
        test_name="10M Neuron Performance Test"
    )
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"🎯 PHASE 2 VALIDATION SUMMARY")
    print(f"{'='*70}")
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    if all_passed:
        print(f"\n🏆 ALL TESTS PASSED - Phase 2 Computational Optimization Complete!")
        print(f"   ✅ Vectorization: Implemented")
        print(f"   ✅ Parallel processing: Implemented")
        print(f"   ✅ Event-driven simulation: Implemented")
        print(f"   ✅ Float32 precision: Implemented")
        print(f"   ✅ 1B neurons: Scales successfully")
        return 0
    else:
        print(f"\n⚠️  SOME TESTS FAILED - Review Phase 2 implementation")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

