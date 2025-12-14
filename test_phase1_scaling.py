#!/usr/bin/env python3
"""
Phase 1 Memory Optimization Test Script
Tests scaling to 10M and 100M neurons with memory and performance validation
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
        
        # Run a simple processing test
        print(f"\n🧠 Running processing test...")
        test_stimulus = {
            'sensory_input': np.random.random(1000),
            'store_memory': np.random.random(100)
        }
        
        process_start = time.time()
        result = brain.multi_region_processing(test_stimulus)
        process_time = time.time() - process_start
        
        total_time = time.time() - start_time
        total_memory = get_memory_usage_mb() - start_memory
        
        print(f"   ✅ Processing complete")
        print(f"   ⏱️  Process time: {process_time:.2f}s")
        print(f"   💾 Total memory: {total_memory:.1f}MB")
        print(f"   📈 Coordination score: {result['coordination_score']:.3f}")
        print(f"   🔥 Active regions: {result['active_regions']}/5")
        
        # Validate results
        time_ok = total_time < max_time
        memory_ok = total_memory < max_memory_mb
        
        print(f"\n{'='*70}")
        print(f"📊 RESULTS SUMMARY")
        print(f"{'='*70}")
        print(f"Total time: {total_time:.2f}s {'✅' if time_ok else '❌'} (target: <{max_time:.1f}s)")
        print(f"Memory usage: {total_memory:.1f}MB {'✅' if memory_ok else '❌'} (target: <{max_memory_mb:.0f}MB)")
        
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
    """Run Phase 1 scaling tests"""
    print("🚀 PHASE 1 MEMORY OPTIMIZATION VALIDATION")
    print("=" * 70)
    print("Testing memory optimizations for scaling to 100M neurons")
    print("=" * 70)
    
    results = {}
    
    # Test 1: 10M neurons (should work, <5s, <1GB)
    results['10M'] = test_scale(
        neuron_count=10_000_000,
        max_time=5.0,
        max_memory_mb=1024,  # 1GB
        test_name="10M Neuron Test"
    )
    
    # Test 2: 100M neurons (should work, <60s, <10GB)
    if results['10M']:
        print(f"\n⏳ Waiting 2 seconds before next test...")
        time.sleep(2)
        results['100M'] = test_scale(
            neuron_count=100_000_000,
            max_time=60.0,
            max_memory_mb=10240,  # 10GB
            test_name="100M Neuron Test"
        )
    else:
        print(f"\n⚠️  Skipping 100M test due to 10M test failure")
        results['100M'] = False
    
    # Test 3: Backward compatibility - 1M neurons (should work as before)
    print(f"\n⏳ Testing backward compatibility...")
    time.sleep(1)
    results['1M_compat'] = test_scale(
        neuron_count=1_000_000,
        max_time=20.0,  # More lenient for compatibility test
        max_memory_mb=5000,  # More lenient
        test_name="1M Neuron Compatibility Test"
    )
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"🎯 PHASE 1 VALIDATION SUMMARY")
    print(f"{'='*70}")
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    if all_passed:
        print(f"\n🏆 ALL TESTS PASSED - Phase 1 Memory Optimization Complete!")
        print(f"   ✅ 10M neurons: Works efficiently")
        print(f"   ✅ 100M neurons: Scales successfully")
        print(f"   ✅ Backward compatibility: Maintained")
        return 0
    else:
        print(f"\n⚠️  SOME TESTS FAILED - Review Phase 1 implementation")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

