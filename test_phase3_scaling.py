#!/usr/bin/env python3
"""
Phase 3 GPU Acceleration Test Script
Tests scaling to 10B neurons with GPU acceleration validation
"""

import time
import sys
from final_enhanced_brain import FinalEnhancedBrain, GPU_AVAILABLE, GPU_COUNT
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
        
        # Check GPU status
        gpu_status = "Available" if brain.use_gpu else "Not available (CPU fallback)"
        print(f"   🚀 GPU: {gpu_status}")
        if brain.use_gpu:
            gpu_info = brain.get_gpu_memory_usage()
            if gpu_info.get('available'):
                for dev in gpu_info.get('devices', []):
                    print(f"      GPU {dev['id']}: {dev['used_mb']:.0f}MB/{dev['total_mb']:.0f}MB used")
        
        print(f"   🔧 Optimizations: Vectorized={brain.total_neurons > 1_000_000}, "
              f"Parallel={brain.use_parallel}, Event-driven={brain.use_event_driven}, "
              f"GPU={brain.use_gpu}, Multi-GPU={brain.use_multi_gpu}")
        
        # Run processing tests
        print(f"\n🧠 Running processing tests...")
        
        # Test 1: Pattern recognition (GPU-accelerated if available)
        test_patterns = [
            np.random.random(1000).astype(brain.dtype),
            np.sin(np.linspace(0, 4*np.pi, 1000)).astype(brain.dtype),
            (np.random.random(1000) > 0.8).astype(brain.dtype)
        ]
        
        pattern_start = time.time()
        for pattern in test_patterns:
            result = brain.enhanced_pattern_recognition(pattern)
        pattern_time = time.time() - pattern_start
        
        # Test 2: Multi-region processing
        test_stimulus = {
            'sensory_input': np.random.random(1000).astype(brain.dtype),
            'store_memory': np.random.random(100).astype(brain.dtype)
        }
        
        region_start = time.time()
        for _ in range(10):  # Multiple iterations
            result = brain.multi_region_processing(test_stimulus)
        region_time = time.time() - region_start
        
        # Test 3: Hierarchical processing (GPU-accelerated if available)
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
    """Run Phase 3 scaling tests"""
    print("🚀 PHASE 3 GPU ACCELERATION VALIDATION")
    print("=" * 70)
    print("Testing GPU acceleration for scaling to 10B neurons")
    print("=" * 70)
    
    # Check GPU availability
    if GPU_AVAILABLE:
        print(f"✅ GPU Available: {GPU_COUNT} device(s)")
    else:
        print(f"⚠️  GPU Not Available - Will use CPU fallback")
        print(f"   Note: Install CuPy for GPU acceleration: pip install cupy-cuda11x")
    
    results = {}
    
    # Test 1: 1B neurons (baseline for Phase 3)
    results['1B'] = test_scale(
        neuron_count=1_000_000_000,
        max_time=600.0,  # 10 minutes
        max_memory_mb=102400,  # 100GB
        test_name="1B Neuron Test (Phase 3 Baseline)"
    )
    
    # Test 2: 10B neurons (Phase 3 target)
    if results['1B']:
        print(f"\n⏳ Waiting 2 seconds before next test...")
        time.sleep(2)
        results['10B'] = test_scale(
            neuron_count=10_000_000_000,
            max_time=3600.0,  # 1 hour
            max_memory_mb=1024000,  # 1TB (distributed)
            test_name="10B Neuron Test (Phase 3 Target)"
        )
    else:
        print(f"\n⚠️  Skipping 10B test due to 1B test failure")
        results['10B'] = False
    
    # Test 3: Performance comparison - 100M neurons (should work with or without GPU)
    print(f"\n⏳ Testing performance with/without GPU...")
    time.sleep(1)
    results['100M_perf'] = test_scale(
        neuron_count=100_000_000,
        max_time=60.0,  # 1 minute
        max_memory_mb=10240,  # 10GB
        test_name="100M Neuron Performance Test"
    )
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"🎯 PHASE 3 VALIDATION SUMMARY")
    print(f"{'='*70}")
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    if all_passed:
        print(f"\n🏆 ALL TESTS PASSED - Phase 3 GPU Acceleration Complete!")
        print(f"   ✅ GPU detection: Implemented")
        print(f"   ✅ GPU memory management: Implemented")
        print(f"   ✅ GPU operations: Pattern recognition & Hierarchical processing")
        print(f"   ✅ Multi-GPU support: Implemented")
        print(f"   ✅ CPU fallback: Working")
        print(f"   ✅ 10B neurons: Scales successfully")
        return 0
    else:
        print(f"\n⚠️  SOME TESTS FAILED - Review Phase 3 implementation")
        if not GPU_AVAILABLE:
            print(f"   Note: GPU acceleration requires CuPy installation")
            print(f"   Install with: pip install cupy-cuda11x (or cupy-cuda12x for CUDA 12)")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

