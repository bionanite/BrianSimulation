#!/bin/bash

# Sequential Benchmark Testing Script
# Tests multiple neuron counts sequentially to analyze scaling behavior

echo "======================================================================"
echo "SEQUENTIAL BENCHMARK TESTING"
echo "======================================================================"
echo ""

# Define neuron counts to test (in millions)
NEURON_COUNTS=(10000 100000 1000000 10000000 100000000)

# Define benchmarks to test
BENCHMARKS=("HellaSwag" "MMLU" "ARC" "GSM8K")

# Create results directory
mkdir -p sequential_test_results

# Test each neuron count
for neurons in "${NEURON_COUNTS[@]}"; do
    echo ""
    echo "======================================================================"
    echo "Testing with $neurons neurons"
    echo "======================================================================"
    echo ""
    
    # Test each benchmark
    for benchmark in "${BENCHMARKS[@]}"; do
        echo "Testing $benchmark with $neurons neurons..."
        python run_benchmarks.py $neurons $benchmark > "sequential_test_results/${benchmark}_${neurons}.log" 2>&1
        
        # Extract accuracy from log
        accuracy=$(grep "Accuracy:" "sequential_test_results/${benchmark}_${neurons}.log" | head -1 | awk '{print $2}')
        echo "  $benchmark: $accuracy"
    done
    
    echo ""
    echo "Running comprehensive suite for $neurons neurons..."
    python comprehensive_benchmark_suite.py $neurons false 10 false > "sequential_test_results/comprehensive_${neurons}.log" 2>&1
    
    echo "Completed testing with $neurons neurons"
    echo ""
done

echo "======================================================================"
echo "SEQUENTIAL TESTING COMPLETE"
echo "======================================================================"
echo ""
echo "Results saved in: sequential_test_results/"
echo ""
echo "To view results:"
echo "  cat sequential_test_results/*.log"
echo ""

