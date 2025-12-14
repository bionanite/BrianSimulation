#!/usr/bin/env python3
"""
Sequential Benchmark Testing Script
Tests multiple neuron counts sequentially to analyze scaling behavior
"""

import subprocess
import sys
import os
from datetime import datetime

# Define neuron counts to test
NEURON_COUNTS = [
    10000,      # 10K
    100000,     # 100K
    1000000,    # 1M
    10000000,   # 10M
    100000000,  # 100M
]

# Define benchmarks to test
BENCHMARKS = ["HellaSwag", "MMLU", "ARC", "GSM8K"]

# Create results directory
RESULTS_DIR = "sequential_test_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_command(cmd, description):
    """Run a command and return output"""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        print(f"Error running command: {e}")
        return "", str(e), 1

def extract_accuracy(output):
    """Extract accuracy from benchmark output"""
    for line in output.split('\n'):
        if 'Accuracy:' in line:
            parts = line.split('Accuracy:')
            if len(parts) > 1:
                acc_part = parts[1].strip().split()[0]
                return acc_part
    return "N/A"

def main():
    print("="*70)
    print("SEQUENTIAL BENCHMARK TESTING")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Neuron counts to test: {NEURON_COUNTS}")
    print(f"Benchmarks to test: {BENCHMARKS}")
    print()
    
    # Summary results
    summary = {}
    
    # Test each neuron count
    for neurons in NEURON_COUNTS:
        print(f"\n{'#'*70}")
        print(f"# Testing with {neurons:,} neurons")
        print(f"{'#'*70}\n")
        
        summary[neurons] = {}
        
        # Test each benchmark
        for benchmark in BENCHMARKS:
            print(f"\n--- Testing {benchmark} with {neurons:,} neurons ---")
            
            cmd = ["python", "run_benchmarks.py", str(neurons), benchmark]
            stdout, stderr, returncode = run_command(
                cmd,
                f"{benchmark} @ {neurons:,} neurons"
            )
            
            # Save output
            log_file = os.path.join(RESULTS_DIR, f"{benchmark}_{neurons}.log")
            with open(log_file, 'w') as f:
                f.write(f"=== {benchmark} @ {neurons:,} neurons ===\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Return code: {returncode}\n\n")
                f.write("STDOUT:\n")
                f.write(stdout)
                f.write("\n\nSTDERR:\n")
                f.write(stderr)
            
            # Extract accuracy
            accuracy = extract_accuracy(stdout)
            summary[neurons][benchmark] = accuracy
            
            print(f"  Result: {accuracy}")
            if returncode != 0:
                print(f"  ⚠️  Warning: Command returned non-zero exit code: {returncode}")
        
        # Run comprehensive suite
        print(f"\n--- Running comprehensive suite for {neurons:,} neurons ---")
        cmd = ["python", "comprehensive_benchmark_suite.py", str(neurons), "false", "10", "false"]
        stdout, stderr, returncode = run_command(
            cmd,
            f"Comprehensive Suite @ {neurons:,} neurons"
        )
        
        # Save comprehensive output
        log_file = os.path.join(RESULTS_DIR, f"comprehensive_{neurons}.log")
        with open(log_file, 'w') as f:
            f.write(f"=== Comprehensive Suite @ {neurons:,} neurons ===\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Return code: {returncode}\n\n")
            f.write("STDOUT:\n")
            f.write(stdout)
            f.write("\n\nSTDERR:\n")
            f.write(stderr)
        
        print(f"  Comprehensive suite completed")
    
    # Print summary
    print("\n" + "="*70)
    print("TESTING SUMMARY")
    print("="*70)
    print(f"\n{'Neurons':<15}", end="")
    for benchmark in BENCHMARKS:
        print(f"{benchmark:<15}", end="")
    print()
    print("-" * (15 + 15 * len(BENCHMARKS)))
    
    for neurons in NEURON_COUNTS:
        print(f"{neurons:<15,}", end="")
        for benchmark in BENCHMARKS:
            acc = summary[neurons].get(benchmark, "N/A")
            print(f"{acc:<15}", end="")
        print()
    
    # Save summary
    summary_file = os.path.join(RESULTS_DIR, "summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Sequential Testing Summary\n")
        f.write("="*70 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"{'Neurons':<15}")
        for benchmark in BENCHMARKS:
            f.write(f"{benchmark:<15}")
        f.write("\n")
        f.write("-" * (15 + 15 * len(BENCHMARKS)) + "\n")
        
        for neurons in NEURON_COUNTS:
            f.write(f"{neurons:<15,}")
            for benchmark in BENCHMARKS:
                acc = summary[neurons].get(benchmark, "N/A")
                f.write(f"{acc:<15}")
            f.write("\n")
    
    print(f"\n{'='*70}")
    print("SEQUENTIAL TESTING COMPLETE")
    print(f"{'='*70}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults saved in: {RESULTS_DIR}/")
    print(f"Summary saved in: {summary_file}")
    print()

if __name__ == "__main__":
    main()

