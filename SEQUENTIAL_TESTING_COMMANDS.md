# Sequential Testing Commands

## Quick Start

### Option 1: Python Script (Recommended)
```bash
python sequential_testing.py
```

This will automatically test:
- Neuron counts: 10K, 100K, 1M, 10M, 100M
- Benchmarks: HellaSwag, MMLU, ARC, GSM8K
- Comprehensive suite for each neuron count

Results will be saved in `sequential_test_results/` directory.

### Option 2: Bash Script
```bash
./sequential_testing.sh
```

### Option 3: Manual Sequential Commands

#### Test Individual Benchmarks at Different Scales

**HellaSwag:**
```bash
python run_benchmarks.py 10000 HellaSwag
python run_benchmarks.py 100000 HellaSwag
python run_benchmarks.py 1000000 HellaSwag
python run_benchmarks.py 10000000 HellaSwag
python run_benchmarks.py 100000000 HellaSwag
```

**MMLU:**
```bash
python run_benchmarks.py 10000 MMLU
python run_benchmarks.py 100000 MMLU
python run_benchmarks.py 1000000 MMLU
python run_benchmarks.py 10000000 MMLU
python run_benchmarks.py 100000000 MMLU
```

**ARC:**
```bash
python run_benchmarks.py 10000 ARC
python run_benchmarks.py 100000 ARC
python run_benchmarks.py 1000000 ARC
python run_benchmarks.py 10000000 ARC
python run_benchmarks.py 100000000 ARC
```

**GSM8K:**
```bash
python run_benchmarks.py 10000 GSM8K
python run_benchmarks.py 100000 GSM8K
python run_benchmarks.py 1000000 GSM8K
python run_benchmarks.py 10000000 GSM8K
python run_benchmarks.py 100000000 GSM8K
```

#### Test Comprehensive Suite at Different Scales

```bash
python comprehensive_benchmark_suite.py 10000 false 10 false
python comprehensive_benchmark_suite.py 100000 false 10 false
python comprehensive_benchmark_suite.py 1000000 false 10 false
python comprehensive_benchmark_suite.py 10000000 false 10 false
python comprehensive_benchmark_suite.py 100000000 false 10 false
```

## Custom Testing

### Test Specific Neuron Counts

Edit `sequential_testing.py` and modify the `NEURON_COUNTS` list:

```python
NEURON_COUNTS = [
    50000,      # 50K
    500000,     # 500K
    5000000,    # 5M
    50000000,   # 50M
]
```

### Test Specific Benchmarks

Edit `sequential_testing.py` and modify the `BENCHMARKS` list:

```python
BENCHMARKS = ["HellaSwag", "MMLU"]  # Only test these
```

### Test with More Questions

```bash
python comprehensive_benchmark_suite.py 1000000 false 50 false  # 50 questions per benchmark
python comprehensive_benchmark_suite.py 1000000 false 100 false  # 100 questions per benchmark
```

## Viewing Results

### View Individual Logs
```bash
cat sequential_test_results/HellaSwag_1000000.log
cat sequential_test_results/comprehensive_1000000.log
```

### View Summary
```bash
cat sequential_test_results/summary.txt
```

### Compare Results Across Scales
```bash
grep "Accuracy:" sequential_test_results/HellaSwag_*.log
grep "Accuracy:" sequential_test_results/MMLU_*.log
grep "Accuracy:" sequential_test_results/ARC_*.log
```

## Quick Test (Fewer Scales)

For faster testing with fewer neuron counts:

```bash
# Test only 1M and 100M
python run_benchmarks.py 1000000 HellaSwag
python run_benchmarks.py 1000000 MMLU
python run_benchmarks.py 1000000 ARC
python comprehensive_benchmark_suite.py 1000000 false 10 false

python run_benchmarks.py 100000000 HellaSwag
python run_benchmarks.py 100000000 MMLU
python run_benchmarks.py 100000000 ARC
python comprehensive_benchmark_suite.py 100000000 false 10 false
```

## Background Execution

To run in background and save output:

```bash
nohup python sequential_testing.py > sequential_testing_output.log 2>&1 &
```

Check progress:
```bash
tail -f sequential_testing_output.log
```

## Expected Duration

- 10K neurons: ~1-2 minutes per benchmark
- 100K neurons: ~1-2 minutes per benchmark
- 1M neurons: ~2-3 minutes per benchmark
- 10M neurons: ~5-10 minutes per benchmark
- 100M neurons: ~10-20 minutes per benchmark

**Total estimated time**: 1-2 hours for full sequential test

## Tips

1. **Start small**: Test with 10K first to ensure everything works
2. **Monitor resources**: Larger neuron counts use more memory
3. **Check logs**: Review logs if any test fails
4. **Compare results**: Use the summary file to compare across scales
5. **Incremental testing**: Test one scale at a time if resources are limited

