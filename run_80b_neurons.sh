#!/bin/bash
# Run Brain Simulation with 80 Billion Neurons
# This is the human brain scale!

echo "🧠 Starting 80 Billion Neuron Brain Simulation..."
echo "================================================"
echo ""
echo "⚠️  WARNING: This will require significant computational resources:"
echo "   - Memory: ~100GB+ (uses sparse matrices and memory-mapped files)"
echo "   - Processing: May take hours depending on hardware"
echo "   - Distributed Computing: Will use MPI if available"
echo ""
echo "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
sleep 5

# Run with 80 billion neurons
python final_enhanced_brain.py 80000000000

echo ""
echo "✅ Simulation complete!"
echo "Results saved to: final_enhanced_brain_results.json"

