"""
Biologically Realistic Neuron Simulation
Based on the Hodgkin-Huxley model with additional biological features
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from dataclasses import dataclass
from typing import List, Dict, Optional
import random

@dataclass
class SynapseConnection:
    """Represents a synaptic connection between neurons"""
    pre_neuron_id: int
    post_neuron_id: int
    weight: float
    delay: float  # synaptic delay in ms
    neurotransmitter: str  # 'excitatory' or 'inhibitory'
    plasticity: float = 1.0  # synaptic plasticity factor

class BiologicalNeuron:
    """
    Biologically realistic neuron based on Hodgkin-Huxley model
    Includes:
    - Action potential generation
    - Refractory periods
    - Synaptic integration
    - Dendritic processing
    - Axonal conduction
    """
    
    def __init__(self, neuron_id: int, neuron_type: str = "pyramidal"):
        self.id = neuron_id
        self.type = neuron_type
        
        # Membrane properties
        self.V_m = -70.0  # resting membrane potential (mV)
        self.V_rest = -70.0
        self.V_threshold = -55.0  # action potential threshold
        self.V_peak = 30.0
        self.V_reset = -80.0
        
        # Membrane capacitance and resistance
        self.C_m = 1.0  # membrane capacitance (µF/cm²)
        self.g_leak = 0.3  # leak conductance
        self.E_leak = -70.0  # leak reversal potential
        
        # Hodgkin-Huxley ion channel parameters
        self.g_Na_max = 120.0  # max sodium conductance
        self.g_K_max = 36.0    # max potassium conductance
        self.E_Na = 50.0       # sodium reversal potential
        self.E_K = -77.0       # potassium reversal potential
        
        # Ion channel gating variables
        self.m = 0.05  # sodium activation
        self.h = 0.6   # sodium inactivation  
        self.n = 0.32  # potassium activation
        
        # Synaptic properties
        self.synapses_in = []  # incoming synapses
        self.synapses_out = []  # outgoing synapses
        self.synaptic_current = 0.0
        
        # Dendritic properties
        self.dendrite_branches = 5
        self.dendritic_inputs = []
        
        # Axonal properties
        self.axon_length = random.uniform(0.1, 10.0)  # mm
        self.conduction_velocity = 1.0  # m/s
        self.myelinated = random.choice([True, False])
        if self.myelinated:
            self.conduction_velocity *= 10
            
        # State tracking
        self.last_spike_time = -100.0
        self.refractory_period = 2.0  # ms
        self.is_firing = False
        self.spike_count = 0
        
        # History for analysis
        self.voltage_history = []
        self.time_history = []
        self.spike_times = []
        
    def alpha_m(self, V):
        """Sodium activation rate"""
        if abs(V + 40) < 1e-6:
            return 1.0
        return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
    
    def beta_m(self, V):
        """Sodium activation rate"""
        return 4.0 * np.exp(-(V + 65) / 18)
    
    def alpha_h(self, V):
        """Sodium inactivation rate"""
        return 0.07 * np.exp(-(V + 65) / 20)
    
    def beta_h(self, V):
        """Sodium inactivation rate"""
        return 1.0 / (1 + np.exp(-(V + 35) / 10))
    
    def alpha_n(self, V):
        """Potassium activation rate"""
        if abs(V + 55) < 1e-6:
            return 0.1
        return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
    
    def beta_n(self, V):
        """Potassium activation rate"""
        return 0.125 * np.exp(-(V + 65) / 80)
    
    def update_gating_variables(self, dt):
        """Update ion channel gating variables using Hodgkin-Huxley equations"""
        V = self.V_m
        
        # Update m (sodium activation)
        alpha_m = self.alpha_m(V)
        beta_m = self.beta_m(V)
        dm_dt = alpha_m * (1 - self.m) - beta_m * self.m
        self.m += dt * dm_dt
        
        # Update h (sodium inactivation)
        alpha_h = self.alpha_h(V)
        beta_h = self.beta_h(V)
        dh_dt = alpha_h * (1 - self.h) - beta_h * self.h
        self.h += dt * dh_dt
        
        # Update n (potassium activation)
        alpha_n = self.alpha_n(V)
        beta_n = self.beta_n(V)
        dn_dt = alpha_n * (1 - self.n) - beta_n * self.n
        self.n += dt * dn_dt
        
        # Clamp values between 0 and 1
        self.m = np.clip(self.m, 0, 1)
        self.h = np.clip(self.h, 0, 1)
        self.n = np.clip(self.n, 0, 1)
    
    def calculate_currents(self):
        """Calculate ionic currents"""
        V = self.V_m
        
        # Sodium current
        g_Na = self.g_Na_max * (self.m ** 3) * self.h
        I_Na = g_Na * (V - self.E_Na)
        
        # Potassium current
        g_K = self.g_K_max * (self.n ** 4)
        I_K = g_K * (V - self.E_K)
        
        # Leak current
        I_leak = self.g_leak * (V - self.E_leak)
        
        return I_Na, I_K, I_leak
    
    def process_synaptic_inputs(self, current_time):
        """Process incoming synaptic inputs"""
        total_current = 0.0
        
        for synapse in self.synapses_in:
            # Check if presynaptic neuron has fired recently
            delay_time = current_time - synapse.delay
            if hasattr(synapse, 'last_activation') and synapse.last_activation > delay_time:
                # Apply synaptic current
                if synapse.neurotransmitter == 'excitatory':
                    current = synapse.weight * synapse.plasticity * 2.0  # EPSC
                else:
                    current = -synapse.weight * synapse.plasticity * 2.0  # IPSC
                
                # Exponential decay
                decay_factor = np.exp(-(current_time - synapse.last_activation) / 2.0)
                total_current += current * decay_factor
        
        self.synaptic_current = total_current
        return total_current
    
    def update(self, dt, current_time, external_current=0.0):
        """Update neuron state using Hodgkin-Huxley equations"""
        
        # Check refractory period
        if current_time - self.last_spike_time < self.refractory_period:
            self.is_firing = False
            return False
        
        # Update gating variables
        self.update_gating_variables(dt)
        
        # Calculate ionic currents
        I_Na, I_K, I_leak = self.calculate_currents()
        
        # Process synaptic inputs
        I_syn = self.process_synaptic_inputs(current_time)
        
        # Total current
        I_total = -I_Na - I_K - I_leak + I_syn + external_current
        
        # Update membrane potential
        dV_dt = I_total / self.C_m
        self.V_m += dt * dV_dt
        
        # Check for action potential
        spike_occurred = False
        if self.V_m >= self.V_threshold and not self.is_firing:
            self.is_firing = True
            self.last_spike_time = current_time
            self.spike_count += 1
            self.spike_times.append(current_time)
            spike_occurred = True
            
            # Propagate spike to output synapses
            for synapse in self.synapses_out:
                synapse.last_activation = current_time
        
        # Reset after peak
        if self.is_firing and self.V_m >= self.V_peak:
            self.V_m = self.V_reset
            self.is_firing = False
        
        # Store history
        self.voltage_history.append(self.V_m)
        self.time_history.append(current_time)
        
        return spike_occurred
    
    def add_synapse(self, target_neuron, weight=1.0, delay=1.0, neurotransmitter='excitatory'):
        """Add synaptic connection to another neuron"""
        synapse = SynapseConnection(
            pre_neuron_id=self.id,
            post_neuron_id=target_neuron.id,
            weight=weight,
            delay=delay,
            neurotransmitter=neurotransmitter
        )
        
        self.synapses_out.append(synapse)
        target_neuron.synapses_in.append(synapse)
        return synapse
    
    def get_firing_rate(self, window_ms=1000):
        """Calculate current firing rate"""
        current_time = self.time_history[-1] if self.time_history else 0
        recent_spikes = [t for t in self.spike_times if current_time - t <= window_ms]
        return len(recent_spikes) * (1000.0 / window_ms)  # spikes per second
    
    def plot_activity(self, title_suffix=""):
        """Plot neuron activity"""
        if len(self.voltage_history) < 2:
            print("Not enough data to plot")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Voltage trace
        ax1.plot(self.time_history, self.voltage_history, 'b-', linewidth=1)
        ax1.axhline(y=self.V_threshold, color='r', linestyle='--', label='Threshold')
        ax1.set_ylabel('Membrane Potential (mV)')
        ax1.set_title(f'Neuron {self.id} - Voltage Trace {title_suffix}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Spike raster
        if self.spike_times:
            ax2.eventplot([self.spike_times], colors=['red'], linewidths=2)
            ax2.set_ylabel('Spikes')
            ax2.set_xlabel('Time (ms)')
            ax2.set_title(f'Spike Times (Rate: {self.get_firing_rate():.1f} Hz)')
        else:
            ax2.set_ylabel('Spikes')
            ax2.set_xlabel('Time (ms)')
            ax2.set_title('No spikes detected')
        
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

def test_single_neuron():
    """Test a single neuron with external stimulation"""
    print("🧠 Testing Single Biological Neuron")
    print("=" * 50)
    
    # Create neuron
    neuron = BiologicalNeuron(neuron_id=1, neuron_type="pyramidal")
    
    # Simulation parameters
    dt = 0.01  # time step (ms)
    total_time = 100.0  # total simulation time (ms)
    current_time = 0.0
    
    print(f"Initial membrane potential: {neuron.V_m:.1f} mV")
    print(f"Threshold: {neuron.V_threshold:.1f} mV")
    print(f"Axon length: {neuron.axon_length:.2f} mm")
    print(f"Myelinated: {neuron.myelinated}")
    print(f"Conduction velocity: {neuron.conduction_velocity:.1f} m/s")
    print()
    
    # Apply stimulus current
    stimulus_start = 20.0
    stimulus_end = 80.0
    stimulus_amplitude = 10.0  # µA/cm²
    
    print(f"Applying stimulus: {stimulus_amplitude} µA/cm² from {stimulus_start}-{stimulus_end} ms")
    print("Running simulation...")
    
    spike_count = 0
    
    # Run simulation
    while current_time < total_time:
        # Apply external current during stimulus period
        if stimulus_start <= current_time <= stimulus_end:
            external_current = stimulus_amplitude
        else:
            external_current = 0.0
        
        # Update neuron
        spike_occurred = neuron.update(dt, current_time, external_current)
        
        if spike_occurred:
            spike_count += 1
            print(f"⚡ Spike #{spike_count} at t = {current_time:.2f} ms")
        
        current_time += dt
    
    # Results
    print(f"\nSimulation Results:")
    print(f"Total spikes: {neuron.spike_count}")
    print(f"Average firing rate: {neuron.get_firing_rate():.1f} Hz")
    print(f"Final membrane potential: {neuron.V_m:.1f} mV")
    
    # Plot results
    fig = neuron.plot_activity("- Single Neuron Test")
    plt.savefig('/home/user/single_neuron_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return neuron

if __name__ == "__main__":
    test_single_neuron()