#!/usr/bin/env python3
"""
Synaptic Plasticity Mechanisms
Implements STDP, LTP/LTD, and homeostatic plasticity for true learning
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

@dataclass
class SynapticConnection:
    """Enhanced synaptic connection with plasticity tracking"""
    pre_neuron_id: int
    post_neuron_id: int
    weight: float
    delay: float
    neurotransmitter: str  # 'excitatory' or 'inhibitory'
    
    # Plasticity tracking
    last_pre_spike_time: float = -np.inf
    last_post_spike_time: float = -np.inf
    initial_weight: float = 0.0
    max_weight: float = 2.0
    min_weight: float = 0.0
    
    # LTP/LTD tracking
    ltp_count: int = 0
    ltd_count: int = 0
    last_ltp_time: float = -np.inf
    last_ltd_time: float = -np.inf

class STDPPlasticity:
    """
    Spike-Timing Dependent Plasticity
    
    If pre-synaptic fires before post-synaptic → strengthen (LTP)
    If post-synaptic fires before pre-synaptic → weaken (LTD)
    """
    
    def __init__(self, 
                 tau_plus: float = 20.0,  # LTP time constant (ms)
                 tau_minus: float = 20.0,  # LTD time constant (ms)
                 A_plus: float = 0.01,     # LTP amplitude
                 A_minus: float = 0.012):  # LTD amplitude (slightly larger)
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.A_plus = A_plus
        self.A_minus = A_minus
    
    def update_weight(self, 
                     synapse: SynapticConnection,
                     pre_spike_time: float,
                     post_spike_time: float,
                     current_time: float) -> float:
        """
        Update synaptic weight based on spike timing
        
        Args:
            synapse: Synaptic connection to update
            pre_spike_time: Time of pre-synaptic spike
            post_spike_time: Time of post-synaptic spike
            current_time: Current simulation time
            
        Returns:
            Weight change (delta_w)
        """
        delta_w = 0.0
        
        # Pre-synaptic fired before post-synaptic (LTP)
        if pre_spike_time < post_spike_time:
            dt = post_spike_time - pre_spike_time
            if dt < self.tau_plus * 3:  # Within effective window
                delta_w = self.A_plus * np.exp(-dt / self.tau_plus)
                synapse.ltp_count += 1
                synapse.last_ltp_time = current_time
        
        # Post-synaptic fired before pre-synaptic (LTD)
        elif post_spike_time < pre_spike_time:
            dt = pre_spike_time - post_spike_time
            if dt < self.tau_minus * 3:  # Within effective window
                delta_w = -self.A_minus * np.exp(-dt / self.tau_minus)
                synapse.ltd_count += 1
                synapse.last_ltd_time = current_time
        
        # Update weight with bounds
        new_weight = synapse.weight + delta_w
        new_weight = np.clip(new_weight, synapse.min_weight, synapse.max_weight)
        
        weight_change = new_weight - synapse.weight
        synapse.weight = new_weight
        
        return weight_change

class LTP_LTD_Mechanisms:
    """
    Long-Term Potentiation and Depression
    
    High-frequency stimulation → LTP
    Low-frequency stimulation → LTD
    """
    
    def __init__(self,
                 ltp_threshold: float = 50.0,  # Hz for LTP
                 ltd_threshold: float = 1.0,   # Hz for LTD
                 ltp_strength: float = 0.05,
                 ltd_strength: float = 0.03):
        self.ltp_threshold = ltp_threshold
        self.ltd_threshold = ltd_threshold
        self.ltp_strength = ltp_strength
        self.ltd_strength = ltd_strength
    
    def calculate_firing_rate(self, spike_times: List[float], window_ms: float = 100.0) -> float:
        """Calculate firing rate in Hz over a time window"""
        if not spike_times:
            return 0.0
        
        if len(spike_times) < 2:
            return 0.0
        
        # Get spikes in window
        recent_spikes = [t for t in spike_times if t > spike_times[-1] - window_ms]
        
        if len(recent_spikes) < 2:
            return 0.0
        
        # Calculate rate
        time_span = recent_spikes[-1] - recent_spikes[0]
        if time_span <= 0:
            return 0.0
        
        rate = (len(recent_spikes) - 1) / (time_span / 1000.0)  # Convert to Hz
        return rate
    
    def update_weight(self,
                     synapse: SynapticConnection,
                     pre_spike_times: List[float],
                     post_spike_times: List[float],
                     current_time: float) -> float:
        """
        Update weight based on firing rate patterns
        
        High pre-synaptic rate → LTP
        Low pre-synaptic rate → LTD
        """
        if not pre_spike_times:
            return 0.0
        
        # Calculate pre-synaptic firing rate
        pre_rate = self.calculate_firing_rate(pre_spike_times)
        
        delta_w = 0.0
        
        # High-frequency stimulation → LTP
        if pre_rate >= self.ltp_threshold:
            delta_w = self.ltp_strength
            synapse.ltp_count += 1
            synapse.last_ltp_time = current_time
        
        # Low-frequency stimulation → LTD
        elif pre_rate <= self.ltd_threshold and pre_rate > 0:
            delta_w = -self.ltd_strength
            synapse.ltd_count += 1
            synapse.last_ltd_time = current_time
        
        # Update weight
        new_weight = synapse.weight + delta_w
        new_weight = np.clip(new_weight, synapse.min_weight, synapse.max_weight)
        
        weight_change = new_weight - synapse.weight
        synapse.weight = new_weight
        
        return weight_change

class HomeostaticPlasticity:
    """
    Homeostatic Plasticity
    
    Maintains stable average firing rates
    Scales all weights if activity too high/low
    """
    
    def __init__(self,
                 target_firing_rate: float = 7.5,  # Target: 5-10 Hz
                 scaling_factor: float = 0.01,
                 update_window: float = 1000.0):  # ms
        self.target_firing_rate = target_firing_rate
        self.scaling_factor = scaling_factor
        self.update_window = update_window
        self.last_update_time = -np.inf
    
    def calculate_average_firing_rate(self, neurons: List[Dict]) -> float:
        """Calculate average firing rate across all neurons"""
        if not neurons:
            return 0.0
        
        total_rate = 0.0
        count = 0
        
        for neuron in neurons:
            if 'spike_times' in neuron and neuron['spike_times']:
                spike_times = neuron['spike_times']
                if len(spike_times) >= 2:
                    # Use recent spikes within update window
                    recent_spikes = [s for s in spike_times if s > spike_times[-1] - self.update_window]
                    if len(recent_spikes) >= 2:
                        time_span = recent_spikes[-1] - recent_spikes[0]
                        if time_span > 0:
                            rate = (len(recent_spikes) - 1) / (time_span / 1000.0)
                            total_rate += rate
                            count += 1
                    elif len(recent_spikes) == 1:
                        # Single spike - estimate rate based on window
                        rate = 1.0 / (self.update_window / 1000.0)
                        total_rate += rate
                        count += 1
        
        return total_rate / count if count > 0 else 0.0
    
    def update_network(self,
                      synapses: List[SynapticConnection],
                      neurons: List[Dict],
                      current_time: float) -> Dict:
        """
        Update all synapses to maintain target firing rate
        
        Returns:
            Dictionary with update statistics
        """
        # Only update periodically
        if current_time - self.last_update_time < self.update_window:
            return {'updated': False}
        
        # Calculate current average firing rate
        avg_rate = self.calculate_average_firing_rate(neurons)
        
        # Calculate scaling factor
        rate_error = avg_rate - self.target_firing_rate
        scaling = 1.0 + self.scaling_factor * (-rate_error / self.target_firing_rate)
        
        # Clamp scaling to reasonable range
        scaling = np.clip(scaling, 0.9, 1.1)
        
        # Apply scaling to all synapses
        weights_changed = 0
        total_change = 0.0
        
        for synapse in synapses:
            old_weight = synapse.weight
            synapse.weight *= scaling
            synapse.weight = np.clip(synapse.weight, synapse.min_weight, synapse.max_weight)
            
            if abs(synapse.weight - old_weight) > 1e-6:
                weights_changed += 1
                total_change += abs(synapse.weight - old_weight)
        
        self.last_update_time = current_time
        
        return {
            'updated': True,
            'avg_firing_rate': avg_rate,
            'target_rate': self.target_firing_rate,
            'rate_error': rate_error,
            'scaling_factor': scaling,
            'weights_changed': weights_changed,
            'total_weight_change': total_change
        }

class PlasticityManager:
    """
    Manages all plasticity mechanisms together
    """
    
    def __init__(self,
                 enable_stdp: bool = True,
                 enable_ltp_ltd: bool = True,
                 enable_homeostatic: bool = True):
        self.enable_stdp = enable_stdp
        self.enable_ltp_ltd = enable_ltp_ltd
        self.enable_homeostatic = enable_homeostatic
        
        self.stdp = STDPPlasticity() if enable_stdp else None
        self.ltp_ltd = LTP_LTD_Mechanisms() if enable_ltp_ltd else None
        self.homeostatic = HomeostaticPlasticity() if enable_homeostatic else None
        
        self.plasticity_history = []
    
    def process_spike_pair(self,
                          synapse: SynapticConnection,
                          pre_spike_time: float,
                          post_spike_time: float,
                          current_time: float) -> Dict:
        """
        Process a spike pair and update weights
        
        Returns:
            Dictionary with plasticity updates
        """
        updates = {
            'stdp_change': 0.0,
            'ltp_ltd_change': 0.0,
            'total_change': 0.0
        }
        
        # Update spike times
        synapse.last_pre_spike_time = pre_spike_time
        synapse.last_post_spike_time = post_spike_time
        
        # STDP update
        if self.enable_stdp and self.stdp:
            updates['stdp_change'] = self.stdp.update_weight(
                synapse, pre_spike_time, post_spike_time, current_time
            )
        
        updates['total_change'] = updates['stdp_change']
        
        return updates
    
    def process_firing_rate(self,
                           synapse: SynapticConnection,
                           pre_spike_times: List[float],
                           post_spike_times: List[float],
                           current_time: float) -> Dict:
        """
        Process firing rate patterns and update weights
        
        Returns:
            Dictionary with LTP/LTD updates
        """
        updates = {
            'ltp_ltd_change': 0.0
        }
        
        # LTP/LTD update
        if self.enable_ltp_ltd and self.ltp_ltd:
            updates['ltp_ltd_change'] = self.ltp_ltd.update_weight(
                synapse, pre_spike_times, post_spike_times, current_time
            )
        
        return updates
    
    def update_homeostasis(self,
                          synapses: List[SynapticConnection],
                          neurons: List[Dict],
                          current_time: float) -> Dict:
        """
        Update homeostatic plasticity
        
        Returns:
            Dictionary with homeostatic update statistics
        """
        if self.enable_homeostatic and self.homeostatic:
            result = self.homeostatic.update_network(synapses, neurons, current_time)
            return result
        return {'updated': False}
    
    def get_statistics(self, synapses: List[SynapticConnection]) -> Dict:
        """Get statistics about plasticity"""
        if not synapses:
            return {}
        
        total_ltp = sum(s.ltp_count for s in synapses)
        total_ltd = sum(s.ltd_count for s in synapses)
        avg_weight = np.mean([s.weight for s in synapses])
        weight_std = np.std([s.weight for s in synapses])
        
        return {
            'total_synapses': len(synapses),
            'total_ltp_events': total_ltp,
            'total_ltd_events': total_ltd,
            'ltp_ltd_ratio': total_ltp / max(1, total_ltd),
            'average_weight': avg_weight,
            'weight_std': weight_std,
            'weight_range': (min(s.weight for s in synapses), max(s.weight for s in synapses))
        }

