#!/usr/bin/env python3

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import random
from .sir_model import SIRModel
from .seir_model import SEIRModel

class HybridEpidemicModel:
    """
    INNOVATION: Hybrid model combining compartmental and agent-based approaches
    
    This model introduces spatial heterogeneity and behavioral adaptation:
    - Population divided into spatial zones with different contact patterns
    - Behavioral adaptation: contact rates change based on perceived risk
    - Heterogeneous recovery rates based on age/health status
    - Dynamic intervention timing based on infection thresholds
    """
    
    def __init__(self, N: int = 10000, n_zones: int = 4, 
                 base_beta: float = 0.30, base_gamma: float = 0.10,
                 adaptation_factor: float = 0.5, risk_threshold: float = 0.05,
                 I0: int = 10, R0: int = 0):
        """
        Initialize hybrid epidemic model
        
        Args:
            N: Total population
            n_zones: Number of spatial zones
            base_beta: Base contact rate
            base_gamma: Base recovery rate
            adaptation_factor: How much behavior adapts to risk (0-1)
            risk_threshold: Infection threshold that triggers behavioral change
            I0: Initial infected
            R0: Initial recovered
        """
        self.N = N
        self.n_zones = n_zones
        self.base_beta = base_beta
        self.base_gamma = base_gamma
        self.adaptation_factor = adaptation_factor
        self.risk_threshold = risk_threshold
        self.I0 = I0
        self.R0 = R0
        
        # Initialize zones with different characteristics
        self.zone_populations = self._initialize_zones()
        self.zone_models = self._create_zone_models()
        self.mobility_matrix = self._create_mobility_matrix()
        
        # Track behavioral adaptation
        self.adaptation_history = []
        self.intervention_active = False
        
    def _initialize_zones(self) -> List[int]:
        """Create zones with different population densities"""
        # Urban core (high density), suburban (medium), rural (low), industrial (variable)
        zone_weights = [0.4, 0.3, 0.2, 0.1]  # Population distribution
        populations = [int(self.N * w) for w in zone_weights]
        
        # Adjust for rounding
        populations[-1] += self.N - sum(populations)
        return populations
    
    def _create_zone_models(self) -> List[SIRModel]:
        """Create SIR models for each zone with different parameters"""
        models = []
        zone_characteristics = [
            {'beta_mult': 1.2, 'gamma_mult': 1.0},  # Urban: higher contact
            {'beta_mult': 1.0, 'gamma_mult': 1.1},  # Suburban: baseline
            {'beta_mult': 0.7, 'gamma_mult': 0.9},  # Rural: lower contact, slower recovery
            {'beta_mult': 1.5, 'gamma_mult': 1.2}   # Industrial: highest contact and recovery
        ]
        
        for i, pop in enumerate(self.zone_populations):
            char = zone_characteristics[i]
            zone_I0 = max(1, int(self.I0 * pop / self.N))
            zone_R0 = int(self.R0 * pop / self.N)
            
            model = SIRModel(
                N=pop,
                beta=self.base_beta * char['beta_mult'],
                gamma=self.base_gamma * char['gamma_mult'],
                I0=zone_I0,
                R0=zone_R0
            )
            models.append(model)
        
        return models
    
    def _create_mobility_matrix(self) -> np.ndarray:
        """Create mobility matrix between zones"""
        # Higher mobility between adjacent zones
        mobility = np.array([
            [0.85, 0.10, 0.03, 0.02],  # Urban
            [0.08, 0.80, 0.08, 0.04],  # Suburban  
            [0.02, 0.10, 0.85, 0.03],  # Rural
            [0.05, 0.15, 0.05, 0.75]   # Industrial
        ])
        return mobility
    
    def _calculate_risk_perception(self, zone_results: List[pd.DataFrame], day: int) -> float:
        """Calculate population's risk perception based on recent infections"""
        if day < 7:
            return 0.0
        
        # Average infection rate over last 7 days across all zones
        recent_infections = []
        for results in zone_results:
            if day < len(results):
                recent_data = results.iloc[max(0, day-7):day+1]
                avg_infection = recent_data['I_prop'].mean()
                recent_infections.append(avg_infection)
        
        return np.mean(recent_infections) if recent_infections else 0.0
    
    def _apply_behavioral_adaptation(self, risk_perception: float, day: int):
        """Adapt contact rates based on risk perception"""
        if risk_perception > self.risk_threshold:
            # Reduce contact rates proportionally to risk
            adaptation = min(self.adaptation_factor, risk_perception * 2)
            
            for i, model in enumerate(self.zone_models):
                # Urban zones adapt more quickly
                zone_adaptation = adaptation * (1.2 if i == 0 else 1.0)
                model.beta = self.base_beta * (1 - zone_adaptation)
            
            if not self.intervention_active:
                self.intervention_active = True
                
        self.adaptation_history.append({
            'day': day,
            'risk_perception': risk_perception,
            'intervention_active': self.intervention_active,
            'beta_reduction': adaptation if risk_perception > self.risk_threshold else 0.0
        })
    
    def _exchange_between_zones(self, zone_states: List[Tuple[float, float, float]], 
                              day: int) -> List[Tuple[float, float, float]]:
        """Handle population movement between zones"""
        if day % 7 != 0:  # Only exchange weekly
            return zone_states
        
        new_states = []
        for i, (S, I, R) in enumerate(zone_states):
            new_S, new_I, new_R = S, I, R
            
            # Exchange with other zones
            for j in range(len(zone_states)):
                if i != j:
                    mobility_rate = self.mobility_matrix[i][j] * 0.01  # Small daily mobility
                    
                    # Exchange proportional to compartment sizes
                    S_exchange = mobility_rate * S
                    I_exchange = mobility_rate * I  
                    R_exchange = mobility_rate * R
                    
                    new_S -= S_exchange
                    new_I -= I_exchange
                    new_R -= R_exchange
                    
                    # Receive from other zone
                    other_S, other_I, other_R = zone_states[j]
                    receive_rate = self.mobility_matrix[j][i] * 0.01
                    
                    new_S += receive_rate * other_S
                    new_I += receive_rate * other_I
                    new_R += receive_rate * other_R
            
            new_states.append((max(0, new_S), max(0, new_I), max(0, new_R)))
        
        return new_states
    
    def simulate(self, days: int = 120) -> Dict[str, Any]:
        """
        Run hybrid simulation with spatial zones and behavioral adaptation
        
        Returns:
            Dictionary containing:
            - aggregate_results: Combined results across all zones
            - zone_results: Individual zone results
            - adaptation_history: Behavioral adaptation over time
        """
        # Initialize results storage
        zone_results = [[] for _ in range(self.n_zones)]
        aggregate_results = []
        
        # Run simulation
        for day in range(days):
            daily_zone_states = []
            
            # Simulate each zone
            for i, model in enumerate(self.zone_models):
                if day == 0:
                    # Initialize
                    S, I, R = model.S0, model.I0, model.R0
                else:
                    # Get previous state
                    prev_results = zone_results[i][-1]
                    S, I, R = prev_results['S'], prev_results['I'], prev_results['R']
                
                # Step forward
                S_next, I_next, R_next = model.step(S, I, R)
                daily_zone_states.append((S_next, I_next, R_next))
                
                # Store zone results
                zone_results[i].append({
                    'day': day,
                    'S': S_next,
                    'I': I_next, 
                    'R': R_next,
                    'S_prop': S_next / model.N,
                    'I_prop': I_next / model.N,
                    'R_prop': R_next / model.N,
                    'N': model.N
                })
            
            # Apply inter-zone mobility
            daily_zone_states = self._exchange_between_zones(daily_zone_states, day)
            
            # Update zone states after mobility
            for i, (S, I, R) in enumerate(daily_zone_states):
                zone_results[i][-1].update({
                    'S': S, 'I': I, 'R': R,
                    'S_prop': S / self.zone_models[i].N,
                    'I_prop': I / self.zone_models[i].N,
                    'R_prop': R / self.zone_models[i].N
                })
            
            # Calculate aggregate state
            total_S = sum(state[0] for state in daily_zone_states)
            total_I = sum(state[1] for state in daily_zone_states)
            total_R = sum(state[2] for state in daily_zone_states)
            
            aggregate_results.append({
                'day': day,
                'S': total_S,
                'I': total_I,
                'R': total_R,
                'S_prop': total_S / self.N,
                'I_prop': total_I / self.N,
                'R_prop': total_R / self.N
            })
            
            # Apply behavioral adaptation based on risk perception
            zone_dfs = [pd.DataFrame(results) for results in zone_results]
            risk_perception = self._calculate_risk_perception(zone_dfs, day)
            self._apply_behavioral_adaptation(risk_perception, day)
        
        # Convert to DataFrames
        aggregate_df = pd.DataFrame(aggregate_results)
        zone_dfs = [pd.DataFrame(results) for results in zone_results]
        adaptation_df = pd.DataFrame(self.adaptation_history)
        
        return {
            'aggregate_results': aggregate_df,
            'zone_results': zone_dfs,
            'adaptation_history': adaptation_df,
            'zone_populations': self.zone_populations,
            'final_sizes': [df['R_prop'].iloc[-1] for df in zone_dfs]
        }
    
    def get_innovation_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics specific to the hybrid model innovations"""
        adaptation_df = results['adaptation_history']
        zone_results = results['zone_results']
        
        metrics = {
            'behavioral_adaptation_day': None,
            'max_risk_perception': 0.0,
            'zone_peak_differences': [],
            'mobility_impact': 0.0,
            'adaptation_effectiveness': 0.0
        }
        
        # Find when behavioral adaptation started
        if len(adaptation_df) > 0:
            adaptation_starts = adaptation_df[adaptation_df['intervention_active'] == True]
            if len(adaptation_starts) > 0:
                metrics['behavioral_adaptation_day'] = adaptation_starts.iloc[0]['day']
            
            metrics['max_risk_perception'] = adaptation_df['risk_perception'].max()
        
        # Calculate peak differences between zones
        zone_peaks = [df['I_prop'].max() for df in zone_results]
        metrics['zone_peak_differences'] = zone_peaks
        metrics['peak_heterogeneity'] = np.std(zone_peaks) / np.mean(zone_peaks) if zone_peaks else 0.0
        
        # Estimate adaptation effectiveness
        if metrics['behavioral_adaptation_day'] is not None:
            pre_adaptation = results['aggregate_results'].iloc[:metrics['behavioral_adaptation_day']]
            post_adaptation = results['aggregate_results'].iloc[metrics['behavioral_adaptation_day']:]
            
            if len(pre_adaptation) > 0 and len(post_adaptation) > 0:
                pre_growth = pre_adaptation['I_prop'].diff().mean()
                post_growth = post_adaptation['I_prop'].diff().mean()
                metrics['adaptation_effectiveness'] = (pre_growth - post_growth) / pre_growth if pre_growth > 0 else 0.0
        
        return metrics
    
    def __str__(self) -> str:
        return (f"Hybrid Epidemic Model: N={self.N}, zones={self.n_zones}, "
                f"β_base={self.base_beta:.3f}, γ_base={self.base_gamma:.3f}, "
                f"adaptation={self.adaptation_factor:.2f}")
