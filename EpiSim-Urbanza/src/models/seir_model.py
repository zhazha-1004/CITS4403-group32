
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


class SEIRModel:
    
    def __init__(self, N: int = 10000, beta: float = 0.30, sigma: float = 0.20,
                 gamma: float = 0.10, S0: Optional[int] = None, E0: int = 0,
                 I0: int = 10, R0: int = 0):
        self.N = N
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        
        self.E0 = E0
        self.I0 = I0
        self.R0 = R0
        self.S0 = S0 if S0 is not None else N - E0 - I0 - R0
        
        if self.S0 + self.E0 + self.I0 + self.R0 != self.N:
            raise ValueError(f"Initial population sum not equal to N: {self.S0 + self.E0 + self.I0 + self.R0} != {self.N}")
        self.R0_value = self.beta / self.gamma
        
    def step(self, S: float, E: float, I: float, R: float) -> Tuple[float, float, float, float]:
        new_exposures = self.beta * S * I / self.N
        new_infections = self.sigma * E
        new_recoveries = self.gamma * I
        
        S_next = S - new_exposures
        E_next = E + new_exposures - new_infections
        I_next = I + new_infections - new_recoveries
        R_next = R + new_recoveries
        
        S_next = max(0, S_next)
        E_next = max(0, E_next)
        I_next = max(0, I_next)
        R_next = max(0, R_next)
        
        return S_next, E_next, I_next, R_next
    
    def simulate(self, days: int = 120) -> pd.DataFrame:
        results = {
            'day': [],
            'S': [],
            'E': [],
            'I': [],
            'R': [],
            'S_prop': [],
            'E_prop': [],
            'I_prop': [],
            'R_prop': [],
            'new_exposures': [],
            'new_infections': [],
            'cumulative_infections': []
        }
        
        S, E, I, R = float(self.S0), float(self.E0), float(self.I0), float(self.R0)
        cumulative_infections = self.I0
        
        results['day'].append(0)
        results['S'].append(S)
        results['E'].append(E)
        results['I'].append(I)
        results['R'].append(R)
        results['S_prop'].append(S / self.N)
        results['E_prop'].append(E / self.N)
        results['I_prop'].append(I / self.N)
        results['R_prop'].append(R / self.N)
        results['new_exposures'].append(0)
        results['new_infections'].append(self.I0)
        results['cumulative_infections'].append(cumulative_infections)
        for day in range(1, days + 1):
            S_prev, E_prev = S, E
            S, E, I, R = self.step(S, E, I, R)
            
            new_exposures = S_prev - S
            new_infections = self.sigma * E_prev
            cumulative_infections += new_infections
            
            results['day'].append(day)
            results['S'].append(S)
            results['E'].append(E)
            results['I'].append(I)
            results['R'].append(R)
            results['S_prop'].append(S / self.N)
            results['E_prop'].append(E / self.N)
            results['I_prop'].append(I / self.N)
            results['R_prop'].append(R / self.N)
            results['new_exposures'].append(new_exposures)
            results['new_infections'].append(new_infections)
            results['cumulative_infections'].append(cumulative_infections)
            
            if I < 1e-6 and E < 1e-6:
                break
        
        return pd.DataFrame(results)
    
    def get_basic_reproduction_number(self) -> float:
        return self.R0_value
    
    def __str__(self) -> str:
        return (f"SEIR Model: N={self.N}, β={self.beta:.3f}, "
                f"σ={self.sigma:.3f}, γ={self.gamma:.3f}, R0={self.R0_value:.2f}")
    
    def __repr__(self) -> str:
        return self.__str__()
