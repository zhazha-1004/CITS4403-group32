
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


class SIRModel:
    
    def __init__(self, N: int = 10000, beta: float = 0.30, gamma: float = 0.10,
                 S0: Optional[int] = None, I0: int = 10, R0: int = 0):
        self.N = N
        self.beta = beta
        self.gamma = gamma
        
        self.I0 = I0
        self.R0 = R0
        self.S0 = S0 if S0 is not None else N - I0 - R0
        
        if self.S0 + self.I0 + self.R0 != self.N:
            raise ValueError(f"Initial population sum not equal to N: {self.S0 + self.I0 + self.R0} != {self.N}")
        self.R0_value = self.beta / self.gamma
        
    def step(self, S: float, I: float, R: float) -> Tuple[float, float, float]:
        new_infections = self.beta * S * I / self.N
        new_recoveries = self.gamma * I
        
        S_next = S - new_infections
        I_next = I + new_infections - new_recoveries
        R_next = R + new_recoveries
        
        S_next = max(0, S_next)
        I_next = max(0, I_next)
        R_next = max(0, R_next)
        
        return S_next, I_next, R_next
    
    def simulate(self, days: int = 120) -> pd.DataFrame:
        results = {
            'day': [],
            'S': [],
            'I': [],
            'R': [],
            'S_prop': [],
            'I_prop': [],
            'R_prop': [],
            'new_infections': [],
            'cumulative_infections': []
        }
        
        S, I, R = float(self.S0), float(self.I0), float(self.R0)
        cumulative_infections = self.I0
        results['day'].append(0)
        results['S'].append(S)
        results['I'].append(I)
        results['R'].append(R)
        results['S_prop'].append(S / self.N)
        results['I_prop'].append(I / self.N)
        results['R_prop'].append(R / self.N)
        results['new_infections'].append(self.I0)
        results['cumulative_infections'].append(cumulative_infections)
        
        for day in range(1, days + 1):
            S_prev = S
            S, I, R = self.step(S, I, R)
            
            new_infections = S_prev - S
            cumulative_infections += new_infections
            
            results['day'].append(day)
            results['S'].append(S)
            results['I'].append(I)
            results['R'].append(R)
            results['S_prop'].append(S / self.N)
            results['I_prop'].append(I / self.N)
            results['R_prop'].append(R / self.N)
            results['new_infections'].append(new_infections)
            results['cumulative_infections'].append(cumulative_infections)
            
            if I < 1e-6:
                break
        
        return pd.DataFrame(results)
    
    def get_basic_reproduction_number(self) -> float:
        return self.R0_value
    
    def __str__(self) -> str:
        return (f"SIR Model: N={self.N}, β={self.beta:.3f}, "
                f"γ={self.gamma:.3f}, R0={self.R0_value:.2f}")
    
    def __repr__(self) -> str:
        return self.__str__()
