"""
Quantum State Simulator - Abstract interface for quantum simulation.

Provides a unified interface to multiple quantum simulation backends
with automatic backend selection based on system requirements.
"""

import numpy as np
from typing import List, Union, Dict, Optional


class QuantumStateSim:
    """
    Unified quantum simulation interface with pluggable backends.
    
    Automatically selects appropriate backend based on:
    - Number of qubits
    - Available libraries
    - User preference
    
    Attributes:
        num_qubits: Number of qubits in the system
        backend: Active simulation backend
        backend_name: Name of the backend in use
    """
    
    def __init__(self, num_qubits: int, backend: str = "auto", noise_model=None):
        """
        Initialize quantum state simulator.
        
        Args:
            num_qubits: Number of qubits (3-5 recommended)
            backend: Backend type ("auto", "native", "qiskit", "cirq")
            noise_model: Optional noise model for qiskit backend
        """
        self.num_qubits = num_qubits
        self.backend_name = backend
        self.noise_model = noise_model
        
        self.backend = self._select_backend(backend)
    
    def _select_backend(self, backend: str):
        """
        Select and initialize appropriate backend.
        
        Args:
            backend: Backend type string
            
        Returns:
            Initialized backend instance
        """
        if backend == "auto":
            # Auto-select based on availability and qubit count
            if self.num_qubits <= 20:
                backend = "native"
            else:
                backend = "qiskit"  # Try qiskit for larger systems
        
        if backend == "native":
            from ftqec.core.quantum_sim.backends.native_complex_sim import NativeComplexSim
            self.backend_name = "native"
            return NativeComplexSim(self.num_qubits)
        
        elif backend == "qiskit":
            from ftqec.core.quantum_sim.backends.qiskit_backend import QiskitBackend
            self.backend_name = "qiskit"
            return QiskitBackend(self.num_qubits, self.noise_model)
        
        elif backend == "cirq":
            from ftqec.core.quantum_sim.backends.cirq_backend import CirqBackend
            self.backend_name = "cirq"
            return CirqBackend(self.num_qubits)
        
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def reset(self):
        """Reset quantum state to |0...0⟩."""
        self.backend.reset()
    
    def get_statevector(self) -> np.ndarray:
        """
        Get current quantum statevector.
        
        Returns:
            Complex numpy array representing quantum state
        """
        return self.backend.get_statevector()
    
    def set_statevector(self, state: np.ndarray):
        """
        Set quantum statevector (if supported by backend).
        
        Args:
            state: Complex array of dimension 2^num_qubits
        """
        if hasattr(self.backend, 'set_statevector'):
            self.backend.set_statevector(state)
        else:
            raise NotImplementedError(f"Backend {self.backend_name} does not support set_statevector")
    
    def apply_gate(self, qubit_ids: List[int], gate_matrix: np.ndarray):
        """
        Apply quantum gate to specified qubits.
        
        Args:
            qubit_ids: List of target qubit indices
            gate_matrix: Unitary gate matrix
        """
        self.backend.apply_gate(qubit_ids, gate_matrix)
    
    def measure(self, qubit_ids: Optional[List[int]] = None) -> Union[int, List[int]]:
        """
        Measure specified qubits.
        
        Args:
            qubit_ids: Qubit indices to measure (None = measure all)
            
        Returns:
            Measurement result(s)
        """
        return self.backend.measure(qubit_ids)
    
    def entangle(self, qubit_ids: List[int]):
        """
        Create entanglement between specified qubits.
        
        Args:
            qubit_ids: List of qubit indices to entangle
        """
        self.backend.entangle(qubit_ids)
    
    def get_probabilities(self) -> Dict[str, float]:
        """
        Get measurement probabilities for all computational basis states.
        
        Returns:
            Dictionary mapping bitstrings to probabilities
        """
        return self.backend.get_probabilities()
    
    def get_coherence(self) -> float:
        """
        Calculate quantum coherence metric.
        
        Returns:
            Coherence value between 0 and 1
        """
        return self.backend.get_coherence()
    
    def get_density_matrix(self) -> np.ndarray:
        """
        Get density matrix representation (if supported).
        
        Returns:
            Density matrix as complex numpy array
        """
        if hasattr(self.backend, 'get_density_matrix'):
            return self.backend.get_density_matrix()
        else:
            # Compute from statevector
            state = self.get_statevector()
            return np.outer(state, state.conj())
    
    def run_circuit(self, shots: int = 1024) -> Dict[str, int]:
        """
        Run circuit multiple times and collect measurement statistics.
        
        Args:
            shots: Number of circuit executions
            
        Returns:
            Dictionary mapping bitstrings to counts
        """
        # Save current state
        original_state = self.get_statevector()
        
        counts = {}
        for _ in range(shots):
            # Restore state for each shot
            if hasattr(self.backend, 'set_statevector'):
                self.backend.set_statevector(original_state.copy())
            
            # Measure all qubits
            result = self.measure(list(range(self.num_qubits)))
            if isinstance(result, int):
                result = [result]
            
            bitstring = ''.join(map(str, result))
            counts[bitstring] = counts.get(bitstring, 0) + 1
        
        # Restore original state
        if hasattr(self.backend, 'set_statevector'):
            self.backend.set_statevector(original_state)
        
        return dict(sorted(counts.items()))
    
    def get_quantum_metrics(self) -> Dict[str, float]:
        """
        Get various quantum metrics for the current state.
        
        Returns:
            Dictionary of metric name to value
        """
        metrics = {
            'coherence': self.get_coherence(),
            'purity': self._calculate_purity(),
            'entanglement_entropy': self._calculate_entanglement_entropy(),
        }
        return metrics
    
    def _calculate_purity(self) -> float:
        """Calculate state purity Tr(ρ²)."""
        rho = self.get_density_matrix()
        purity = np.trace(rho @ rho)
        return float(np.real(purity))
    
    def _calculate_entanglement_entropy(self) -> float:
        """Calculate von Neumann entropy of the full system."""
        rho = self.get_density_matrix()
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
        return float(entropy)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"QuantumStateSim(num_qubits={self.num_qubits}, backend={self.backend_name})"
