"""
Cirq Backend - Advanced circuit compilation via Google Cirq.

This backend provides access to Cirq's quantum simulators and circuit optimization.
Requires cirq to be installed.
"""

import numpy as np
from typing import List, Union, Dict


class CirqBackend:
    """
    Cirq-based quantum simulator backend.
    
    Provides advanced circuit compilation and optimization via Google Cirq.
    Falls back to native simulation if Cirq is not available.
    """
    
    def __init__(self, num_qubits: int):
        """
        Initialize Cirq backend.
        
        Args:
            num_qubits: Number of qubits
        """
        self.num_qubits = num_qubits
        self._cirq_available = False
        
        try:
            import cirq
            
            self._cirq_available = True
            self.qubits = [cirq.LineQubit(i) for i in range(num_qubits)]
            self.circuit = cirq.Circuit()
            self.simulator = cirq.Simulator()
            
        except ImportError:
            print("Warning: Cirq not available, using fallback native simulation")
            from ftqec.core.quantum_sim.backends.native_complex_sim import NativeComplexSim
            self._fallback = NativeComplexSim(num_qubits)
    
    def reset(self):
        """Reset the quantum state."""
        if self._cirq_available:
            import cirq
            self.circuit = cirq.Circuit()
        else:
            self._fallback.reset()
    
    def get_statevector(self) -> np.ndarray:
        """Get current statevector."""
        if self._cirq_available:
            result = self.simulator.simulate(self.circuit)
            return result.final_state_vector
        else:
            return self._fallback.get_statevector()
    
    def apply_gate(self, qubit_ids: List[int], gate_matrix: np.ndarray):
        """Apply quantum gate."""
        if self._cirq_available:
            import cirq
            qubits = [self.qubits[i] for i in qubit_ids]
            gate = cirq.MatrixGate(gate_matrix)
            self.circuit.append(gate.on(*qubits))
        else:
            self._fallback.apply_gate(qubit_ids, gate_matrix)
    
    def measure(self, qubit_ids: Union[List[int], None] = None) -> Union[int, List[int]]:
        """Measure qubits."""
        if self._cirq_available:
            # Get statevector and perform measurement classically
            state = self.get_statevector()
            probs = np.abs(state) ** 2
            outcome = np.random.choice(len(state), p=probs)
            
            if qubit_ids is None:
                qubit_ids = list(range(self.num_qubits))
            
            results = []
            for qubit in qubit_ids:
                bit = (outcome >> (self.num_qubits - 1 - qubit)) & 1
                results.append(bit)
            
            return results[0] if len(results) == 1 else results
        else:
            return self._fallback.measure(qubit_ids)
    
    def entangle(self, qubit_ids: List[int]):
        """Create entanglement."""
        if self._cirq_available:
            import cirq
            for i in range(len(qubit_ids) - 1):
                self.circuit.append(cirq.CNOT(self.qubits[qubit_ids[i]], 
                                             self.qubits[qubit_ids[i+1]]))
        else:
            self._fallback.entangle(qubit_ids)
    
    def get_probabilities(self) -> Dict[str, float]:
        """Get measurement probabilities."""
        if self._cirq_available:
            state = self.get_statevector()
            probs = {}
            for i, amplitude in enumerate(state):
                prob = abs(amplitude) ** 2
                if prob > 1e-10:
                    bitstring = format(i, f'0{self.num_qubits}b')
                    probs[bitstring] = float(prob)
            return probs
        else:
            return self._fallback.get_probabilities()
    
    def get_coherence(self) -> float:
        """Calculate quantum coherence."""
        if self._cirq_available:
            state = self.get_statevector()
            rho = np.outer(state, state.conj())
            coherence = 0.0
            dim = len(state)
            for i in range(dim):
                for j in range(dim):
                    if i != j:
                        coherence += abs(rho[i, j])
            max_coherence = dim * (dim - 1)
            return float(coherence / max_coherence) if max_coherence > 0 else 0.0
        else:
            return self._fallback.get_coherence()
    
    def __repr__(self) -> str:
        """String representation."""
        backend_type = "Cirq" if self._cirq_available else "Fallback"
        return f"CirqBackend(num_qubits={self.num_qubits}, type={backend_type})"
