"""
Native Complex Simulator - NumPy-based statevector simulation.

This backend provides efficient simulation for small quantum systems (≤20 qubits)
using NumPy's complex number support.
"""

import numpy as np
from typing import List, Union, Dict


class NativeComplexSim:
    """
    Native quantum simulator using NumPy complex arrays.
    
    Optimized for systems with ≤20 qubits. Uses statevector representation
    with complex amplitudes.
    """
    
    def __init__(self, num_qubits: int):
        """
        Initialize the native simulator.
        
        Args:
            num_qubits: Number of qubits in the system
        """
        if num_qubits > 20:
            print(f"Warning: {num_qubits} qubits may require significant memory")
        
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits
        
        # Initialize state to |0...0⟩
        self.statevector = np.zeros(self.dim, dtype=complex)
        self.statevector[0] = 1.0
    
    def reset(self):
        """Reset state to |0...0⟩."""
        self.statevector = np.zeros(self.dim, dtype=complex)
        self.statevector[0] = 1.0
    
    def get_statevector(self) -> np.ndarray:
        """Get current statevector."""
        return self.statevector.copy()
    
    def set_statevector(self, state: np.ndarray):
        """
        Set the statevector.
        
        Args:
            state: Complex array of dimension 2^num_qubits
        """
        if len(state) != self.dim:
            raise ValueError(f"State must have dimension {self.dim}")
        
        norm = np.linalg.norm(state)
        if norm == 0:
            raise ValueError("State cannot be zero vector")
        
        self.statevector = state / norm
    
    def apply_gate(self, qubit_ids: List[int], gate_matrix: np.ndarray):
        """
        Apply a quantum gate to specified qubits.
        
        Args:
            qubit_ids: List of qubit indices
            gate_matrix: Unitary gate matrix
        """
        # Build full operator
        full_operator = self._expand_operator(gate_matrix, qubit_ids)
        
        # Apply operator
        self.statevector = full_operator @ self.statevector
    
    def _expand_operator(self, operator: np.ndarray, target_qubits: List[int]) -> np.ndarray:
        """Expand operator to full Hilbert space."""
        if len(target_qubits) == self.num_qubits:
            return operator
        
        # For single or adjacent qubits, use tensor product
        if len(target_qubits) == 1:
            return self._expand_single_qubit_gate(operator, target_qubits[0])
        elif len(target_qubits) == 2 and abs(target_qubits[0] - target_qubits[1]) == 1:
            return self._expand_two_qubit_gate(operator, target_qubits)
        else:
            # General case for non-adjacent qubits
            return self._expand_general_gate(operator, target_qubits)
    
    def _expand_single_qubit_gate(self, gate: np.ndarray, qubit: int) -> np.ndarray:
        """Expand single-qubit gate to full space."""
        operators = []
        for i in range(self.num_qubits):
            if i == qubit:
                operators.append(gate)
            else:
                operators.append(np.eye(2, dtype=complex))
        
        result = operators[0]
        for op in operators[1:]:
            result = np.kron(result, op)
        
        return result
    
    def _expand_two_qubit_gate(self, gate: np.ndarray, qubits: List[int]) -> np.ndarray:
        """Expand two-qubit gate for adjacent qubits."""
        min_q = min(qubits)
        max_q = max(qubits)
        
        operators = []
        i = 0
        while i < self.num_qubits:
            if i == min_q:
                if qubits[0] < qubits[1]:
                    operators.append(gate)
                else:
                    # Swap control/target
                    swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
                    operators.append(swap @ gate @ swap)
                i += 2
            else:
                operators.append(np.eye(2, dtype=complex))
                i += 1
        
        result = operators[0]
        for op in operators[1:]:
            result = np.kron(result, op)
        
        return result
    
    def _expand_general_gate(self, gate: np.ndarray, qubits: List[int]) -> np.ndarray:
        """Expand gate for non-adjacent qubits."""
        # Build operator by iterating over computational basis
        dim = 2 ** self.num_qubits
        gate_dim = 2 ** len(qubits)
        full_gate = np.zeros((dim, dim), dtype=complex)
        
        for i in range(dim):
            # Extract target qubit values
            input_bits = [(i >> (self.num_qubits - 1 - q)) & 1 for q in qubits]
            gate_input_idx = sum(bit << (len(qubits) - 1 - j) for j, bit in enumerate(input_bits))
            
            # Apply gate
            for gate_output_idx in range(gate_dim):
                if abs(gate[gate_output_idx, gate_input_idx]) > 1e-10:
                    # Construct output state
                    output_bits = [(gate_output_idx >> (len(qubits) - 1 - j)) & 1 
                                   for j in range(len(qubits))]
                    
                    j = i
                    for qubit_idx, bit in zip(qubits, output_bits):
                        j &= ~(1 << (self.num_qubits - 1 - qubit_idx))
                        j |= (bit << (self.num_qubits - 1 - qubit_idx))
                    
                    full_gate[j, i] = gate[gate_output_idx, gate_input_idx]
        
        return full_gate
    
    def measure(self, qubit_ids: Union[List[int], None] = None) -> Union[int, List[int]]:
        """
        Measure specified qubits.
        
        Args:
            qubit_ids: List of qubit indices (None = measure all)
            
        Returns:
            Measurement result(s)
        """
        if qubit_ids is None:
            qubit_ids = list(range(self.num_qubits))
        
        if len(qubit_ids) == 1:
            return self._measure_single(qubit_ids[0])
        else:
            return self._measure_multiple(qubit_ids)
    
    def _measure_single(self, qubit: int) -> int:
        """Measure a single qubit."""
        # Calculate probability of measuring 0
        prob_0 = 0.0
        for i in range(self.dim):
            if not (i & (1 << (self.num_qubits - 1 - qubit))):
                prob_0 += np.abs(self.statevector[i]) ** 2
        
        # Measure
        outcome = 0 if np.random.random() < prob_0 else 1
        
        # Collapse state
        norm = 0.0
        for i in range(self.dim):
            bit = (i >> (self.num_qubits - 1 - qubit)) & 1
            if bit != outcome:
                self.statevector[i] = 0.0
            else:
                norm += np.abs(self.statevector[i]) ** 2
        
        if norm > 0:
            self.statevector /= np.sqrt(norm)
        
        return outcome
    
    def _measure_multiple(self, qubits: List[int]) -> List[int]:
        """Measure multiple qubits."""
        results = []
        for qubit in sorted(qubits):
            results.append(self._measure_single(qubit))
        return results
    
    def entangle(self, qubit_ids: List[int]):
        """
        Create entanglement between qubits using CNOT gates.
        
        Args:
            qubit_ids: List of qubit indices to entangle
        """
        if len(qubit_ids) < 2:
            raise ValueError("Need at least 2 qubits to entangle")
        
        # Apply CNOT gates to create entanglement
        cnot = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        
        for i in range(len(qubit_ids) - 1):
            self.apply_gate([qubit_ids[i], qubit_ids[i+1]], cnot)
    
    def get_probabilities(self) -> Dict[str, float]:
        """Get measurement probabilities for all basis states."""
        probs = {}
        for i in range(self.dim):
            prob = abs(self.statevector[i]) ** 2
            if prob > 1e-10:
                bitstring = format(i, f'0{self.num_qubits}b')
                probs[bitstring] = float(prob)
        return probs
    
    def get_density_matrix(self) -> np.ndarray:
        """Get the density matrix representation."""
        return np.outer(self.statevector, self.statevector.conj())
    
    def get_coherence(self) -> float:
        """
        Calculate quantum coherence metric.
        
        Returns:
            Coherence value between 0 and 1
        """
        # Use l1-norm of off-diagonal elements as coherence measure
        rho = self.get_density_matrix()
        coherence = 0.0
        for i in range(self.dim):
            for j in range(self.dim):
                if i != j:
                    coherence += abs(rho[i, j])
        
        # Normalize
        max_coherence = self.dim * (self.dim - 1)
        return float(coherence / max_coherence) if max_coherence > 0 else 0.0
    
    def __repr__(self) -> str:
        """String representation."""
        return f"NativeComplexSim(num_qubits={self.num_qubits})"
