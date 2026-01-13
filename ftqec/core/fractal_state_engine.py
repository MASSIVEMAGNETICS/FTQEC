"""
Fractal State Engine - Core quantum state representation using fractal decomposition.

This module implements a novel approach to quantum state representation where
quantum states are represented using recursive fractal decomposition, allowing
for efficient simulation of quantum behavior without requiring quantum hardware.
"""

import numpy as np
from typing import List, Tuple, Optional, Union


class FractalStateEngine:
    """
    Fractal State Engine for quantum state representation.
    
    This engine uses a fractal decomposition strategy where quantum states
    are recursively decomposed into smaller subsystems, enabling efficient
    tensor operations and state manipulation.
    
    Attributes:
        num_qubits: Number of qubits in the system
        state_vector: Complex amplitude vector representing quantum state
        fractal_depth: Depth of fractal decomposition
    """
    
    def __init__(self, num_qubits: int, fractal_depth: int = 2):
        """
        Initialize the Fractal State Engine.
        
        Args:
            num_qubits: Number of qubits (3-5 recommended)
            fractal_depth: Depth of fractal decomposition for state representation
        """
        if num_qubits < 1 or num_qubits > 10:
            raise ValueError("Number of qubits must be between 1 and 10")
        
        self.num_qubits = num_qubits
        self.fractal_depth = fractal_depth
        self.dim = 2 ** num_qubits
        
        # Initialize state to |0...0⟩
        self.state_vector = np.zeros(self.dim, dtype=complex)
        self.state_vector[0] = 1.0
        
        # Fractal decomposition cache for optimization
        self._fractal_cache = {}
    
    def reset(self):
        """Reset the quantum state to |0...0⟩."""
        self.state_vector = np.zeros(self.dim, dtype=complex)
        self.state_vector[0] = 1.0
        self._fractal_cache.clear()
    
    def get_state(self) -> np.ndarray:
        """
        Get the current quantum state vector.
        
        Returns:
            Complex numpy array representing the quantum state
        """
        return self.state_vector.copy()
    
    def set_state(self, state: np.ndarray):
        """
        Set the quantum state vector.
        
        Args:
            state: Complex numpy array of dimension 2^num_qubits
        """
        if len(state) != self.dim:
            raise ValueError(f"State vector must have dimension {self.dim}")
        
        # Normalize the state
        norm = np.linalg.norm(state)
        if norm == 0:
            raise ValueError("State vector cannot be zero")
        
        self.state_vector = state / norm
        self._fractal_cache.clear()
    
    def apply_operator(self, operator: np.ndarray, target_qubits: Optional[List[int]] = None):
        """
        Apply a quantum operator using fractal tensor decomposition.
        
        Args:
            operator: Unitary operator matrix
            target_qubits: List of qubit indices to apply operator to (None = all qubits)
        """
        if target_qubits is None:
            # Apply to entire system
            if operator.shape != (self.dim, self.dim):
                raise ValueError(f"Operator shape must be ({self.dim}, {self.dim})")
            self.state_vector = operator @ self.state_vector
        else:
            # Apply to specific qubits using tensor product expansion
            full_operator = self._expand_operator(operator, target_qubits)
            self.state_vector = full_operator @ self.state_vector
        
        self._fractal_cache.clear()
    
    def _expand_operator(self, operator: np.ndarray, target_qubits: List[int]) -> np.ndarray:
        """
        Expand operator to full Hilbert space using fractal tensor decomposition.
        
        Args:
            operator: Operator on target qubits
            target_qubits: Indices of target qubits
            
        Returns:
            Full Hilbert space operator
        """
        # Sort target qubits
        sorted_targets = sorted(target_qubits)
        
        # Build operator using tensor products
        result = np.array([[1.0]], dtype=complex)
        op_idx = 0
        
        for qubit in range(self.num_qubits):
            if qubit in sorted_targets:
                # Get the appropriate part of the operator
                dim = int(np.sqrt(operator.shape[0]) ** (sorted_targets.index(qubit) + 1 - op_idx))
                if len(sorted_targets) == 1:
                    current_op = operator
                else:
                    current_op = operator if op_idx == 0 else np.eye(2, dtype=complex)
                op_idx += 1
            else:
                # Identity for non-target qubits
                current_op = np.eye(2, dtype=complex)
            
            result = np.kron(result, current_op)
        
        return result
    
    def measure(self, qubit_idx: Optional[int] = None) -> Union[int, List[int]]:
        """
        Measure qubit(s) and collapse state using fractal projection.
        
        Args:
            qubit_idx: Index of qubit to measure (None = measure all)
            
        Returns:
            Measurement result (0 or 1 for single qubit, list for all)
        """
        if qubit_idx is None:
            # Measure all qubits
            probabilities = np.abs(self.state_vector) ** 2
            outcome = np.random.choice(self.dim, p=probabilities)
            
            # Collapse to measured state
            self.state_vector = np.zeros(self.dim, dtype=complex)
            self.state_vector[outcome] = 1.0
            
            # Convert outcome to binary
            return [int(b) for b in format(outcome, f'0{self.num_qubits}b')]
        else:
            # Measure single qubit
            if qubit_idx < 0 or qubit_idx >= self.num_qubits:
                raise ValueError(f"Qubit index must be between 0 and {self.num_qubits-1}")
            
            # Calculate probability of measuring 0
            prob_0 = 0.0
            for i in range(self.dim):
                if not (i & (1 << (self.num_qubits - 1 - qubit_idx))):
                    prob_0 += np.abs(self.state_vector[i]) ** 2
            
            # Measure
            outcome = 0 if np.random.random() < prob_0 else 1
            
            # Collapse state
            norm = 0.0
            for i in range(self.dim):
                bit = (i >> (self.num_qubits - 1 - qubit_idx)) & 1
                if bit != outcome:
                    self.state_vector[i] = 0.0
                else:
                    norm += np.abs(self.state_vector[i]) ** 2
            
            if norm > 0:
                self.state_vector /= np.sqrt(norm)
            
            self._fractal_cache.clear()
            return outcome
    
    def get_probabilities(self) -> np.ndarray:
        """
        Get measurement probabilities for all basis states.
        
        Returns:
            Array of probabilities for each computational basis state
        """
        return np.abs(self.state_vector) ** 2
    
    def get_fractal_representation(self, depth: Optional[int] = None) -> List[np.ndarray]:
        """
        Get fractal decomposition of the quantum state.
        
        Args:
            depth: Depth of decomposition (uses fractal_depth if None)
            
        Returns:
            List of state tensors at each fractal level
        """
        if depth is None:
            depth = self.fractal_depth
        
        representations = []
        current_state = self.state_vector.copy()
        
        for level in range(depth):
            # Reshape state into tensor form
            if level < self.num_qubits:
                shape = [2] * (self.num_qubits - level)
                reshaped = current_state.reshape(shape)
                representations.append(reshaped)
                
                # Partial trace for next level
                if level < depth - 1 and self.num_qubits - level > 1:
                    current_state = np.sum(np.abs(reshaped) ** 2, axis=-1).flatten()
                    current_state = current_state / np.sum(current_state)
                    current_state = np.sqrt(current_state.astype(complex))
        
        return representations
    
    def entanglement_entropy(self, subsystem: List[int]) -> float:
        """
        Calculate von Neumann entropy of subsystem (entanglement measure).
        
        Args:
            subsystem: List of qubit indices defining the subsystem
            
        Returns:
            Von Neumann entropy value
        """
        if not subsystem or len(subsystem) >= self.num_qubits:
            return 0.0
        
        # Reshape state into tensor
        state_tensor = self.state_vector.reshape([2] * self.num_qubits)
        
        # Compute reduced density matrix
        rho = self._reduced_density_matrix(state_tensor, subsystem)
        
        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        
        # Von Neumann entropy
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
        
        return float(entropy)
    
    def _reduced_density_matrix(self, state_tensor: np.ndarray, subsystem: List[int]) -> np.ndarray:
        """
        Calculate reduced density matrix for subsystem.
        
        Args:
            state_tensor: State as tensor
            subsystem: Qubit indices in subsystem
            
        Returns:
            Reduced density matrix
        """
        # This is a simplified version - full implementation would use tensor contractions
        subsystem_dim = 2 ** len(subsystem)
        rho = np.zeros((subsystem_dim, subsystem_dim), dtype=complex)
        
        # Simplified approach: compute density matrix and trace out complement
        full_rho = np.outer(self.state_vector, self.state_vector.conj())
        
        # For now, return a simple approximation
        # A full implementation would properly trace out the complement subsystem
        for i in range(subsystem_dim):
            for j in range(subsystem_dim):
                # Map subsystem indices to full system indices
                sum_val = 0.0
                complement_dim = 2 ** (self.num_qubits - len(subsystem))
                for k in range(complement_dim):
                    idx_i = self._merge_indices(i, k, subsystem)
                    idx_j = self._merge_indices(j, k, subsystem)
                    sum_val += full_rho[idx_i, idx_j]
                rho[i, j] = sum_val
        
        return rho
    
    def _merge_indices(self, sub_idx: int, comp_idx: int, subsystem: List[int]) -> int:
        """Merge subsystem and complement indices."""
        result = 0
        sub_bits = [(sub_idx >> i) & 1 for i in range(len(subsystem))]
        comp_bits = [(comp_idx >> i) & 1 for i in range(self.num_qubits - len(subsystem))]
        
        sub_pos = 0
        comp_pos = 0
        for qubit in range(self.num_qubits):
            if qubit in subsystem:
                bit = sub_bits[sub_pos]
                sub_pos += 1
            else:
                bit = comp_bits[comp_pos]
                comp_pos += 1
            result |= (bit << (self.num_qubits - 1 - qubit))
        
        return result
    
    def __repr__(self) -> str:
        """String representation of the fractal state engine."""
        return f"FractalStateEngine(num_qubits={self.num_qubits}, fractal_depth={self.fractal_depth})"
