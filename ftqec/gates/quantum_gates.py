"""
Quantum Gates - Standard quantum gate operations for FTQEC.

This module provides implementations of common quantum gates using tensor operations.
"""

import numpy as np


class QuantumGates:
    """Collection of standard quantum gates as unitary matrices."""
    
    @staticmethod
    def I() -> np.ndarray:
        """Identity gate."""
        return np.array([[1, 0], [0, 1]], dtype=complex)
    
    @staticmethod
    def X() -> np.ndarray:
        """Pauli-X gate (NOT gate)."""
        return np.array([[0, 1], [1, 0]], dtype=complex)
    
    @staticmethod
    def Y() -> np.ndarray:
        """Pauli-Y gate."""
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    
    @staticmethod
    def Z() -> np.ndarray:
        """Pauli-Z gate."""
        return np.array([[1, 0], [0, -1]], dtype=complex)
    
    @staticmethod
    def H() -> np.ndarray:
        """Hadamard gate."""
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    
    @staticmethod
    def S() -> np.ndarray:
        """S gate (phase gate)."""
        return np.array([[1, 0], [0, 1j]], dtype=complex)
    
    @staticmethod
    def T() -> np.ndarray:
        """T gate (π/8 gate)."""
        return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
    
    @staticmethod
    def RX(theta: float) -> np.ndarray:
        """
        Rotation around X-axis.
        
        Args:
            theta: Rotation angle in radians
        """
        return np.array([
            [np.cos(theta/2), -1j*np.sin(theta/2)],
            [-1j*np.sin(theta/2), np.cos(theta/2)]
        ], dtype=complex)
    
    @staticmethod
    def RY(theta: float) -> np.ndarray:
        """
        Rotation around Y-axis.
        
        Args:
            theta: Rotation angle in radians
        """
        return np.array([
            [np.cos(theta/2), -np.sin(theta/2)],
            [np.sin(theta/2), np.cos(theta/2)]
        ], dtype=complex)
    
    @staticmethod
    def RZ(theta: float) -> np.ndarray:
        """
        Rotation around Z-axis.
        
        Args:
            theta: Rotation angle in radians
        """
        return np.array([
            [np.exp(-1j*theta/2), 0],
            [0, np.exp(1j*theta/2)]
        ], dtype=complex)
    
    @staticmethod
    def CNOT() -> np.ndarray:
        """
        Controlled-NOT gate (2-qubit gate).
        
        Control qubit is first, target is second.
        """
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
    
    @staticmethod
    def CZ() -> np.ndarray:
        """Controlled-Z gate (2-qubit gate)."""
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=complex)
    
    @staticmethod
    def SWAP() -> np.ndarray:
        """SWAP gate (2-qubit gate)."""
        return np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=complex)
    
    @staticmethod
    def TOFFOLI() -> np.ndarray:
        """Toffoli gate (CCNOT, 3-qubit gate)."""
        gate = np.eye(8, dtype=complex)
        # Swap last two rows (|110⟩ and |111⟩)
        gate[6:8, 6:8] = np.array([[0, 1], [1, 0]], dtype=complex)
        return gate
    
    @staticmethod
    def FREDKIN() -> np.ndarray:
        """Fredkin gate (CSWAP, 3-qubit gate)."""
        gate = np.eye(8, dtype=complex)
        # Swap rows 5 and 6 (|101⟩ and |110⟩)
        gate[5:7, 5:7] = np.array([[0, 1], [1, 0]], dtype=complex)
        return gate
    
    @staticmethod
    def controlled_gate(gate: np.ndarray) -> np.ndarray:
        """
        Create a controlled version of a single-qubit gate.
        
        Args:
            gate: 2x2 single-qubit gate matrix
            
        Returns:
            4x4 controlled gate matrix
        """
        if gate.shape != (2, 2):
            raise ValueError("Gate must be 2x2 for single-qubit gates")
        
        controlled = np.eye(4, dtype=complex)
        controlled[2:4, 2:4] = gate
        return controlled


def tensor_product(*operators: np.ndarray) -> np.ndarray:
    """
    Compute tensor product (Kronecker product) of operators.
    
    Args:
        *operators: Variable number of operator matrices
        
    Returns:
        Tensor product of all operators
    """
    if not operators:
        raise ValueError("At least one operator required")
    
    result = operators[0]
    for op in operators[1:]:
        result = np.kron(result, op)
    
    return result
