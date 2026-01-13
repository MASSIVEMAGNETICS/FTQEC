"""
Quantum Simulator - Main interface for quantum circuit simulation using FTQEC.

This module provides a high-level interface for building and simulating
quantum circuits using the fractal state engine.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
from ftqec.core.fractal_state_engine import FractalStateEngine
from ftqec.gates.quantum_gates import QuantumGates, tensor_product


class QuantumSimulator:
    """
    Quantum circuit simulator using fractal state engines.
    
    This simulator supports 3-5 qubits and provides an accessible interface
    for quantum computing without requiring expensive quantum hardware.
    
    Attributes:
        num_qubits: Number of qubits in the system
        engine: Fractal state engine for state representation
        circuit: List of quantum operations
    """
    
    def __init__(self, num_qubits: int, fractal_depth: int = 2):
        """
        Initialize the quantum simulator.
        
        Args:
            num_qubits: Number of qubits (3-5 recommended)
            fractal_depth: Depth of fractal decomposition
        """
        if num_qubits < 3 or num_qubits > 5:
            print(f"Warning: This simulator is optimized for 3-5 qubits. Using {num_qubits} qubits.")
        
        self.num_qubits = num_qubits
        self.engine = FractalStateEngine(num_qubits, fractal_depth)
        self.circuit = []
        self.gates = QuantumGates()
        
    def reset(self):
        """Reset the quantum state and clear the circuit."""
        self.engine.reset()
        self.circuit = []
    
    def get_state_vector(self) -> np.ndarray:
        """Get the current quantum state vector."""
        return self.engine.get_state()
    
    def get_statevector(self) -> np.ndarray:
        """Alias for get_state_vector for compatibility."""
        return self.get_state_vector()
    
    # Single-qubit gates
    def h(self, qubit: int):
        """
        Apply Hadamard gate to qubit.
        
        Args:
            qubit: Target qubit index
        """
        self._apply_single_qubit_gate(self.gates.H(), qubit, "H")
        return self
    
    def x(self, qubit: int):
        """
        Apply Pauli-X gate to qubit.
        
        Args:
            qubit: Target qubit index
        """
        self._apply_single_qubit_gate(self.gates.X(), qubit, "X")
        return self
    
    def y(self, qubit: int):
        """
        Apply Pauli-Y gate to qubit.
        
        Args:
            qubit: Target qubit index
        """
        self._apply_single_qubit_gate(self.gates.Y(), qubit, "Y")
        return self
    
    def z(self, qubit: int):
        """
        Apply Pauli-Z gate to qubit.
        
        Args:
            qubit: Target qubit index
        """
        self._apply_single_qubit_gate(self.gates.Z(), qubit, "Z")
        return self
    
    def s(self, qubit: int):
        """
        Apply S gate to qubit.
        
        Args:
            qubit: Target qubit index
        """
        self._apply_single_qubit_gate(self.gates.S(), qubit, "S")
        return self
    
    def t(self, qubit: int):
        """
        Apply T gate to qubit.
        
        Args:
            qubit: Target qubit index
        """
        self._apply_single_qubit_gate(self.gates.T(), qubit, "T")
        return self
    
    def rx(self, qubit: int, theta: float):
        """
        Apply RX rotation gate to qubit.
        
        Args:
            qubit: Target qubit index
            theta: Rotation angle in radians
        """
        self._apply_single_qubit_gate(self.gates.RX(theta), qubit, f"RX({theta:.3f})")
        return self
    
    def ry(self, qubit: int, theta: float):
        """
        Apply RY rotation gate to qubit.
        
        Args:
            qubit: Target qubit index
            theta: Rotation angle in radians
        """
        self._apply_single_qubit_gate(self.gates.RY(theta), qubit, f"RY({theta:.3f})")
        return self
    
    def rz(self, qubit: int, theta: float):
        """
        Apply RZ rotation gate to qubit.
        
        Args:
            qubit: Target qubit index
            theta: Rotation angle in radians
        """
        self._apply_single_qubit_gate(self.gates.RZ(theta), qubit, f"RZ({theta:.3f})")
        return self
    
    # Two-qubit gates
    def cnot(self, control: int, target: int):
        """
        Apply CNOT gate.
        
        Args:
            control: Control qubit index
            target: Target qubit index
        """
        self._apply_two_qubit_gate(self.gates.CNOT(), control, target, "CNOT")
        return self
    
    def cx(self, control: int, target: int):
        """Alias for CNOT gate."""
        return self.cnot(control, target)
    
    def cz(self, control: int, target: int):
        """
        Apply CZ gate.
        
        Args:
            control: Control qubit index
            target: Target qubit index
        """
        self._apply_two_qubit_gate(self.gates.CZ(), control, target, "CZ")
        return self
    
    def swap(self, qubit1: int, qubit2: int):
        """
        Apply SWAP gate.
        
        Args:
            qubit1: First qubit index
            qubit2: Second qubit index
        """
        self._apply_two_qubit_gate(self.gates.SWAP(), qubit1, qubit2, "SWAP")
        return self
    
    def _apply_single_qubit_gate(self, gate: np.ndarray, qubit: int, name: str):
        """Apply a single-qubit gate to the specified qubit."""
        if qubit < 0 or qubit >= self.num_qubits:
            raise ValueError(f"Qubit index must be between 0 and {self.num_qubits-1}")
        
        # Build full operator using tensor products
        operators = []
        for i in range(self.num_qubits):
            if i == qubit:
                operators.append(gate)
            else:
                operators.append(self.gates.I())
        
        full_gate = tensor_product(*operators)
        self.engine.apply_operator(full_gate)
        self.circuit.append({"gate": name, "qubits": [qubit]})
    
    def _apply_two_qubit_gate(self, gate: np.ndarray, qubit1: int, qubit2: int, name: str):
        """Apply a two-qubit gate."""
        if qubit1 < 0 or qubit1 >= self.num_qubits:
            raise ValueError(f"Qubit index must be between 0 and {self.num_qubits-1}")
        if qubit2 < 0 or qubit2 >= self.num_qubits:
            raise ValueError(f"Qubit index must be between 0 and {self.num_qubits-1}")
        if qubit1 == qubit2:
            raise ValueError("Qubit indices must be different")
        
        # For adjacent qubits
        if abs(qubit1 - qubit2) == 1:
            min_q = min(qubit1, qubit2)
            max_q = max(qubit1, qubit2)
            
            operators = []
            for i in range(self.num_qubits):
                if i == min_q:
                    # Include the 2-qubit gate here and skip next iteration
                    if qubit1 < qubit2:
                        operators.append(gate)
                    else:
                        # Swap control and target
                        swap = self.gates.SWAP()
                        swapped_gate = swap @ gate @ swap
                        operators.append(swapped_gate)
                elif i == max_q:
                    # Already included in previous iteration
                    continue
                else:
                    operators.append(self.gates.I())
            
            # Special handling when gate spans qubits
            if max_q - min_q == 1:
                operators_clean = []
                i = 0
                while i < self.num_qubits:
                    if i == min_q:
                        if qubit1 < qubit2:
                            operators_clean.append(gate)
                        else:
                            swap = self.gates.SWAP()
                            swapped_gate = swap @ gate @ swap
                            operators_clean.append(swapped_gate)
                        i += 2  # Skip next qubit as it's part of the 2-qubit gate
                    else:
                        operators_clean.append(self.gates.I())
                        i += 1
                
                full_gate = tensor_product(*operators_clean)
            else:
                full_gate = tensor_product(*operators)
        else:
            # For non-adjacent qubits, use SWAP gates to make them adjacent
            # This is a simplified approach
            full_gate = self._build_non_adjacent_gate(gate, qubit1, qubit2)
        
        self.engine.apply_operator(full_gate)
        self.circuit.append({"gate": name, "qubits": [qubit1, qubit2]})
    
    def _build_non_adjacent_gate(self, gate: np.ndarray, q1: int, q2: int) -> np.ndarray:
        """Build operator for non-adjacent qubits."""
        # Simplified: Create identity on all qubits except target pair
        # More efficient implementation would use SWAP network
        dim = 2 ** self.num_qubits
        full_gate = np.eye(dim, dtype=complex)
        
        # Apply gate to computational basis states
        for i in range(dim):
            # Extract bits for q1 and q2
            bit_q1 = (i >> (self.num_qubits - 1 - q1)) & 1
            bit_q2 = (i >> (self.num_qubits - 1 - q2)) & 1
            
            # Create 2-bit index for the gate
            gate_input_idx = (bit_q1 << 1) | bit_q2
            
            # Apply gate to find output bits
            for j in range(4):
                if abs(gate[j, gate_input_idx]) > 1e-10:
                    out_bit_q1 = (j >> 1) & 1
                    out_bit_q2 = j & 1
                    
                    # Construct output state index
                    output_idx = i
                    output_idx &= ~(1 << (self.num_qubits - 1 - q1))
                    output_idx &= ~(1 << (self.num_qubits - 1 - q2))
                    output_idx |= (out_bit_q1 << (self.num_qubits - 1 - q1))
                    output_idx |= (out_bit_q2 << (self.num_qubits - 1 - q2))
                    
                    full_gate[output_idx, i] = gate[j, gate_input_idx]
        
        # Re-normalize to fix any numerical errors
        for col in range(dim):
            norm = np.linalg.norm(full_gate[:, col])
            if norm > 1e-10:
                full_gate[:, col] /= norm
        
        return full_gate
    
    def measure(self, qubit: Optional[int] = None) -> int | List[int]:
        """
        Measure qubit(s).
        
        Args:
            qubit: Qubit index to measure (None = measure all)
            
        Returns:
            Measurement result (0 or 1 for single qubit, list for all)
        """
        result = self.engine.measure(qubit)
        self.circuit.append({"gate": "MEASURE", "qubits": [qubit] if qubit is not None else list(range(self.num_qubits))})
        return result
    
    def measure_all(self) -> List[int]:
        """Measure all qubits."""
        return self.measure(None)
    
    def get_counts(self, shots: int = 1024) -> Dict[str, int]:
        """
        Run circuit multiple times and get measurement statistics.
        
        Args:
            shots: Number of circuit executions
            
        Returns:
            Dictionary mapping bitstrings to counts
        """
        # Save current state
        original_state = self.engine.get_state()
        
        counts = {}
        for _ in range(shots):
            # Restore state for each shot
            self.engine.set_state(original_state)
            
            # Measure all qubits
            result = self.measure_all()
            bitstring = ''.join(map(str, result))
            counts[bitstring] = counts.get(bitstring, 0) + 1
        
        # Restore original state
        self.engine.set_state(original_state)
        
        return dict(sorted(counts.items()))
    
    def get_probabilities(self) -> Dict[str, float]:
        """
        Get measurement probabilities for all basis states.
        
        Returns:
            Dictionary mapping bitstrings to probabilities
        """
        probs = self.engine.get_probabilities()
        result = {}
        
        for i, prob in enumerate(probs):
            if prob > 1e-10:  # Only include non-zero probabilities
                bitstring = format(i, f'0{self.num_qubits}b')
                result[bitstring] = float(prob)
        
        return result
    
    def get_entanglement_entropy(self, subsystem: List[int]) -> float:
        """
        Calculate entanglement entropy of subsystem.
        
        Args:
            subsystem: List of qubit indices
            
        Returns:
            Von Neumann entropy
        """
        return self.engine.entanglement_entropy(subsystem)
    
    def print_circuit(self):
        """Print a text representation of the circuit."""
        print(f"Quantum Circuit ({self.num_qubits} qubits):")
        print("-" * 50)
        for i, op in enumerate(self.circuit):
            gate = op['gate']
            qubits = op['qubits']
            qubit_str = ', '.join(map(str, qubits))
            print(f"  {i+1}. {gate} on qubit(s): {qubit_str}")
        print("-" * 50)
    
    def print_state(self, threshold: float = 1e-3):
        """
        Print the current quantum state in Dirac notation.
        
        Args:
            threshold: Minimum amplitude to display
        """
        print("Quantum State:")
        state = self.get_state_vector()
        
        terms = []
        for i, amplitude in enumerate(state):
            if abs(amplitude) > threshold:
                bitstring = format(i, f'0{self.num_qubits}b')
                real = amplitude.real
                imag = amplitude.imag
                
                # Format amplitude
                if abs(imag) < 1e-10:
                    amp_str = f"{real:.3f}"
                elif abs(real) < 1e-10:
                    amp_str = f"{imag:.3f}i"
                else:
                    sign = '+' if imag > 0 else ''
                    amp_str = f"{real:.3f}{sign}{imag:.3f}i"
                
                terms.append(f"{amp_str}|{bitstring}⟩")
        
        if terms:
            print("  " + " + ".join(terms))
        else:
            print("  (state below threshold)")
    
    def __repr__(self) -> str:
        """String representation."""
        return f"QuantumSimulator(num_qubits={self.num_qubits}, circuit_depth={len(self.circuit)})"
