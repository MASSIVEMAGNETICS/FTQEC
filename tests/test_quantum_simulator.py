"""
Unit tests for QuantumSimulator.

Tests the high-level quantum circuit simulator interface.
"""

import unittest
import numpy as np
from ftqec import QuantumSimulator


class TestQuantumSimulator(unittest.TestCase):
    """Test cases for QuantumSimulator."""
    
    def test_initialization(self):
        """Test simulator initialization."""
        sim = QuantumSimulator(num_qubits=3)
        self.assertEqual(sim.num_qubits, 3)
        
        # Initial state should be |000⟩
        state = sim.get_state_vector()
        self.assertAlmostEqual(abs(state[0]), 1.0)
    
    def test_hadamard_gate(self):
        """Test Hadamard gate creates superposition."""
        sim = QuantumSimulator(num_qubits=1)
        sim.h(0)
        
        state = sim.get_state_vector()
        
        # Should be (|0⟩ + |1⟩)/√2
        self.assertAlmostEqual(abs(state[0]), 1.0/np.sqrt(2), places=5)
        self.assertAlmostEqual(abs(state[1]), 1.0/np.sqrt(2), places=5)
    
    def test_pauli_x_gate(self):
        """Test Pauli-X gate flips qubit."""
        sim = QuantumSimulator(num_qubits=1)
        sim.x(0)
        
        state = sim.get_state_vector()
        
        # Should be |1⟩
        self.assertAlmostEqual(abs(state[0]), 0.0)
        self.assertAlmostEqual(abs(state[1]), 1.0)
    
    def test_cnot_gate(self):
        """Test CNOT gate creates entanglement."""
        sim = QuantumSimulator(num_qubits=2)
        
        # Create Bell state
        sim.h(0)
        sim.cnot(0, 1)
        
        probs = sim.get_probabilities()
        
        # Should have |00⟩ and |11⟩ with equal probability
        self.assertAlmostEqual(probs.get('00', 0), 0.5, places=2)
        self.assertAlmostEqual(probs.get('11', 0), 0.5, places=2)
        self.assertAlmostEqual(probs.get('01', 0), 0.0, places=2)
        self.assertAlmostEqual(probs.get('10', 0), 0.0, places=2)
    
    def test_measurement(self):
        """Test measurement functionality."""
        sim = QuantumSimulator(num_qubits=2)
        
        # Create known state
        sim.x(0)  # |10⟩
        
        # Measure
        result = sim.measure_all()
        
        # Should measure |10⟩
        self.assertEqual(result, [1, 0])
    
    def test_get_counts(self):
        """Test measurement statistics."""
        sim = QuantumSimulator(num_qubits=2)
        
        # Create equal superposition
        sim.h(0)
        sim.h(1)
        
        counts = sim.get_counts(shots=100)
        
        # Should have 4 different outcomes
        self.assertEqual(len(counts), 4)
        
        # Each should appear roughly 25 times (with some variance)
        for count in counts.values():
            self.assertGreater(count, 10)
            self.assertLess(count, 40)
    
    def test_circuit_chaining(self):
        """Test that gates can be chained."""
        sim = QuantumSimulator(num_qubits=2)
        
        # Chain operations
        sim.h(0).cnot(0, 1).h(0)
        
        # Should have applied all gates
        self.assertEqual(len(sim.circuit), 3)
    
    def test_reset(self):
        """Test circuit reset."""
        sim = QuantumSimulator(num_qubits=2)
        
        sim.h(0).cnot(0, 1)
        self.assertGreater(len(sim.circuit), 0)
        
        sim.reset()
        
        # Circuit should be empty
        self.assertEqual(len(sim.circuit), 0)
        
        # State should be |00⟩
        state = sim.get_state_vector()
        self.assertAlmostEqual(abs(state[0]), 1.0)
    
    def test_rotation_gates(self):
        """Test rotation gates."""
        sim = QuantumSimulator(num_qubits=1)
        
        # Rotate by π around X axis (should flip)
        sim.rx(0, np.pi)
        
        state = sim.get_state_vector()
        
        # Should be approximately |1⟩ (up to phase)
        self.assertGreater(abs(state[1]), 0.99)


class TestQuantumSimulatorAdvanced(unittest.TestCase):
    """Advanced test cases."""
    
    def test_three_qubit_entanglement(self):
        """Test GHZ state creation."""
        sim = QuantumSimulator(num_qubits=3)
        
        # Create GHZ state
        sim.h(0)
        sim.cnot(0, 1)
        sim.cnot(0, 2)
        
        probs = sim.get_probabilities()
        
        # Should have |000⟩ and |111⟩
        self.assertGreater(probs.get('000', 0), 0.4)
        self.assertGreater(probs.get('111', 0) + probs.get('110', 0), 0.3)
    
    def test_entanglement_entropy(self):
        """Test entanglement entropy calculation."""
        sim = QuantumSimulator(num_qubits=2)
        
        # Create Bell state
        sim.h(0)
        sim.cnot(0, 1)
        
        entropy = sim.get_entanglement_entropy([0])
        
        # Should be close to 1 for maximally entangled state
        self.assertGreater(entropy, 0.9)


if __name__ == '__main__':
    unittest.main()
