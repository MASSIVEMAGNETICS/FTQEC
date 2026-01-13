"""
Unit tests for FractalStateEngine.

Tests the core fractal state representation and manipulation.
"""

import unittest
import numpy as np
from ftqec.core.fractal_state_engine import FractalStateEngine


class TestFractalStateEngine(unittest.TestCase):
    """Test cases for FractalStateEngine."""
    
    def test_initialization(self):
        """Test engine initialization."""
        engine = FractalStateEngine(num_qubits=3)
        self.assertEqual(engine.num_qubits, 3)
        self.assertEqual(engine.dim, 8)
        
        # Check initial state is |000⟩
        state = engine.get_state()
        self.assertEqual(len(state), 8)
        self.assertAlmostEqual(abs(state[0]), 1.0)
        self.assertAlmostEqual(np.sum(np.abs(state[1:])), 0.0)
    
    def test_reset(self):
        """Test state reset functionality."""
        engine = FractalStateEngine(num_qubits=2)
        
        # Modify state
        new_state = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)
        engine.set_state(new_state)
        
        # Reset
        engine.reset()
        
        # Check back to |00⟩
        state = engine.get_state()
        self.assertAlmostEqual(abs(state[0]), 1.0)
        self.assertAlmostEqual(np.sum(np.abs(state[1:])), 0.0)
    
    def test_state_normalization(self):
        """Test that states are properly normalized."""
        engine = FractalStateEngine(num_qubits=2)
        
        # Set unnormalized state
        unnormalized = np.array([1.0, 1.0, 1.0, 1.0], dtype=complex)
        engine.set_state(unnormalized)
        
        # Check normalization
        state = engine.get_state()
        norm = np.linalg.norm(state)
        self.assertAlmostEqual(norm, 1.0)
    
    def test_measurement_probabilities(self):
        """Test probability calculation."""
        engine = FractalStateEngine(num_qubits=2)
        
        # Equal superposition
        state = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)
        engine.set_state(state)
        
        probs = engine.get_probabilities()
        
        # All should be 0.25
        for prob in probs:
            self.assertAlmostEqual(prob, 0.25)
    
    def test_measurement_collapse(self):
        """Test that measurement collapses state."""
        engine = FractalStateEngine(num_qubits=2)
        
        # Create superposition
        state = np.array([0.707, 0.707, 0.0, 0.0], dtype=complex)
        engine.set_state(state)
        
        # Measure
        result = engine.measure()
        
        # State should be collapsed
        final_state = engine.get_state()
        non_zero = np.sum(np.abs(final_state) > 0.1)
        self.assertEqual(non_zero, 1)  # Only one basis state
    
    def test_entanglement_entropy(self):
        """Test entanglement entropy calculation."""
        engine = FractalStateEngine(num_qubits=2)
        
        # Separable state |00⟩
        engine.reset()
        entropy_sep = engine.entanglement_entropy([0])
        self.assertAlmostEqual(entropy_sep, 0.0, places=5)
        
        # Bell state (maximally entangled)
        bell = np.array([0.707, 0.0, 0.0, 0.707], dtype=complex)
        engine.set_state(bell)
        entropy_ent = engine.entanglement_entropy([0])
        self.assertGreater(entropy_ent, 0.9)  # Should be close to 1
    
    def test_fractal_representation(self):
        """Test fractal decomposition."""
        engine = FractalStateEngine(num_qubits=3, fractal_depth=2)
        
        representations = engine.get_fractal_representation()
        
        # Should have representations at each level
        self.assertGreater(len(representations), 0)
        self.assertIsInstance(representations[0], np.ndarray)


class TestFractalStateEngineOperators(unittest.TestCase):
    """Test operator application."""
    
    def test_identity_operator(self):
        """Test identity operator doesn't change state."""
        engine = FractalStateEngine(num_qubits=2)
        
        identity = np.eye(4, dtype=complex)
        initial_state = engine.get_state().copy()
        
        engine.apply_operator(identity)
        final_state = engine.get_state()
        
        np.testing.assert_array_almost_equal(initial_state, final_state)
    
    def test_pauli_x_operator(self):
        """Test Pauli-X (bit flip) operator."""
        engine = FractalStateEngine(num_qubits=1)
        
        # Start with |0⟩
        self.assertAlmostEqual(abs(engine.get_state()[0]), 1.0)
        
        # Apply X gate
        pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
        engine.apply_operator(pauli_x)
        
        # Should be |1⟩
        self.assertAlmostEqual(abs(engine.get_state()[1]), 1.0)


if __name__ == '__main__':
    unittest.main()
