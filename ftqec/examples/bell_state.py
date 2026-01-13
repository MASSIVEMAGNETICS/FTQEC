"""
Example: Bell State Creation and Entanglement
Demonstrates quantum entanglement using the FTQEC simulator.
"""

import numpy as np
from ftqec import QuantumSimulator


def create_bell_state():
    """Create a Bell state (maximally entangled 2-qubit state)."""
    print("=" * 60)
    print("Bell State Creation Example")
    print("=" * 60)
    
    # Create a 3-qubit simulator (we'll use 2 qubits)
    sim = QuantumSimulator(num_qubits=3)
    
    print("\n1. Initial state: |000⟩")
    sim.print_state()
    
    # Apply Hadamard to qubit 0
    print("\n2. Apply Hadamard to qubit 0")
    sim.h(0)
    sim.print_state()
    
    # Apply CNOT with control=0, target=1
    print("\n3. Apply CNOT (control=0, target=1)")
    sim.cnot(0, 1)
    sim.print_state()
    
    print("\n4. Bell state created! (|000⟩ + |110⟩)/√2")
    print("   This is a maximally entangled state.")
    
    # Calculate entanglement entropy
    entropy = sim.get_entanglement_entropy([0])
    print(f"\n5. Entanglement entropy of qubit 0: {entropy:.3f}")
    print(f"   (Maximum entropy = 1.0 for a 2-qubit system)")
    
    # Get measurement probabilities
    print("\n6. Measurement probabilities:")
    probs = sim.get_probabilities()
    for bitstring, prob in probs.items():
        print(f"   |{bitstring}⟩: {prob:.3f} ({prob*100:.1f}%)")
    
    # Run measurements
    print("\n7. Running 100 measurements:")
    counts = sim.get_counts(shots=100)
    for bitstring, count in counts.items():
        print(f"   {bitstring}: {count} times")
    
    print("\n" + "=" * 60)
    return sim


def ghz_state():
    """Create a GHZ state (3-qubit entangled state)."""
    print("\n" + "=" * 60)
    print("GHZ State Creation Example")
    print("=" * 60)
    
    # Create a 3-qubit simulator
    sim = QuantumSimulator(num_qubits=3)
    
    print("\n1. Initial state: |000⟩")
    sim.print_state()
    
    # Create GHZ state: (|000⟩ + |111⟩)/√2
    print("\n2. Apply Hadamard to qubit 0")
    sim.h(0)
    
    print("\n3. Apply CNOT (0→1)")
    sim.cnot(0, 1)
    
    print("\n4. Apply CNOT (0→2)")
    sim.cnot(0, 2)
    
    print("\n5. GHZ state created: (|000⟩ + |111⟩)/√2")
    sim.print_state()
    
    # Get measurement probabilities
    print("\n6. Measurement probabilities:")
    probs = sim.get_probabilities()
    for bitstring, prob in probs.items():
        print(f"   |{bitstring}⟩: {prob:.3f}")
    
    # Calculate entanglement
    entropy_0 = sim.get_entanglement_entropy([0])
    entropy_01 = sim.get_entanglement_entropy([0, 1])
    print(f"\n7. Entanglement metrics:")
    print(f"   Entropy of qubit 0: {entropy_0:.3f}")
    print(f"   Entropy of qubits 0,1: {entropy_01:.3f}")
    
    print("\n" + "=" * 60)
    return sim


def quantum_teleportation():
    """Demonstrate quantum teleportation protocol (simplified)."""
    print("\n" + "=" * 60)
    print("Quantum Teleportation Example")
    print("=" * 60)
    
    sim = QuantumSimulator(num_qubits=3)
    
    print("\n1. Prepare qubit 0 in state to teleport: (|0⟩ + |1⟩)/√2")
    sim.h(0)
    sim.print_state()
    
    print("\n2. Create Bell pair between qubits 1 and 2")
    sim.h(1)
    sim.cnot(1, 2)
    sim.print_state()
    
    print("\n3. Alice's operations: CNOT and H on qubits 0,1")
    sim.cnot(0, 1)
    sim.h(0)
    sim.print_state()
    
    print("\n4. In real teleportation, Alice measures qubits 0,1")
    print("   and Bob applies corrections to qubit 2 based on results")
    print("   (Full implementation would require classical communication)")
    
    print("\n" + "=" * 60)
    return sim


if __name__ == "__main__":
    # Run examples
    bell_sim = create_bell_state()
    ghz_sim = ghz_state()
    teleport_sim = quantum_teleportation()
    
    print("\n✓ All examples completed successfully!")
    print("\nThese examples demonstrate:")
    print("  • Quantum superposition (Hadamard gate)")
    print("  • Quantum entanglement (CNOT gate)")
    print("  • Multi-qubit entangled states (GHZ state)")
    print("  • Complex quantum protocols (Teleportation)")
