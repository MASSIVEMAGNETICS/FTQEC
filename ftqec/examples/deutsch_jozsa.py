"""
Example: Deutsch-Jozsa Algorithm
Demonstrates a quantum algorithm that provides exponential speedup over classical algorithms.
"""

import numpy as np
from ftqec import QuantumSimulator


def deutsch_jozsa_constant():
    """
    Deutsch-Jozsa algorithm for a constant function.
    This determines if a function is constant or balanced in one query.
    """
    print("=" * 60)
    print("Deutsch-Jozsa Algorithm - Constant Function")
    print("=" * 60)
    
    # 3 qubits: 2 input qubits + 1 output qubit
    sim = QuantumSimulator(num_qubits=3)
    
    print("\n1. Initial state: |000⟩")
    
    # Initialize output qubit in |1⟩
    print("\n2. Initialize qubit 2 to |1⟩")
    sim.x(2)
    
    # Apply Hadamard to all qubits
    print("\n3. Apply Hadamard to all qubits (create superposition)")
    sim.h(0)
    sim.h(1)
    sim.h(2)
    sim.print_state()
    
    # Oracle for constant function (does nothing)
    print("\n4. Apply oracle (constant function - no operation)")
    print("   Constant function: f(x) = 0 for all x")
    
    # Apply Hadamard to input qubits
    print("\n5. Apply Hadamard to input qubits")
    sim.h(0)
    sim.h(1)
    sim.print_state()
    
    # Measure input qubits
    print("\n6. Measure input qubits")
    probs = sim.get_probabilities()
    input_probs = {}
    for bitstring, prob in probs.items():
        input_bits = bitstring[:2]  # First 2 qubits
        if input_bits in input_probs:
            input_probs[input_bits] += prob
        else:
            input_probs[input_bits] = prob
    
    print("   Input qubit probabilities:")
    for bits, prob in input_probs.items():
        print(f"   |{bits}⟩: {prob:.3f}")
    
    print("\n7. Result: |00⟩ measured → Function is CONSTANT")
    print("   Classical algorithm would need 2^(n-1)+1 queries")
    print("   Quantum algorithm needs only 1 query!")
    
    print("\n" + "=" * 60)
    return sim


def deutsch_jozsa_balanced():
    """
    Deutsch-Jozsa algorithm for a balanced function.
    """
    print("\n" + "=" * 60)
    print("Deutsch-Jozsa Algorithm - Balanced Function")
    print("=" * 60)
    
    sim = QuantumSimulator(num_qubits=3)
    
    print("\n1. Initialize qubit 2 to |1⟩")
    sim.x(2)
    
    print("\n2. Apply Hadamard to all qubits")
    sim.h(0)
    sim.h(1)
    sim.h(2)
    
    # Oracle for balanced function: flip output if first input is 1
    print("\n3. Apply oracle (balanced function)")
    print("   Balanced function: f(x) = x[0] (first bit)")
    sim.cnot(0, 2)  # If qubit 0 is |1⟩, flip qubit 2
    sim.print_state()
    
    print("\n4. Apply Hadamard to input qubits")
    sim.h(0)
    sim.h(1)
    sim.print_state()
    
    print("\n5. Measure input qubits")
    probs = sim.get_probabilities()
    input_probs = {}
    for bitstring, prob in probs.items():
        input_bits = bitstring[:2]
        if input_bits in input_probs:
            input_probs[input_bits] += prob
        else:
            input_probs[input_bits] = prob
    
    print("   Input qubit probabilities:")
    for bits, prob in input_probs.items():
        print(f"   |{bits}⟩: {prob:.3f}")
    
    print("\n6. Result: Non-zero state measured → Function is BALANCED")
    
    print("\n" + "=" * 60)
    return sim


def quantum_superposition_demo():
    """Demonstrate quantum superposition with 4 qubits."""
    print("\n" + "=" * 60)
    print("Quantum Superposition - 4 Qubit Demo")
    print("=" * 60)
    
    sim = QuantumSimulator(num_qubits=4)
    
    print("\n1. Initial state: |0000⟩")
    sim.print_state()
    
    print("\n2. Apply Hadamard to all qubits")
    print("   This creates equal superposition of all 2^4 = 16 basis states")
    sim.h(0)
    sim.h(1)
    sim.h(2)
    sim.h(3)
    sim.print_state()
    
    print("\n3. Each basis state has probability 1/16 = 0.0625")
    probs = sim.get_probabilities()
    print(f"   Number of non-zero states: {len(probs)}")
    print(f"   Each state probability: {list(probs.values())[0]:.4f}")
    
    print("\n4. This demonstrates quantum parallelism:")
    print("   The quantum computer is effectively in all 16 states")
    print("   simultaneously, enabling parallel computation!")
    
    # Apply some operations
    print("\n5. Apply CNOT operations to create entanglement")
    sim.cnot(0, 1)
    sim.cnot(2, 3)
    sim.print_state()
    
    print("\n6. Run measurements:")
    counts = sim.get_counts(shots=160)
    print(f"   Total unique outcomes: {len(counts)}")
    top_5 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
    print("   Top 5 outcomes:")
    for bitstring, count in top_5:
        print(f"   {bitstring}: {count} times")
    
    print("\n" + "=" * 60)
    return sim


if __name__ == "__main__":
    # Run examples
    print("\nRunning Deutsch-Jozsa Algorithm Examples...")
    print("This algorithm demonstrates quantum advantage!\n")
    
    constant_sim = deutsch_jozsa_constant()
    balanced_sim = deutsch_jozsa_balanced()
    superposition_sim = quantum_superposition_demo()
    
    print("\n✓ All algorithm examples completed successfully!")
    print("\nKey insights:")
    print("  • Quantum algorithms can solve certain problems exponentially faster")
    print("  • Superposition enables parallel computation")
    print("  • Interference amplifies correct answers")
    print("  • Deutsch-Jozsa shows quantum advantage with 1 query vs 2^(n-1)+1")
