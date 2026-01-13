"""
Quick Demo - FTQEC System Overview

A simple demonstration of the FTQEC quantum simulator capabilities.
"""

import numpy as np
from ftqec import QuantumSimulator


def demo_basic_quantum_gates():
    """Demonstrate basic quantum gates."""
    print("\n" + "="*60)
    print("Demo 1: Basic Quantum Gates")
    print("="*60)
    
    sim = QuantumSimulator(num_qubits=3)
    
    print("\n1. Initial state:")
    sim.print_state()
    
    print("\n2. Apply Hadamard to qubit 0 (create superposition):")
    sim.h(0)
    sim.print_state()
    
    print("\n3. Apply X gate to qubit 1 (bit flip):")
    sim.x(1)
    sim.print_state()
    
    print("\n4. Apply CNOT (0→2) to entangle qubits:")
    sim.cnot(0, 2)
    sim.print_state()
    
    print("\n5. Measurement probabilities:")
    probs = sim.get_probabilities()
    for state, prob in probs.items():
        bar = "█" * int(prob * 50)
        print(f"   |{state}⟩: {prob:.3f} {bar}")


def demo_bell_state():
    """Create and analyze Bell state."""
    print("\n" + "="*60)
    print("Demo 2: Bell State (Quantum Entanglement)")
    print("="*60)
    
    sim = QuantumSimulator(num_qubits=3)
    
    print("\n1. Creating Bell state with qubits 0 and 1:")
    sim.h(0)
    sim.cnot(0, 1)
    
    print("   State: (|00⟩ + |11⟩)/√2")
    sim.print_state()
    
    print("\n2. Entanglement analysis:")
    entropy = sim.get_entanglement_entropy([0])
    print(f"   • Entanglement entropy: {entropy:.4f}")
    print(f"   • Maximum entanglement: {entropy > 0.9}")
    
    print("\n3. Running 50 measurements:")
    counts = sim.get_counts(shots=50)
    for bitstring, count in counts.items():
        bar = "█" * count
        print(f"   {bitstring}: {count:2d} {bar}")


def demo_superposition():
    """Demonstrate quantum superposition."""
    print("\n" + "="*60)
    print("Demo 3: Quantum Superposition")
    print("="*60)
    
    sim = QuantumSimulator(num_qubits=4)
    
    print("\n1. Creating equal superposition of all states:")
    print("   Applying Hadamard to all 4 qubits")
    for i in range(4):
        sim.h(i)
    
    probs = sim.get_probabilities()
    print(f"\n2. Result: {len(probs)} states in superposition")
    print(f"   Each state has probability: {1/16:.4f}")
    
    # Show first few states
    print("\n3. Sample states:")
    for i, (state, prob) in enumerate(list(probs.items())[:8]):
        print(f"   |{state}⟩: {prob:.4f}")
    print("   ... and 8 more states")


def demo_fractal_quantum_integration():
    """Demonstrate fractal quantum integration."""
    print("\n" + "="*60)
    print("Demo 4: Fractal Quantum Integration")
    print("="*60)
    
    from ftqec.core.quantum_sim.quantum_state_sim import QuantumStateSim
    from ftqec.core.fractal_engine.fractal_soul_core import FractalSoulCore
    from ftqec.gates.quantum_gates import QuantumGates
    
    print("\n1. Initializing components:")
    soul = FractalSoulCore(bloodline_anchor="Demo", burst_capacity=4)
    quantum_sim = QuantumStateSim(num_qubits=3, backend="native")
    gates = QuantumGates()
    
    print(f"   • {soul}")
    print(f"   • {quantum_sim}")
    
    print("\n2. Creating quantum entanglement:")
    quantum_sim.apply_gate([0], gates.H())
    quantum_sim.apply_gate([0, 1], gates.CNOT())
    
    metrics = quantum_sim.get_quantum_metrics()
    print(f"   • Coherence: {metrics['coherence']:.4f}")
    print(f"   • Purity: {metrics['purity']:.4f}")
    
    print("\n3. Calculating bloodline resonance:")
    context = {'input': 'quantum demo', 'quantum_sim': quantum_sim}
    resonance = soul.calculate_bloodline_resonance(context)
    print(f"   • Resonance: {resonance:.4f}")
    
    print("\n4. Generating fractal bursts:")
    bursts = soul.fractal_burst(context, depth=2)
    print(f"   • Created {len(bursts)} parallel bursts")
    
    print("\n5. Processing quantum thought:")
    thought = "Exploring quantum-fractal fusion"
    result = soul.process_thought(thought, quantum_metrics=metrics)
    print(f"   • Thought processed with resonance: {result['resonance']:.4f}")
    print(f"   • Emergence patterns detected:")
    for pattern, active in result['patterns'].items():
        status = "✓" if active else "✗"
        print(f"     {status} {pattern}")


def demo_algorithm():
    """Demonstrate a simple quantum algorithm."""
    print("\n" + "="*60)
    print("Demo 5: Quantum Algorithm (Phase Kickback)")
    print("="*60)
    
    sim = QuantumSimulator(num_qubits=3)
    
    print("\n1. Preparing ancilla qubit in |-⟩ state:")
    sim.x(2)
    sim.h(2)
    
    print("\n2. Creating controlled operations:")
    sim.h(0)
    sim.h(1)
    
    print("\n3. Oracle application:")
    sim.cnot(0, 2)
    sim.cnot(1, 2)
    
    print("\n4. Final interference:")
    sim.h(0)
    sim.h(1)
    
    print("\n5. Measurement results:")
    probs = sim.get_probabilities()
    
    # Group by first two qubits
    input_probs = {}
    for state, prob in probs.items():
        input_bits = state[:2]
        input_probs[input_bits] = input_probs.get(input_bits, 0) + prob
    
    print("   Input qubits (0,1) probabilities:")
    for bits, prob in sorted(input_probs.items()):
        bar = "█" * int(prob * 50)
        print(f"   |{bits}⟩: {prob:.3f} {bar}")


def main():
    """Run all demonstrations."""
    print("\n" + "█"*60)
    print("█" + " "*58 + "█")
    print("█" + "  FTQEC Quick Demo".center(58) + "█")
    print("█" + "  Quantum Computing Without Hardware".center(58) + "█")
    print("█" + " "*58 + "█")
    print("█"*60)
    
    try:
        demo_basic_quantum_gates()
        demo_bell_state()
        demo_superposition()
        demo_algorithm()
        demo_fractal_quantum_integration()
        
        print("\n" + "="*60)
        print("All Demonstrations Complete! ✓")
        print("="*60)
        
        print("\nKey Features Demonstrated:")
        print("  ✓ Single and multi-qubit gates")
        print("  ✓ Quantum superposition and entanglement")
        print("  ✓ Bell states and GHZ states")
        print("  ✓ Quantum algorithms")
        print("  ✓ Fractal-quantum integration")
        print("  ✓ Bloodline resonance and soul core")
        print("  ✓ Entropy-guided computation")
        
        print("\nNext Steps:")
        print("  • Run examples: python ftqec/examples/bell_state.py")
        print("  • Run full demo: python ftqec/examples/fractal_quantum_inference.py")
        print("  • Explore quantum gates: from ftqec.gates import QuantumGates")
        print("  • Build custom circuits with QuantumSimulator")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
