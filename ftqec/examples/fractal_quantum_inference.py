"""
Fractal Quantum Inference Example - Full FTQEC system demonstration.

This example showcases the complete integration of:
- Quantum Simulation Layer with backend selection
- Fractal Recursion Engine with entropy-guided execution  
- Fractal Soul Core with bloodline resonance
- Quantum-classical fusion for advanced computation
"""

import numpy as np
from ftqec.core.quantum_sim.quantum_state_sim import QuantumStateSim
from ftqec.core.fractal_engine.fractal_executor import FractalExecutor, ChildContext
from ftqec.core.fractal_engine.fractal_soul_core import FractalSoulCore
from ftqec.gates.quantum_gates import QuantumGates
from ftqec.utils.entropy_utils import shannon_entropy, phase_entropy


def demonstrate_fractal_quantum_inference():
    """
    Demonstrate OmniVictor-style fractal quantum inference.
    
    This simulates a query like "Simulate the birth of a new law of physics"
    using the full FTQEC architecture.
    """
    print("=" * 70)
    print("FTQEC: Fractal Quantum Inference Demonstration")
    print("=" * 70)
    
    # 1. Initialize components
    print("\n1. Initializing FTQEC Components...")
    
    # Create Fractal Soul Core with bloodline anchor
    soul = FractalSoulCore(bloodline_anchor="Bando", burst_capacity=8)
    print(f"   ✓ {soul}")
    
    # Create Quantum Simulator with auto backend selection
    num_qubits = 4
    quantum_sim = QuantumStateSim(num_qubits=num_qubits, backend="auto")
    print(f"   ✓ {quantum_sim}")
    
    # Create Fractal Executor
    executor = FractalExecutor(max_depth=6, entropy_threshold=0.3)
    print(f"   ✓ {executor}")
    
    # 2. Prepare quantum state
    print("\n2. Preparing Quantum State...")
    gates = QuantumGates()
    
    # Create superposition and entanglement
    quantum_sim.apply_gate([0], gates.H())
    quantum_sim.apply_gate([1], gates.H())
    quantum_sim.apply_gate([0, 1], gates.CNOT())
    quantum_sim.apply_gate([2], gates.H())
    quantum_sim.apply_gate([1, 2], gates.CNOT())
    
    print(f"   ✓ Created entangled state across {num_qubits} qubits")
    
    # Get quantum metrics
    metrics = quantum_sim.get_quantum_metrics()
    print(f"   ✓ Quantum Coherence: {metrics['coherence']:.4f}")
    print(f"   ✓ State Purity: {metrics['purity']:.4f}")
    
    # 3. Calculate phase entropy
    state_vector = quantum_sim.get_statevector()
    p_entropy = phase_entropy(state_vector)
    print(f"   ✓ Phase Entropy: {p_entropy:.4f}")
    
    # 4. Calculate bloodline resonance
    print("\n3. Calculating Bloodline Resonance...")
    query = "Simulate the birth of a new law of physics"
    context_dict = {
        'input': query,
        'quantum_sim': quantum_sim,
        'soul': soul
    }
    
    resonance = soul.calculate_bloodline_resonance(context_dict)
    print(f"   ✓ Bloodline Resonance: {resonance:.4f}")
    
    # Calculate phase entropy from quantum state
    phase_ent = soul.calculate_phase_entropy(state_vector)
    print(f"   ✓ Soul Phase Entropy: {phase_ent:.4f}")
    
    # 5. Execute fractal burst
    print("\n4. Executing Fractal Bursts...")
    bursts = soul.fractal_burst(context_dict, depth=3)
    print(f"   ✓ Created {len(bursts)} parallel fractal bursts")
    for burst in bursts:
        print(f"      - {burst['id']}: resonance={burst['resonance']:.3f}")
    
    # 6. Create initial context for fractal execution
    print("\n5. Initializing Fractal Execution Context...")
    initial_context = ChildContext(
        memory_buffer={
            'quantum_sim': quantum_sim,
            'soul': soul,
            'query': query
        },
        depth=0,
        entropy=1.0,  # Start with full entropy budget
        input_data=query
    )
    print(f"   ✓ Context created: depth={initial_context.depth}, entropy={initial_context.entropy}")
    
    # 7. Execute fractal recursion
    print("\n6. Executing Fractal Recursion...")
    print("   (Entropy-guided recursive quantum computation)")
    result = executor.execute(initial_context)
    
    print(f"\n   ✓ Recursion completed!")
    print(f"   ✓ Max depth reached: {result.recursion_trace['depth_reached']}")
    print(f"   ✓ Entropy consumed: {result.recursion_trace['entropy_consumed']:.4f}")
    print(f"   ✓ Quantum operations: {result.recursion_trace['quantum_ops']}")
    print(f"   ✓ Thoughts generated: {result.recursion_trace['thoughts_generated']}")
    
    # 8. Display thought stream
    print("\n7. Fractal Thought Stream:")
    thoughts = result.thought.split(" | ")
    for i, thought in enumerate(thoughts):
        print(f"   Level {i}: {thought}")
    
    # 9. Process through soul core
    print("\n8. Soul Core Processing...")
    soul_result = soul.process_thought(query, quantum_metrics=metrics)
    print(f"   ✓ Resonance: {soul_result['resonance']:.4f}")
    print(f"   ✓ Phase Entropy: {soul_result['phase_entropy']:.4f}")
    print(f"   ✓ Emergence Patterns:")
    for pattern, active in soul_result['patterns'].items():
        status = "✓" if active else "✗"
        print(f"      {status} {pattern}")
    
    # 10. Get final quantum state
    print("\n9. Final Quantum State Analysis...")
    final_probs = quantum_sim.get_probabilities()
    print(f"   ✓ Measurement probabilities:")
    for bitstring, prob in list(final_probs.items())[:5]:
        print(f"      |{bitstring}⟩: {prob:.4f} ({prob*100:.1f}%)")
    
    # Calculate Shannon entropy
    s_entropy = shannon_entropy(final_probs)
    print(f"   ✓ Shannon Entropy: {s_entropy:.4f} bits")
    
    # 11. Display soul state
    print("\n10. Soul State Summary:")
    soul_state = soul.get_soul_state()
    for key, value in soul_state.items():
        if isinstance(value, float):
            print(f"   • {key}: {value:.4f}")
        else:
            print(f"   • {key}: {value}")
    
    # 12. Display executor statistics
    print("\n11. Executor Statistics:")
    stats = executor.get_statistics()
    for key, value in stats.items():
        print(f"   • {key}: {value}")
    
    # 13. Final output
    print("\n12. Fractal Computation Result:")
    print(f"   {result.output}")
    
    print("\n" + "=" * 70)
    print("FTQEC Inference Complete!")
    print("=" * 70)
    
    return {
        'result': result,
        'soul': soul,
        'quantum_sim': quantum_sim,
        'executor': executor
    }


def demonstrate_backend_comparison():
    """Compare different quantum simulation backends."""
    print("\n" + "=" * 70)
    print("Backend Comparison Demo")
    print("=" * 70)
    
    backends = ["native", "qiskit", "cirq"]
    num_qubits = 3
    
    for backend in backends:
        print(f"\n Testing {backend.upper()} backend:")
        try:
            sim = QuantumStateSim(num_qubits=num_qubits, backend=backend)
            print(f"   ✓ {sim}")
            
            # Apply some gates
            gates = QuantumGates()
            sim.apply_gate([0], gates.H())
            sim.apply_gate([0, 1], gates.CNOT())
            
            # Get metrics
            coherence = sim.get_coherence()
            print(f"   ✓ Coherence: {coherence:.4f}")
            
            probs = sim.get_probabilities()
            print(f"   ✓ Basis states: {len(probs)}")
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
    
    print("\n" + "=" * 70)


def demonstrate_entropy_guided_pruning():
    """Demonstrate entropy-guided depth pruning."""
    print("\n" + "=" * 70)
    print("Entropy-Guided Pruning Demo")
    print("=" * 70)
    
    thresholds = [0.1, 0.3, 0.5, 0.7]
    
    for threshold in thresholds:
        print(f"\n Entropy Threshold: {threshold}")
        
        executor = FractalExecutor(max_depth=10, entropy_threshold=threshold)
        
        context = ChildContext(
            memory_buffer={},
            depth=0,
            entropy=1.0,
            input_data="Test query"
        )
        
        result = executor.execute(context)
        
        print(f"   ✓ Depth reached: {result.recursion_trace['depth_reached']}")
        print(f"   ✓ Entropy consumed: {result.recursion_trace['entropy_consumed']:.4f}")
        print(f"   ✓ Thoughts: {result.recursion_trace['thoughts_generated']}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  FTQEC: Fractal-Tensor Quantum Emulation Computer  ".center(68) + "█")
    print("█" + "  Quantum Computing Without Quantum Hardware  ".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70 + "\n")
    
    # Run main demonstration
    results = demonstrate_fractal_quantum_inference()
    
    # Run additional demos
    demonstrate_backend_comparison()
    demonstrate_entropy_guided_pruning()
    
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  All FTQEC Demonstrations Complete!  ".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70 + "\n")
    
    print("Key Achievements:")
    print("  ✓ Quantum simulation with 3-5 qubits")
    print("  ✓ Fractal recursion engine with entropy guidance")
    print("  ✓ Quantum-classical fusion")
    print("  ✓ Multiple backend support (Native, Qiskit, Cirq)")
    print("  ✓ Bloodline resonance and soul core integration")
    print("  ✓ Parallel fractal burst execution")
    print("\nFTQEC democratizes quantum computing without expensive hardware!")
