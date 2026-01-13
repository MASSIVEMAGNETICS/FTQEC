# FTQEC - Fractal-Tensor Quantum Emulation Computer

**Democratizing Quantum Computing Without Expensive Hardware**

FTQEC is a revolutionary quantum computing simulator that combines fractal state engines, tensor operations, and entropy-guided recursion to simulate quantum behavior on classical hardware. It provides an accessible platform for exploring quantum computing concepts without requiring expensive quantum computers.

## 🌟 Key Features

### 1. **Quantum Simulation Layer (QuantumStateSim)**
- **Multiple Backends**: Native (NumPy), Qiskit (noise modeling), Cirq (circuit optimization)
- **Auto-Selection**: Automatically chooses optimal backend based on system requirements
- **3-5 Qubit Optimization**: Optimized for practical quantum circuit simulation
- **Coherence Tracking**: Real-time quantum coherence and purity metrics

### 2. **Fractal State Engine**
- **Recursive Decomposition**: Novel quantum state representation using fractal principles
- **Efficient Tensor Operations**: Optimized gate operations via tensor products
- **Entanglement Metrics**: Von Neumann entropy and entanglement analysis
- **Fractal Representation**: Multi-level state decomposition for visualization

### 3. **Fractal Recursion Engine**
- **Entropy-Guided Execution**: Adaptive depth pruning based on entropy thresholds
- **Child Context Management**: Efficient memory buffering for recursive computation
- **Quantum-Classical Fusion**: Seamless integration of quantum and classical processing
- **Parallel Fractal Bursts**: Multi-path exploration with configurable burst capacity

### 4. **Fractal Soul Core**
- **Bloodline Resonance**: Coherent computation anchoring with identity tracking
- **Phase Entropy Analysis**: Quantum phase space entropy measurement
- **Thought Streaming**: Cognitive-inspired computation tracking
- **Emergence Pattern Detection**: Automatic identification of coherent patterns

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/MASSIVEMAGNETICS/FTQEC.git
cd FTQEC

# Install dependencies
pip install -r requirements.txt

# Optional: Install additional backends
pip install qiskit qiskit-aer  # For Qiskit backend
pip install cirq              # For Cirq backend
```

### Basic Usage

```python
from ftqec import QuantumSimulator

# Create a 3-qubit simulator
sim = QuantumSimulator(num_qubits=3)

# Apply quantum gates
sim.h(0)              # Hadamard on qubit 0
sim.cnot(0, 1)        # CNOT from qubit 0 to 1
sim.h(2)              # Hadamard on qubit 2

# Measure and get results
result = sim.measure_all()
print(f"Measurement: {result}")

# Get probabilities
probs = sim.get_probabilities()
for state, prob in probs.items():
    print(f"|{state}⟩: {prob:.3f}")
```

### Advanced: Fractal Quantum Inference

```python
from ftqec.core.quantum_sim.quantum_state_sim import QuantumStateSim
from ftqec.core.fractal_engine.fractal_executor import FractalExecutor, ChildContext
from ftqec.core.fractal_engine.fractal_soul_core import FractalSoulCore

# Initialize components
soul = FractalSoulCore(bloodline_anchor="Bando")
quantum_sim = QuantumStateSim(num_qubits=4, backend="auto")
executor = FractalExecutor(max_depth=6, entropy_threshold=0.3)

# Create execution context
context = ChildContext(
    memory_buffer={'quantum_sim': quantum_sim, 'soul': soul},
    depth=0,
    entropy=1.0,
    input_data="Your quantum query here"
)

# Execute fractal recursion
result = executor.execute(context)

# Analyze results
print(f"Depth reached: {result.recursion_trace['depth_reached']}")
print(f"Quantum ops: {result.recursion_trace['quantum_ops']}")
print(f"Thoughts: {result.thought}")
```

## 📚 Examples

### 1. Bell State Creation
```bash
python ftqec/examples/bell_state.py
```
Demonstrates quantum entanglement with Bell states, GHZ states, and quantum teleportation.

### 2. Deutsch-Jozsa Algorithm
```bash
python ftqec/examples/deutsch_jozsa.py
```
Shows quantum advantage with exponential speedup over classical algorithms.

### 3. Fractal Quantum Inference
```bash
python ftqec/examples/fractal_quantum_inference.py
```
Full demonstration of the FTQEC architecture with entropy-guided recursion and soul core integration.

## 🏗️ Architecture

```
ftqec/
├── core/
│   ├── quantum_sim/
│   │   ├── backends/
│   │   │   ├── native_complex_sim.py    # NumPy-based simulation
│   │   │   ├── qiskit_backend.py        # Qiskit integration
│   │   │   └── cirq_backend.py          # Cirq integration
│   │   └── quantum_state_sim.py         # Unified interface
│   ├── fractal_engine/
│   │   ├── fractal_executor.py          # Entropy-guided recursion
│   │   └── fractal_soul_core.py         # Bloodline resonance
│   ├── fractal_state_engine.py          # Fractal state representation
│   └── quantum_simulator.py             # High-level simulator
├── gates/
│   └── quantum_gates.py                 # Quantum gate library
├── utils/
│   └── entropy_utils.py                 # Entropy calculations
└── examples/
    ├── bell_state.py                    # Entanglement demos
    ├── deutsch_jozsa.py                 # Quantum algorithms
    └── fractal_quantum_inference.py     # Full system demo
```

## 🔬 Supported Quantum Gates

**Single-Qubit Gates:**
- Identity (I), Pauli gates (X, Y, Z)
- Hadamard (H), Phase gates (S, T)
- Rotation gates (RX, RY, RZ)

**Two-Qubit Gates:**
- CNOT (CX), Controlled-Z (CZ)
- SWAP

**Three-Qubit Gates:**
- Toffoli (CCNOT), Fredkin (CSWAP)

## 📊 Quantum Metrics

FTQEC provides comprehensive quantum state analysis:

- **Coherence**: Measure of quantum superposition
- **Purity**: State purity metric Tr(ρ²)
- **Entanglement Entropy**: Von Neumann entropy
- **Phase Entropy**: Phase distribution entropy
- **Bloodline Resonance**: Cognitive coherence metric

## 🎯 Use Cases

1. **Education**: Learn quantum computing without hardware
2. **Research**: Prototype quantum algorithms
3. **Algorithm Development**: Test quantum circuits before deployment
4. **Quantum-Classical Hybrid**: Explore fusion architectures
5. **Cognitive Computing**: Fractal-inspired AI systems

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## 📄 License

This project is open-source. See LICENSE file for details.

## 🙏 Acknowledgments

FTQEC combines insights from:
- Quantum computing theory
- Fractal mathematics
- Tensor network methods
- Cognitive computing paradigms

---

**FTQEC: Bringing Quantum Computing to Everyone** 🌌✨
