"""
Entropy Utilities - Entropy calculation and analysis tools.

Provides various entropy measures for quantum and classical systems.
"""

import numpy as np
from typing import Dict, List


def shannon_entropy(probabilities: Dict[str, float]) -> float:
    """
    Calculate Shannon entropy from probability distribution.
    
    Args:
        probabilities: Dictionary mapping states to probabilities
        
    Returns:
        Shannon entropy in bits
    """
    entropy = 0.0
    for prob in probabilities.values():
        if prob > 1e-10:
            entropy -= prob * np.log2(prob)
    return float(entropy)


def von_neumann_entropy(density_matrix: np.ndarray) -> float:
    """
    Calculate von Neumann entropy of quantum state.
    
    Args:
        density_matrix: Density matrix of quantum system
        
    Returns:
        Von Neumann entropy
    """
    eigenvalues = np.linalg.eigvalsh(density_matrix)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    
    if len(eigenvalues) == 0:
        return 0.0
    
    entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
    return float(entropy)


def relative_entropy(p: Dict[str, float], q: Dict[str, float]) -> float:
    """
    Calculate relative entropy (KL divergence) D(P||Q).
    
    Args:
        p: First probability distribution
        q: Second probability distribution
        
    Returns:
        Relative entropy
    """
    rel_entropy = 0.0
    
    for state in p:
        if state in q and p[state] > 1e-10 and q[state] > 1e-10:
            rel_entropy += p[state] * np.log2(p[state] / q[state])
    
    return float(rel_entropy)


def phase_entropy(state_vector: np.ndarray, bins: int = 16) -> float:
    """
    Calculate entropy of phase distribution.
    
    Args:
        state_vector: Complex quantum state vector
        bins: Number of phase bins
        
    Returns:
        Phase entropy
    """
    # Extract phases of non-zero amplitudes
    phases = np.angle(state_vector[np.abs(state_vector) > 1e-10])
    
    if len(phases) == 0:
        return 0.0
    
    # Create histogram
    hist, _ = np.histogram(phases, bins=bins, range=(-np.pi, np.pi))
    hist = hist / np.sum(hist)
    hist = hist[hist > 0]
    
    # Calculate entropy
    entropy = -np.sum(hist * np.log2(hist))
    return float(entropy)


def mutual_information(joint_probs: Dict[tuple, float],
                       marginal_p: Dict[str, float],
                       marginal_q: Dict[str, float]) -> float:
    """
    Calculate mutual information I(X:Y).
    
    Args:
        joint_probs: Joint probability distribution
        marginal_p: Marginal distribution for X
        marginal_q: Marginal distribution for Y
        
    Returns:
        Mutual information
    """
    mi = 0.0
    
    for (x, y), p_xy in joint_probs.items():
        if p_xy > 1e-10 and marginal_p.get(x, 0) > 1e-10 and marginal_q.get(y, 0) > 1e-10:
            mi += p_xy * np.log2(p_xy / (marginal_p[x] * marginal_q[y]))
    
    return float(mi)


def entanglement_entropy(state_vector: np.ndarray, subsystem_qubits: List[int],
                         total_qubits: int) -> float:
    """
    Calculate entanglement entropy of a subsystem.
    
    Args:
        state_vector: Full system state vector
        subsystem_qubits: List of qubit indices in subsystem
        total_qubits: Total number of qubits
        
    Returns:
        Entanglement entropy
    """
    # Reshape state into tensor
    state_tensor = state_vector.reshape([2] * total_qubits)
    
    # Compute reduced density matrix (simplified)
    subsystem_dim = 2 ** len(subsystem_qubits)
    complement_dim = 2 ** (total_qubits - len(subsystem_qubits))
    
    # Full density matrix
    rho_full = np.outer(state_vector, state_vector.conj())
    
    # Trace out complement (simplified approach)
    rho_reduced = np.zeros((subsystem_dim, subsystem_dim), dtype=complex)
    
    for i in range(subsystem_dim):
        for j in range(subsystem_dim):
            for k in range(complement_dim):
                idx_i = _merge_indices(i, k, subsystem_qubits, total_qubits)
                idx_j = _merge_indices(j, k, subsystem_qubits, total_qubits)
                rho_reduced[i, j] += rho_full[idx_i, idx_j]
    
    # Calculate von Neumann entropy
    return von_neumann_entropy(rho_reduced)


def _merge_indices(sub_idx: int, comp_idx: int, subsystem: List[int],
                   total_qubits: int) -> int:
    """Helper to merge subsystem and complement indices."""
    result = 0
    sub_bits = [(sub_idx >> i) & 1 for i in range(len(subsystem))]
    comp_bits = [(comp_idx >> i) & 1 for i in range(total_qubits - len(subsystem))]
    
    sub_pos = 0
    comp_pos = 0
    for qubit in range(total_qubits):
        if qubit in subsystem:
            bit = sub_bits[sub_pos]
            sub_pos += 1
        else:
            bit = comp_bits[comp_pos]
            comp_pos += 1
        result |= (bit << (total_qubits - 1 - qubit))
    
    return result


def entropy_budget(initial_entropy: float, depth: int, decay_rate: float = 0.9) -> float:
    """
    Calculate remaining entropy budget at given depth.
    
    Args:
        initial_entropy: Starting entropy
        depth: Current recursion depth
        decay_rate: Entropy decay rate per level
        
    Returns:
        Remaining entropy
    """
    return initial_entropy * (decay_rate ** depth)


def entropy_threshold_reached(current_entropy: float, threshold: float) -> bool:
    """
    Check if entropy has fallen below threshold.
    
    Args:
        current_entropy: Current entropy value
        threshold: Threshold value
        
    Returns:
        True if threshold reached
    """
    return current_entropy < threshold
