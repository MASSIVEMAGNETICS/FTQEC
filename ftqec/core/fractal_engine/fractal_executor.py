"""
Fractal Executor - Entropy-guided recursive computation engine.

Implements adaptive depth pruning and parallel fractal burst execution
with quantum-classical fusion.
"""

import numpy as np
from typing import Dict, Any, Optional, List
import copy


class ChildContext:
    """
    Child execution context for fractal recursion.
    
    Attributes:
        memory_buffer: Local memory state
        depth: Current recursion depth
        entropy: Entropy budget remaining
        output: Accumulated output
        parent_id: Parent context identifier
    """
    
    def __init__(self, memory_buffer: Dict, depth: int, entropy: float,
                 input_data: Any = None, parent_id: str = "root"):
        """
        Initialize child context.
        
        Args:
            memory_buffer: Memory state dictionary
            depth: Recursion depth
            entropy: Entropy budget
            input_data: Input data for processing
            parent_id: Parent context ID
        """
        self.memory_buffer = copy.deepcopy(memory_buffer)
        self.depth = depth
        self.entropy = entropy
        self.input = input_data
        self.output = None
        self.parent_id = parent_id
        self.id = f"{parent_id}_d{depth}"
        
        # Quantum state tracking
        self.quantum_state = None
        self.quantum_metrics = {}
        
        # Thought tracking
        self.thought = ""
        self.patterns = []
        
        # Recursion trace
        self.recursion_trace = {
            'depth_reached': depth,
            'entropy_consumed': 0.0,
            'quantum_ops': 0,
            'thoughts_generated': 0
        }
    
    def consume_entropy(self, amount: float):
        """Consume entropy budget."""
        self.entropy = max(0.0, self.entropy - amount)
        self.recursion_trace['entropy_consumed'] += amount
    
    def record_quantum_op(self):
        """Record quantum operation."""
        self.recursion_trace['quantum_ops'] += 1
    
    def add_thought(self, thought: str):
        """Add thought to context."""
        if self.thought:
            self.thought += " | " + thought
        else:
            self.thought = thought
        self.recursion_trace['thoughts_generated'] += 1


class FractalExecutor:
    """
    Fractal recursion engine with entropy-guided execution.
    
    Features:
    - Adaptive depth pruning based on entropy threshold
    - Quantum-classical fusion through quantum_sim integration
    - Parallel fractal burst execution via soul core
    - Memory-efficient child context spawning
    
    Attributes:
        max_depth: Maximum recursion depth
        entropy_threshold: Minimum entropy to continue recursion
        entropy_decay: Entropy decay rate per level
    """
    
    def __init__(self, max_depth: int = 6, entropy_threshold: float = 0.3,
                 entropy_decay: float = 0.9):
        """
        Initialize Fractal Executor.
        
        Args:
            max_depth: Maximum recursion depth
            entropy_threshold: Minimum entropy for continuation
            entropy_decay: Entropy decay multiplier per level
        """
        self.max_depth = max_depth
        self.entropy_threshold = entropy_threshold
        self.entropy_decay = entropy_decay
        
        # Execution statistics
        self.total_contexts_spawned = 0
        self.max_depth_reached = 0
        self.total_quantum_ops = 0
        
        # Result cache
        self._result_cache = {}
    
    def execute(self, context: ChildContext) -> ChildContext:
        """
        Execute fractal recursion on context.
        
        Args:
            context: Initial execution context
            
        Returns:
            Final context with accumulated results
        """
        # Base case: check termination conditions
        if self._should_terminate(context):
            return self._finalize_context(context)
        
        # Process current level
        context = self._process_level(context)
        
        # Spawn child context for next level
        child_context = self._spawn_child_context(context)
        
        # Recursive call
        result_context = self.execute(child_context)
        
        # Merge results back to parent
        context = self._merge_results(context, result_context)
        
        return context
    
    def execute_parallel(self, contexts: List[ChildContext]) -> List[ChildContext]:
        """
        Execute multiple contexts in parallel (simulated).
        
        Args:
            contexts: List of contexts to execute
            
        Returns:
            List of result contexts
        """
        results = []
        for context in contexts:
            result = self.execute(context)
            results.append(result)
        return results
    
    def _should_terminate(self, context: ChildContext) -> bool:
        """Check if recursion should terminate."""
        # Depth limit
        if context.depth >= self.max_depth:
            return True
        
        # Entropy threshold
        if context.entropy < self.entropy_threshold:
            return True
        
        # Custom termination conditions
        if context.output is not None and context.output != "":
            # If we have a strong output, we can terminate early
            if context.entropy < self.entropy_threshold * 2:
                return True
        
        return False
    
    def _process_level(self, context: ChildContext) -> ChildContext:
        """Process computation at current recursion level."""
        # Consume entropy for processing
        context.consume_entropy(0.1)
        
        # Quantum processing if quantum_sim available
        if 'quantum_sim' in context.memory_buffer:
            context = self._quantum_process(context)
        
        # Classical processing
        context = self._classical_process(context)
        
        # Update statistics
        self.max_depth_reached = max(self.max_depth_reached, context.depth)
        
        return context
    
    def _quantum_process(self, context: ChildContext) -> ChildContext:
        """Perform quantum processing operations."""
        quantum_sim = context.memory_buffer.get('quantum_sim')
        
        if quantum_sim is None:
            return context
        
        try:
            # Get quantum metrics
            metrics = quantum_sim.get_quantum_metrics()
            context.quantum_metrics = metrics
            
            # Consume entropy based on quantum coherence
            coherence = metrics.get('coherence', 0.5)
            entropy_cost = (1.0 - coherence) * 0.05
            context.consume_entropy(entropy_cost)
            
            # Record quantum operation
            context.record_quantum_op()
            self.total_quantum_ops += 1
            
        except Exception as e:
            # Graceful degradation if quantum operations fail
            pass
        
        return context
    
    def _classical_process(self, context: ChildContext) -> ChildContext:
        """Perform classical processing operations."""
        # Generate thought for this level
        if context.input is not None:
            thought = self._generate_thought(context)
            context.add_thought(thought)
        
        # Process memory buffer
        if 'soul' in context.memory_buffer:
            soul = context.memory_buffer['soul']
            soul_state = soul.get_soul_state()
            context.memory_buffer['soul_resonance'] = soul_state['resonance']
        
        return context
    
    def _generate_thought(self, context: ChildContext) -> str:
        """Generate thought content for current level."""
        depth_thoughts = [
            "Initializing quantum fractal space",
            "Expanding recursive thought patterns",
            "Deepening quantum entanglement",
            "Exploring emergent structures",
            "Synthesizing fractal insights",
            "Converging on quantum truth",
            "Manifesting coherent output"
        ]
        
        idx = min(context.depth, len(depth_thoughts) - 1)
        base_thought = depth_thoughts[idx]
        
        # Add context-specific information
        if isinstance(context.input, str):
            # Incorporate input keywords
            words = str(context.input).split()
            if words:
                key_word = words[min(context.depth, len(words) - 1)]
                base_thought += f" [{key_word}]"
        
        return base_thought
    
    def _spawn_child_context(self, parent_context: ChildContext) -> ChildContext:
        """Spawn child context for next recursion level."""
        child = ChildContext(
            memory_buffer=parent_context.memory_buffer,
            depth=parent_context.depth + 1,
            entropy=parent_context.entropy * self.entropy_decay,
            input_data=parent_context.input,
            parent_id=parent_context.id
        )
        
        # Inherit quantum state
        child.quantum_state = parent_context.quantum_state
        child.quantum_metrics = parent_context.quantum_metrics.copy()
        
        self.total_contexts_spawned += 1
        
        return child
    
    def _merge_results(self, parent: ChildContext, child: ChildContext) -> ChildContext:
        """Merge child results back into parent."""
        # Merge thoughts
        if child.thought:
            parent.add_thought(child.thought)
        
        # Merge output
        if child.output is not None:
            if parent.output is None:
                parent.output = child.output
            else:
                parent.output = str(parent.output) + " > " + str(child.output)
        
        # Update recursion trace
        parent.recursion_trace['depth_reached'] = max(
            parent.recursion_trace['depth_reached'],
            child.recursion_trace['depth_reached']
        )
        parent.recursion_trace['entropy_consumed'] += child.recursion_trace['entropy_consumed']
        parent.recursion_trace['quantum_ops'] += child.recursion_trace['quantum_ops']
        parent.recursion_trace['thoughts_generated'] += child.recursion_trace['thoughts_generated']
        
        return parent
    
    def _finalize_context(self, context: ChildContext) -> ChildContext:
        """Finalize context at termination."""
        if context.output is None:
            # Generate default output
            context.output = f"Fractal computation complete at depth {context.depth}"
        
        return context
    
    def get_statistics(self) -> Dict:
        """Get execution statistics."""
        return {
            'total_contexts_spawned': self.total_contexts_spawned,
            'max_depth_reached': self.max_depth_reached,
            'total_quantum_ops': self.total_quantum_ops,
            'cache_size': len(self._result_cache)
        }
    
    def reset_statistics(self):
        """Reset execution statistics."""
        self.total_contexts_spawned = 0
        self.max_depth_reached = 0
        self.total_quantum_ops = 0
        self._result_cache.clear()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"FractalExecutor(max_depth={self.max_depth}, threshold={self.entropy_threshold})"
