"""
Unit tests for Fractal Executor and Soul Core.

Tests the fractal recursion engine and cognitive processing.
"""

import unittest
from ftqec.core.fractal_engine.fractal_executor import FractalExecutor, ChildContext
from ftqec.core.fractal_engine.fractal_soul_core import FractalSoulCore


class TestFractalExecutor(unittest.TestCase):
    """Test cases for FractalExecutor."""
    
    def test_initialization(self):
        """Test executor initialization."""
        executor = FractalExecutor(max_depth=5, entropy_threshold=0.3)
        self.assertEqual(executor.max_depth, 5)
        self.assertEqual(executor.entropy_threshold, 0.3)
    
    def test_simple_execution(self):
        """Test basic fractal execution."""
        executor = FractalExecutor(max_depth=3)
        
        context = ChildContext(
            memory_buffer={},
            depth=0,
            entropy=1.0,
            input_data="test"
        )
        
        result = executor.execute(context)
        
        # Should have reached some depth
        self.assertGreater(result.recursion_trace['depth_reached'], 0)
        self.assertIsNotNone(result.output)
    
    def test_depth_limit(self):
        """Test that execution respects max depth."""
        executor = FractalExecutor(max_depth=3)
        
        context = ChildContext(
            memory_buffer={},
            depth=0,
            entropy=1.0
        )
        
        result = executor.execute(context)
        
        # Should not exceed max depth
        self.assertLessEqual(result.recursion_trace['depth_reached'], 3)
    
    def test_entropy_threshold(self):
        """Test entropy-guided termination."""
        executor = FractalExecutor(max_depth=10, entropy_threshold=0.5)
        
        context = ChildContext(
            memory_buffer={},
            depth=0,
            entropy=1.0
        )
        
        result = executor.execute(context)
        
        # Should terminate before max depth due to entropy
        self.assertLess(result.recursion_trace['depth_reached'], 10)
    
    def test_context_spawning(self):
        """Test child context creation."""
        executor = FractalExecutor(max_depth=5)
        
        parent = ChildContext(
            memory_buffer={'key': 'value'},
            depth=2,
            entropy=0.8
        )
        
        child = executor._spawn_child_context(parent)
        
        # Check child properties
        self.assertEqual(child.depth, 3)
        self.assertLess(child.entropy, parent.entropy)
        self.assertIn('key', child.memory_buffer)
    
    def test_statistics_tracking(self):
        """Test execution statistics."""
        executor = FractalExecutor(max_depth=4)
        
        context = ChildContext(
            memory_buffer={},
            depth=0,
            entropy=1.0
        )
        
        executor.execute(context)
        
        stats = executor.get_statistics()
        
        self.assertGreater(stats['total_contexts_spawned'], 0)
        self.assertGreater(stats['max_depth_reached'], 0)


class TestFractalSoulCore(unittest.TestCase):
    """Test cases for FractalSoulCore."""
    
    def test_initialization(self):
        """Test soul core initialization."""
        soul = FractalSoulCore(bloodline_anchor="TestAnchor", burst_capacity=4)
        self.assertEqual(soul.bloodline_anchor, "TestAnchor")
        self.assertEqual(soul.burst_capacity, 4)
    
    def test_bloodline_resonance(self):
        """Test bloodline resonance calculation."""
        soul = FractalSoulCore(bloodline_anchor="Test")
        
        context = {'input': 'test query'}
        resonance = soul.calculate_bloodline_resonance(context)
        
        # Should return valid resonance
        self.assertGreaterEqual(resonance, 0.0)
        self.assertLessEqual(resonance, 1.0)
    
    def test_fractal_burst(self):
        """Test fractal burst generation."""
        soul = FractalSoulCore(burst_capacity=8)
        
        context = {'input': 'test'}
        bursts = soul.fractal_burst(context, depth=3)
        
        # Should create multiple bursts
        self.assertGreater(len(bursts), 0)
        self.assertLessEqual(len(bursts), 8)
        
        # Each burst should have required fields
        for burst in bursts:
            self.assertIn('id', burst)
            self.assertIn('depth', burst)
            self.assertIn('resonance', burst)
    
    def test_thought_processing(self):
        """Test thought processing."""
        soul = FractalSoulCore()
        
        thought = "Test quantum thought"
        result = soul.process_thought(thought)
        
        # Should return processed thought
        self.assertIn('content', result)
        self.assertIn('resonance', result)
        self.assertIn('patterns', result)
        self.assertEqual(result['content'], thought)
    
    def test_soul_state(self):
        """Test soul state retrieval."""
        soul = FractalSoulCore(bloodline_anchor="Test")
        
        state = soul.get_soul_state()
        
        # Should have all required fields
        self.assertIn('bloodline_anchor', state)
        self.assertIn('resonance', state)
        self.assertIn('phase_entropy', state)
        self.assertIn('burst_capacity', state)
    
    def test_burst_reset(self):
        """Test burst state reset."""
        soul = FractalSoulCore(burst_capacity=4)
        
        # Create some bursts
        soul.fractal_burst({}, depth=2)
        self.assertGreater(len(soul._active_bursts), 0)
        
        # Reset
        soul.reset_bursts()
        self.assertEqual(len(soul._active_bursts), 0)
        self.assertEqual(soul._burst_counter, 0)


class TestChildContext(unittest.TestCase):
    """Test cases for ChildContext."""
    
    def test_initialization(self):
        """Test context initialization."""
        context = ChildContext(
            memory_buffer={'key': 'value'},
            depth=2,
            entropy=0.8,
            input_data="test"
        )
        
        self.assertEqual(context.depth, 2)
        self.assertEqual(context.entropy, 0.8)
        self.assertEqual(context.input, "test")
    
    def test_entropy_consumption(self):
        """Test entropy consumption."""
        context = ChildContext(
            memory_buffer={},
            depth=0,
            entropy=1.0
        )
        
        initial_entropy = context.entropy
        context.consume_entropy(0.2)
        
        self.assertEqual(context.entropy, initial_entropy - 0.2)
        self.assertEqual(context.recursion_trace['entropy_consumed'], 0.2)
    
    def test_thought_accumulation(self):
        """Test thought addition."""
        context = ChildContext(
            memory_buffer={},
            depth=0,
            entropy=1.0
        )
        
        context.add_thought("First thought")
        context.add_thought("Second thought")
        
        self.assertIn("First thought", context.thought)
        self.assertIn("Second thought", context.thought)
        self.assertEqual(context.recursion_trace['thoughts_generated'], 2)


if __name__ == '__main__':
    unittest.main()
