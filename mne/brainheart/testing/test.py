import pytest
import numpy as np
from typing import List, Tuple

def _onset_ends_nonoverlapping(onsets, ends): 
    """Your function - paste the actual implementation here"""
    onsets_nonoverlapping, ends_nonoverlapping = [], []
    win_idx = 0
    if not len(onsets):
        return onsets_nonoverlapping, ends_nonoverlapping
    curr_seg_start = onsets[0]
    while win_idx < len(onsets):
        win_onset = onsets[win_idx]
        win_end = ends[win_idx]
        window_mask_to_fuse = (onsets >= win_onset) & (onsets <= win_end) & (win_end < ends)
        if not np.any(window_mask_to_fuse):
            onsets_nonoverlapping.append(curr_seg_start)
            ends_nonoverlapping.append(win_end)
            #Next segment
            mask_for_next_seg = win_end < onsets
            if not np.any(mask_for_next_seg):
                break
            win_idx = np.where(mask_for_next_seg)[0][0]
            if win_idx < len(onsets):
                curr_seg_start = onsets[win_idx]
        else:
            win_idx = np.argmax(ends*window_mask_to_fuse)
    onsets_nonoverlapping = np.array(onsets_nonoverlapping)
    ends_nonoverlapping = np.array(ends_nonoverlapping)
    return onsets_nonoverlapping, ends_nonoverlapping


class TestOnsetEndsNonoverlapping:
    """Aggressive test suite for _onset_ends_nonoverlapping function"""
    
    def test_empty_arrays(self):
        """Test with empty input arrays"""
        onsets = np.array([])
        ends = np.array([])
        result_onsets, result_ends = _onset_ends_nonoverlapping(onsets, ends)
        assert len(result_onsets) == 0
        assert len(result_ends) == 0
    
    def test_single_segment(self):
        """Test with single segment"""
        onsets = np.array([1.0])
        ends = np.array([3.0])
        result_onsets, result_ends = _onset_ends_nonoverlapping(onsets, ends)
        assert len(result_onsets) == 1
        assert len(result_ends) == 1
        # This will likely fail due to curr_seg_start=0 bug
        assert result_onsets[0] == 1.0, f"Expected 1.0, got {result_onsets[0]}"
        assert result_ends[0] == 3.0
    
    def test_two_non_overlapping_segments(self):
        """Test with two clearly separated segments"""
        onsets = np.array([0, 5])
        ends = np.array([2, 7])
        result_onsets, result_ends = _onset_ends_nonoverlapping(onsets, ends)
        assert len(result_onsets) == 2
        assert len(result_ends) == 2
        assert result_onsets[0] == 0
        assert result_ends[0] == 2
        assert result_onsets[1] == 5
        assert result_ends[1] == 7
    
    def test_two_overlapping_segments(self):
        """Test with two overlapping segments"""
        onsets = np.array([0, 2])
        ends = np.array([4, 6])
        result_onsets, result_ends = _onset_ends_nonoverlapping(onsets, ends)
        # Should merge into single segment [0, 6]
        assert len(result_onsets) == 1, f"Expected 1 segment, got {len(result_onsets)}"
        assert len(result_ends) == 1
        assert result_onsets[0] == 0
        assert result_ends[0] == 6
    
    def test_nested_segments(self):
        """Test when one segment is completely inside another"""
        onsets = np.array([0, 1])
        ends = np.array([5, 3])
        result_onsets, result_ends = _onset_ends_nonoverlapping(onsets, ends)
        # Should result in single segment [0, 5]
        assert len(result_onsets) == 1
        assert result_onsets[0] == 0
        assert result_ends[0] == 5
    
    def test_touching_segments(self):
        """Test segments that touch at boundaries"""
        onsets = np.array([0, 2, 4])
        ends = np.array([2, 4, 6])
        result_onsets, result_ends = _onset_ends_nonoverlapping(onsets, ends)
        # Touching segments might or might not be considered overlapping
        # This tests the boundary condition behavior
        assert result_onsets == np.array([0])
        assert result_ends == np.array([6])
    
    def test_multiple_overlapping_chain(self):
        """Test chain of overlapping segments"""
        onsets = np.array([0, 1, 2, 3])
        ends = np.array([2, 3, 4, 5])
        result_onsets, result_ends = _onset_ends_nonoverlapping(onsets, ends)
        # Should merge into single segment
        assert len(result_onsets) == 1
        assert result_onsets[0] == 0
        assert result_ends[0] == 5
    
    def test_complex_overlapping_pattern(self):
        """Test complex overlapping pattern"""
        onsets = np.array([0, 1, 3, 7, 8])
        ends = np.array([4, 2, 5, 9, 10])
        result_onsets, result_ends = _onset_ends_nonoverlapping(onsets, ends)
        # Should handle complex merging correctly
        assert len(result_onsets) >= 1
        assert len(result_ends) >= 1
        # Check that output is valid (onsets <= ends)
        for i in range(len(result_onsets)):
            assert result_onsets[i] <= result_ends[i]
    
    def test_zero_duration_segments(self):
        """Test segments with zero duration"""
        onsets = np.array([1, 3, 5])
        ends = np.array([1, 3, 5])
        result_onsets, result_ends = _onset_ends_nonoverlapping(onsets, ends)
        # Should handle zero-duration segments
        assert len(result_onsets) >= 1
        assert len(result_ends) >= 1
    
    def test_negative_values(self):
        """Test with negative onset/end values"""
        onsets = np.array([-5, -2, 1])
        ends = np.array([-3, 0, 3])
        result_onsets, result_ends = _onset_ends_nonoverlapping(onsets, ends)
        assert len(result_onsets) >= 1
        assert len(result_ends) >= 1
    
    def test_floating_point_precision(self):
        """Test floating point edge cases"""
        onsets = np.array([0.1, 0.2, 0.3])
        ends = np.array([0.15, 0.25, 0.35])
        result_onsets, result_ends = _onset_ends_nonoverlapping(onsets, ends)
        assert len(result_onsets) >= 1
        assert len(result_ends) >= 1
    
    def test_large_arrays(self):
        """Test with large arrays to check performance/memory"""
        n = 1000
        onsets = np.linspace(0, 100, n)
        ends = onsets + 0.5  # Small overlaps
        result_onsets, result_ends = _onset_ends_nonoverlapping(onsets, ends)
        assert len(result_onsets) >= 1
        assert len(result_ends) >= 1
        assert len(result_onsets) == len(result_ends)
    
    def test_unsorted_input_behavior(self):
        """Test behavior with unsorted input (function assumes sorted)"""
        onsets = np.array([5, 0, 3])
        ends = np.array([7, 2, 4])
        # This should probably fail or give unexpected results
        # since function assumes sorted input
        with pytest.warns(None) as warnings:
            result_onsets, result_ends = _onset_ends_nonoverlapping(onsets, ends)
            # Just check it doesn't crash
            assert isinstance(result_onsets, list)
            assert isinstance(result_ends, list)
    
    def test_identical_segments(self):
        """Test with identical segments"""
        onsets = np.array([1, 1, 1])
        ends = np.array([3, 3, 3])
        result_onsets, result_ends = _onset_ends_nonoverlapping(onsets, ends)
        assert result_onsets == np.array([1])
        assert result_ends == np.array([3])
    
    def test_same_start(self): 
        onsets = np.array([1, 1])
        ends = np.array([3, 5])    
        result_onsets, result_ends = _onset_ends_nonoverlapping(onsets, ends)
        assert result_onsets == np.array([1])
        assert result_ends == np.array([5])
    
    def test_output_validity(self):
        """Test that output maintains basic validity constraints"""
        test_cases = [
            ([0, 2, 5], [3, 4, 7]),
            ([1, 3, 6], [2, 5, 8]),
            ([0, 1, 2, 3], [2, 3, 4, 5]),
        ]
        
        for onsets, ends in test_cases:
            result_onsets, result_ends = _onset_ends_nonoverlapping(
                np.array(onsets), np.array(ends)
            )
            
            # Basic validity checks
            assert len(result_onsets) == len(result_ends), "Mismatched output lengths"
            
            # Each segment should have onset <= end
            for i in range(len(result_onsets)):
                assert result_onsets[i] <= result_ends[i], \
                    f"Invalid segment: onset {result_onsets[i]} > end {result_ends[i]}"
            
            # Output segments should be non-overlapping and sorted
            for i in range(len(result_onsets) - 1):
                assert result_ends[i] <= result_onsets[i + 1], \
                    f"Overlapping output segments: [{result_onsets[i]}, {result_ends[i]}] and [{result_onsets[i+1]}, {result_ends[i+1]}]"
    
    def test_edge_case_argmax_zero(self):
        """Test edge case where argmax could return 0 and cause infinite loop"""
        onsets = np.array([0, 10])
        ends = np.array([1, 11])
        # This should not cause infinite loop
        result_onsets, result_ends = _onset_ends_nonoverlapping(onsets, ends)
        assert len(result_onsets) >= 1
        assert len(result_ends) >= 1
    
    def test_curr_seg_start_initialization_bug(self):
        """Specific test for the curr_seg_start=0 initialization bug"""
        onsets = np.array([5, 10, 15])
        ends = np.array([7, 12, 17])
        result_onsets, result_ends = _onset_ends_nonoverlapping(onsets, ends)
        
        # This will fail with current implementation
        assert result_onsets[0] == 5, \
            f"First onset should be 5, got {result_onsets[0]}. curr_seg_start initialization bug!"
    
    def test_mask_logic_specific_case(self):
        """Test specific case that exposes mask logic issues"""
        onsets = np.array([0, 2, 5])
        ends = np.array([4, 6, 7])
        result_onsets, result_ends = _onset_ends_nonoverlapping(onsets, ends)
        
        # Check if the mask logic properly identifies overlaps
        # Segment [0,4] overlaps with [2,6]: onset 2 is in [0,4]
        # Segment [2,6] overlaps with [5,7]: onset 5 is in [2,6]
        # Should result in single merged segment [0,7]
        
        expected_merged = True  # These should all merge
        if expected_merged:
            assert len(result_onsets) == 1, f"Expected 1 merged segment, got {len(result_onsets)}"
            assert result_onsets[0] == 0
            assert result_ends[0] == 7
    
    @pytest.mark.parametrize("onsets,ends,expected_count", [
        ([0], [1], 1),
        ([0, 5], [2, 7], 2),  # Non-overlapping
        ([0, 1], [3, 2], 1),  # Overlapping
        ([0, 1, 2], [3, 4, 5], 1),  # Chain overlap
        ([0, 5, 10], [2, 7, 12], 3),  # All separate
    ])
    def test_parametrized_cases(self, onsets, ends, expected_count):
        """Parametrized tests for various input patterns"""
        result_onsets, result_ends = _onset_ends_nonoverlapping(
            np.array(onsets), np.array(ends)
        )
        assert len(result_onsets) == expected_count, \
            f"Expected {expected_count} segments, got {len(result_onsets)}"
        assert len(result_ends) == expected_count
    
    def test_infinite_loop_prevention(self):
        """Test cases that could cause infinite loops"""
        # Case where argmax might return same index
        onsets = np.array([0, 1])
        ends = np.array([0.5, 2])
        
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Function took too long - likely infinite loop")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)  # 5 second timeout
        
        try:
            result_onsets, result_ends = _onset_ends_nonoverlapping(onsets, ends)
            signal.alarm(0)  # Cancel alarm
            assert len(result_onsets) >= 1
            assert len(result_ends) >= 1
        except TimeoutError:
            pytest.fail("Function appears to have infinite loop")
        finally:
            signal.alarm(0)
    
    def test_type_consistency(self):
        """Test that function handles different numpy dtypes"""
        test_dtypes = [np.int32, np.int64, np.float32, np.float64]
        
        for dtype in test_dtypes:
            onsets = np.array([0, 2, 5], dtype=dtype)
            ends = np.array([3, 4, 7], dtype=dtype)
            result_onsets, result_ends = _onset_ends_nonoverlapping(onsets, ends)
            assert len(result_onsets) >= 1
            assert len(result_ends) >= 1
    
    def test_very_close_values(self):
        """Test with values very close to each other (floating point edge cases)"""
        eps = np.finfo(float).eps
        onsets = np.array([0, eps, 2*eps])
        ends = np.array([eps, 2*eps, 3*eps])
        result_onsets, result_ends = _onset_ends_nonoverlapping(onsets, ends)
        assert len(result_onsets) >= 1
        assert len(result_ends) >= 1
    
    def test_descending_ends(self):
        """Test case where ends are not in ascending order"""
        onsets = np.array([0, 1, 2])
        ends = np.array([5, 3, 4])
        result_onsets, result_ends = _onset_ends_nonoverlapping(onsets, ends)
        assert len(result_onsets) >= 1
        assert len(result_ends) >= 1
    
    def test_output_properties_invariants(self):
        """Test that output maintains required invariants"""
        test_cases = [
            ([0, 2, 5], [3, 4, 7]),
            ([1, 3, 6], [2, 5, 8]),
            ([0, 1, 2, 3, 4], [1.5, 2.5, 3.5, 4.5, 5.5]),
            ([10, 20, 30], [15, 25, 35]),
        ]
        
        for onsets, ends in test_cases:
            result_onsets, result_ends = _onset_ends_nonoverlapping(
                np.array(onsets), np.array(ends)
            )
            
            # Invariant 1: Equal length outputs
            assert len(result_onsets) == len(result_ends)
            
            # Invariant 2: Each segment valid (onset <= end)
            for i in range(len(result_onsets)):
                assert result_onsets[i] <= result_ends[i], \
                    f"Invalid segment {i}: onset > end"
            
            # Invariant 3: No overlapping output segments
            for i in range(len(result_onsets) - 1):
                assert result_ends[i] <= result_onsets[i + 1], \
                    f"Output segments {i} and {i+1} overlap"
            
            # Invariant 4: Output segments should be sorted
            for i in range(len(result_onsets) - 1):
                assert result_onsets[i] <= result_onsets[i + 1], \
                    "Output onsets not sorted"
                assert result_ends[i] <= result_ends[i + 1], \
                    "Output ends not sorted"
    
    def test_extreme_values(self):
        """Test with extreme values"""
        # Very large values
        onsets = np.array([1e6, 1e6 + 1])
        ends = np.array([1e6 + 0.5, 1e6 + 2])
        result_onsets, result_ends = _onset_ends_nonoverlapping(onsets, ends)
        assert len(result_onsets) >= 1
        
        # Very small positive values
        onsets = np.array([1e-10, 2e-10])
        ends = np.array([1.5e-10, 3e-10])
        result_onsets, result_ends = _onset_ends_nonoverlapping(onsets, ends)
        assert len(result_onsets) >= 1
    
    def test_curr_seg_start_bug_comprehensive(self):
        """Comprehensive test for curr_seg_start initialization bug"""
        test_cases = [
            ([1], [2]),  # Single segment not starting at 0
            ([5, 10], [7, 12]),  # Multiple segments not starting at 0
            ([100, 200, 300], [150, 250, 350]),  # Large starting values
        ]
        
        for onsets, ends in test_cases:
            result_onsets, result_ends = _onset_ends_nonoverlapping(
                np.array(onsets), np.array(ends)
            )
            
            # First output onset should match first input onset (when no merging)
            if len(onsets) == len(result_onsets):  # No merging occurred
                assert result_onsets[0] == onsets[0], \
                    f"curr_seg_start bug: expected {onsets[0]}, got {result_onsets[0]}"
    
    def test_argmax_edge_cases(self):
        """Test edge cases for argmax behavior"""
        # Case where all mask values are False
        onsets = np.array([0, 10, 20])
        ends = np.array([1, 11, 21])
        result_onsets, result_ends = _onset_ends_nonoverlapping(onsets, ends)
        assert len(result_onsets) == 3  # Should be separate segments
    
    def test_stress_random_inputs(self):
        """Stress test with random inputs"""
        np.random.seed(42)  # Reproducible
        
        for _ in range(50):  # Run many random tests
            n = np.random.randint(1, 20)
            onsets = np.sort(np.random.uniform(0, 100, n))
            ends = onsets + np.random.uniform(0.1, 10, n)
            
            try:
                result_onsets, result_ends = _onset_ends_nonoverlapping(onsets, ends)
                
                # Basic sanity checks
                assert len(result_onsets) >= 1
                assert len(result_ends) >= 1
                assert len(result_onsets) == len(result_ends)
                
                # Check validity
                for i in range(len(result_onsets)):
                    assert result_onsets[i] <= result_ends[i]
                    
            except Exception as e:
                pytest.fail(f"Function failed on random input: onsets={onsets}, ends={ends}. Error: {e}")
    
    def test_memory_and_performance(self):
        """Test memory usage and basic performance"""
        # Large input test
        n = 10000
        onsets = np.arange(n, dtype=np.float64)
        ends = onsets + 0.1
        
        import time
        start_time = time.time()
        result_onsets, result_ends = _onset_ends_nonoverlapping(onsets, ends)
        execution_time = time.time() - start_time
        
        # Should complete in reasonable time (adjust threshold as needed)
        assert execution_time < 10.0, f"Function too slow: {execution_time:.2f}s"
        assert len(result_onsets) >= 1
    
    def test_boundary_conditions_comprehensive(self):
        """Test various boundary conditions"""
        # Segments that barely touch
        onsets = np.array([0, 1.0])
        ends = np.array([1.0, 2.0])
        result_onsets, result_ends = _onset_ends_nonoverlapping(onsets, ends)
        # Behavior depends on whether touching counts as overlapping
        
        # Segments with same onset, different ends
        onsets = np.array([0, 0])
        ends = np.array([1, 2])
        result_onsets, result_ends = _onset_ends_nonoverlapping(onsets, ends)
        assert len(result_onsets) >= 1
        
        # Segments with different onsets, same end
        onsets = np.array([0, 1])
        ends = np.array([2, 2])
        result_onsets, result_ends = _onset_ends_nonoverlapping(onsets, ends)
        assert len(result_onsets) >= 1


    def test_custom(self): 
        onsets = np.array([0, 1, 2, 4, 7, 8, 15])
        ends = np.array([5, 2, 3, 6, 12, 13, 16])
        result_onsets, result_ends = _onset_ends_nonoverlapping(onsets, ends)
        # Behavior depends on whether touching counts as overlapping
        print(result_onsets)
        print(result_ends)
        assert result_onsets == np.array([0, 7, 15], dtype=int)
        assert result_ends == np.array([6, 13, 16], dtype=int)

# Additional utility tests
class TestFunctionRobustness:
    """Tests for robustness and error handling"""
    
    def test_mismatched_array_lengths(self):
        """Test behavior with mismatched input array lengths"""
        onsets = np.array([0, 1, 2])
        ends = np.array([1, 2])  # Shorter array
        
        with pytest.raises((IndexError, ValueError)):
            _onset_ends_nonoverlapping(onsets, ends)
    
    def test_invalid_segments_onset_gt_end(self):
        """Test with invalid segments where onset > end"""
        onsets = np.array([3, 1])
        ends = np.array([1, 3])
        
        # Function might crash or give unexpected results
        try:
            result_onsets, result_ends = _onset_ends_nonoverlapping(onsets, ends)
            # If it doesn't crash, check basic validity
            assert len(result_onsets) >= 0
            assert len(result_ends) >= 0
        except Exception:
            # It's okay if it crashes on invalid input
            pass
    
    def test_numpy_array_vs_list_input(self):
        """Test behavior with both numpy arrays and Python lists"""
        onsets_list = [0, 2, 5]
        ends_list = [3, 4, 7]
        onsets_array = np.array(onsets_list)
        ends_array = np.array(ends_list)
        
        # Both should work (or both should fail consistently)
        try:
            result1 = _onset_ends_nonoverlapping(onsets_list, ends_list)
            result2 = _onset_ends_nonoverlapping(onsets_array, ends_array)
            
            # Results should be equivalent
            np.testing.assert_array_equal(result1[0], result2[0])
            np.testing.assert_array_equal(result1[1], result2[1])
        except Exception as e:
            # Document which input types fail
            pytest.fail(f"Function failed with mixed input types: {e}")


if __name__ == "__main__":
    # Run specific failing tests to debug

    obj = TestOnsetEndsNonoverlapping()
    obj.test_empty_arrays()
    obj.test_single_segment()
    obj.test_two_non_overlapping_segments()
    obj.test_two_overlapping_segments()
    obj.test_nested_segments()
    obj.test_touching_segments()
    obj.test_multiple_overlapping_chain()
    obj.test_complex_overlapping_pattern()
    obj.test_zero_duration_segments()
    obj.test_negative_values()
    obj.test_floating_point_precision()
    obj.test_large_arrays()
    obj.test_identical_segments()
    obj.test_same_start()
    obj.test_output_validity()
    obj.test_edge_case_argmax_zero()
    obj.test_custom()
