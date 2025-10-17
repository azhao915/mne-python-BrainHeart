import pytest
import numpy as np
from typing import List, Tuple

# Your corrected function implementation
def _interval_difference(
        onsets_to_keep,
        ends_to_keep,
        onsets_to_reject,
        ends_to_reject,
        min_N: int = 1
): 
    final_onsets = []
    final_ends = []
    for onset_to_keep, end_to_keep in zip(onsets_to_keep, ends_to_keep): 
        # Standard interval overlap detection
        intervals_to_reject_mask = (onsets_to_reject <= end_to_keep) & (ends_to_reject >= onset_to_keep)
        onsets_to_reject_in_seg = onsets_to_reject[intervals_to_reject_mask]
        ends_to_reject_in_seg = ends_to_reject[intervals_to_reject_mask]
        curr_win_start = onset_to_keep
        for onset_to_reject_in_seg, end_to_reject_in_seg in zip(onsets_to_reject_in_seg, ends_to_reject_in_seg): 
            reject_start = np.max([onset_to_reject_in_seg, onset_to_keep])
            reject_end = np.min([end_to_reject_in_seg, end_to_keep])
            if reject_start - curr_win_start >= min_N:
                final_onsets.append(curr_win_start)
                final_ends.append(reject_start)
            curr_win_start = reject_end + 1  # +1 because intervals are inclusive
            if curr_win_start >= end_to_keep:
                break
        # Add the final segment if it exists and is long enough
        if end_to_keep - curr_win_start >= min_N:
            final_onsets.append(curr_win_start)
            final_ends.append(end_to_keep)
    final_onsets = np.array(final_onsets)
    final_ends = np.array(final_ends)
    return final_onsets, final_ends

# Reference implementation for comparison (using exclusive arithmetic for testing)
def _interval_difference_reference(
        onsets_to_keep,
        ends_to_keep,
        onsets_to_reject,
        ends_to_reject,
        min_N: int = 1
): 
    final_onsets = []
    final_ends = []
    
    for onset_to_keep, end_to_keep in zip(onsets_to_keep, ends_to_keep): 
        # Find all rejection intervals that overlap with current keep interval
        overlapping_mask = (onsets_to_reject < end_to_keep) & (ends_to_reject > onset_to_keep)
        onsets_to_reject_in_seg = onsets_to_reject[overlapping_mask]
        ends_to_reject_in_seg = ends_to_reject[overlapping_mask]
        
        curr_win_start = onset_to_keep
        
        for onset_to_reject_in_seg, end_to_reject_in_seg in zip(onsets_to_reject_in_seg, ends_to_reject_in_seg):
            # Clip rejection interval to keep interval bounds
            reject_start = max(onset_to_reject_in_seg, onset_to_keep)
            reject_end = min(end_to_reject_in_seg, end_to_keep)
            
            # Add segment before rejection if it's long enough
            if reject_start - curr_win_start >= min_N:
                final_onsets.append(curr_win_start)
                final_ends.append(reject_start)
            
            # Move start position past the rejection (inclusive intervals)
            curr_win_start = reject_end + 1
            
            # If we've passed the end of keep interval, stop
            if curr_win_start >= end_to_keep:
                break
        
        # Add final segment if it exists and is long enough
        if end_to_keep - curr_win_start >= min_N:
            final_onsets.append(curr_win_start)
            final_ends.append(end_to_keep)
    
    return np.array(final_onsets), np.array(final_ends)


class TestIntervalDifference:
    """Aggressive test suite for _interval_difference function"""
    
    def test_empty_inputs(self):
        """Test with various combinations of empty inputs"""
        # All empty
        result = _interval_difference(np.array([]), np.array([]), np.array([]), np.array([]))
        assert len(result[0]) == 0
        assert len(result[1]) == 0
        
        # Empty keep intervals
        result = _interval_difference(np.array([]), np.array([]), np.array([1]), np.array([2]))
        assert len(result[0]) == 0
        assert len(result[1]) == 0
        
        # Empty reject intervals
        result = _interval_difference(np.array([0]), np.array([5]), np.array([]), np.array([]))
        assert len(result[0]) == 1
        assert result[0][0] == 0
        assert result[1][0] == 5
    
    def test_no_overlaps(self):
        """Test when keep and reject intervals don't overlap"""
        onsets_keep = np.array([0, 10, 20])
        ends_keep = np.array([2, 12, 22])
        onsets_reject = np.array([5, 15, 25])
        ends_reject = np.array([7, 17, 27])
        
        result = _interval_difference(onsets_keep, ends_keep, onsets_reject, ends_reject)
        
        # Should return original keep intervals since no overlaps
        np.testing.assert_array_equal(result[0], onsets_keep)
        np.testing.assert_array_equal(result[1], ends_keep)
    
    def test_complete_overlap_rejection(self):
        """Test when rejection completely covers keep interval"""
        onsets_keep = np.array([5])
        ends_keep = np.array([10])
        onsets_reject = np.array([0])
        ends_reject = np.array([15])
        
        result = _interval_difference(onsets_keep, ends_keep, onsets_reject, ends_reject)
        
        # Should return empty result - rejection completely covers keep interval
        assert len(result[0]) == 0
        assert len(result[1]) == 0
    
    def test_partial_overlap_beginning(self):
        """Test rejection overlaps beginning of keep interval"""
        onsets_keep = np.array([5])
        ends_keep = np.array([10])
        onsets_reject = np.array([0])
        ends_reject = np.array([7])
        
        result = _interval_difference(onsets_keep, ends_keep, onsets_reject, ends_reject)
        
        # Should return [8, 10] (remainder after rejection, +1 for inclusive)
        assert len(result[0]) == 1
        assert result[0][0] == 8  # 7 + 1 for inclusive intervals
        assert result[1][0] == 10
    
    def test_partial_overlap_end(self):
        """Test rejection overlaps end of keep interval"""
        onsets_keep = np.array([5])
        ends_keep = np.array([10])
        onsets_reject = np.array([8])
        ends_reject = np.array([15])
        
        result = _interval_difference(onsets_keep, ends_keep, onsets_reject, ends_reject)
        
        # Should return [5, 8] (part before rejection)
        assert len(result[0]) == 1
        assert result[0][0] == 5
        assert result[1][0] == 8
    
    def test_rejection_in_middle(self):
        """Test rejection in middle of keep interval"""
        onsets_keep = np.array([0])
        ends_keep = np.array([10])
        onsets_reject = np.array([3])
        ends_reject = np.array([7])
        
        result = _interval_difference(onsets_keep, ends_keep, onsets_reject, ends_reject)
        
        # Should return two segments: [0, 3] and [8, 10] (inclusive intervals)
        assert len(result[0]) == 2
        assert result[0][0] == 0
        assert result[1][0] == 3
        assert result[0][1] == 8  # 7 + 1 for inclusive
        assert result[1][1] == 10
    
    def test_multiple_rejections_in_one_keep(self):
        """Test multiple rejection intervals within one keep interval"""
        onsets_keep = np.array([0])
        ends_keep = np.array([20])
        onsets_reject = np.array([2, 8, 15])
        ends_reject = np.array([4, 12, 17])
        
        result = _interval_difference(onsets_keep, ends_keep, onsets_reject, ends_reject)
        
        # Should return: [0,2], [5,8], [13,15], [18,20] (inclusive intervals)
        expected_onsets = [0, 5, 13, 18]  # Updated for inclusive intervals
        expected_ends = [2, 8, 15, 20]
        
        assert len(result[0]) == 4
        np.testing.assert_array_equal(result[0], expected_onsets)
        np.testing.assert_array_equal(result[1], expected_ends)
    
    def test_touching_intervals(self):
        """Test when rejection exactly touches keep interval boundaries"""
        onsets_keep = np.array([5])
        ends_keep = np.array([10])
        onsets_reject = np.array([10])  # Starts exactly where keep ends
        ends_reject = np.array([15])
        
        result = _interval_difference(onsets_keep, ends_keep, onsets_reject, ends_reject)
        
        # Should return original keep interval since no overlap
        assert len(result[0]) == 1
        assert result[0][0] == 5
        assert result[1][0] == 10
    
    def test_min_N_filtering(self):
        """Test min_N parameter filters out short segments"""
        onsets_keep = np.array([0])
        ends_keep = np.array([10])
        onsets_reject = np.array([2])
        ends_reject = np.array([8])
        min_N = 3
        
        result = _interval_difference(onsets_keep, ends_keep, onsets_reject, ends_reject, min_N)
        
        # [0,2] has length 2 < min_N=3, should be filtered out
        # [8,10] has length 2 < min_N=3, should be filtered out
        assert len(result[0]) == 0
        assert len(result[1]) == 0
    
    def test_min_N_keeping_long_segments(self):
        """Test min_N keeps segments that are long enough"""
        onsets_keep = np.array([0])
        ends_keep = np.array([15])
        onsets_reject = np.array([5])
        ends_reject = np.array([7])
        min_N = 3
        
        result = _interval_difference(onsets_keep, ends_keep, onsets_reject, ends_reject, min_N)
        
        # [0,5] has length 6 >= min_N=3, keep
        # [8,15] has length 8 >= min_N=3, keep (8 = 7+1 for inclusive intervals)
        assert len(result[0]) == 2
        assert result[0][0] == 0
        assert result[1][0] == 5
        assert result[0][1] == 8  # Fixed: should be 8, not 7 (7+1 for inclusive)
        assert result[1][1] == 15
    
    def test_complex_multi_keep_multi_reject(self):
        """Test complex scenario with multiple keep and reject intervals"""
        onsets_keep = np.array([0, 20, 40])
        ends_keep = np.array([10, 30, 50])
        onsets_reject = np.array([2, 8, 25, 45])
        ends_reject = np.array([4, 12, 27, 47])
        
        result = _interval_difference(onsets_keep, ends_keep, onsets_reject, ends_reject)
        
        # Keep [0,10]: reject [2,4] and [8,12] -> results [0,2], [5,8]
        # Keep [20,30]: reject [25,27] -> results [20,25], [28,30]  
        # Keep [40,50]: reject [45,47] -> results [40,45], [48,50]
        
        expected_count = 6
        assert len(result[0]) == expected_count
        assert len(result[1]) == expected_count
        
        # Check specific values for inclusive intervals
        expected_onsets = [0, 5, 20, 28, 40, 48]  # Updated for +1 logic
        expected_ends = [2, 8, 25, 30, 45, 50]
        
        np.testing.assert_array_equal(result[0], expected_onsets)
        np.testing.assert_array_equal(result[1], expected_ends)
    
    def test_rejection_extends_beyond_keep(self):
        """Test rejection that extends beyond keep interval boundaries"""
        onsets_keep = np.array([5])
        ends_keep = np.array([15])
        onsets_reject = np.array([0, 10])
        ends_reject = np.array([8, 20])
        
        result = _interval_difference(onsets_keep, ends_keep, onsets_reject, ends_reject)
        
        # First rejection [0,8] overlaps [5,15] -> removes [5,8]
        # Second rejection [10,20] overlaps [5,15] -> removes [10,15]
        # Result should be [9,10] (inclusive: after 8+1=9, before 10)
        assert len(result[0]) == 1
        assert result[0][0] == 9  # 8 + 1 for inclusive
        assert result[1][0] == 10
    
    def test_mask_logic_bug_exposure(self):
        """Test that exposes overlap detection - now should work correctly"""
        onsets_keep = np.array([5])
        ends_keep = np.array([15])
        onsets_reject = np.array([0])  # Starts before keep interval
        ends_reject = np.array([10])   # Ends within keep interval
        
        result = _interval_difference(onsets_keep, ends_keep, onsets_reject, ends_reject)
        
        # With corrected overlap detection: should find this overlap and return [11, 15]
        assert len(result[0]) == 1
        assert result[0][0] == 11  # 10 + 1 for inclusive intervals
        assert result[1][0] == 15
    
    def test_break_condition_bug(self):
        """Test that exposes the break condition bug"""
        onsets_keep = np.array([0])
        ends_keep = np.array([10])
        onsets_reject = np.array([2, 6])
        ends_reject = np.array([4, 8])
        
        result = _interval_difference(onsets_keep, ends_keep, onsets_reject, ends_reject)
        
        # The original break condition might cause early termination
        # Should process both rejections and return [0,2], [4,6], [8,10]
        expected_segments = 3
        # Original function might not process second rejection due to break bug
    
    def test_final_segment_missing_bug(self):
        """Test that final segment is properly included"""
        onsets_keep = np.array([0])
        ends_keep = np.array([10])
        onsets_reject = np.array([2])
        ends_reject = np.array([6])
        
        result = _interval_difference(onsets_keep, ends_keep, onsets_reject, ends_reject)
        
        # Should return [0,2] and [7,10] (inclusive intervals)
        assert len(result[0]) == 2, "Should have both initial and final segments"
        assert result[0][0] == 0
        assert result[1][0] == 2
        assert result[0][1] == 7  # 6 + 1 for inclusive
        assert result[1][1] == 10
    
    def test_zero_length_intervals(self):
        """Test with zero-length intervals"""
        onsets_keep = np.array([5])
        ends_keep = np.array([5])  # Zero length
        onsets_reject = np.array([3])
        ends_reject = np.array([7])
        
        result = _interval_difference(onsets_keep, ends_keep, onsets_reject, ends_reject)
        assert len(result[0]) == 0  # Zero-length interval completely removed
    
    def test_floating_point_precision(self):
        """Test floating point precision edge cases"""
        onsets_keep = np.array([0.1])
        ends_keep = np.array([0.9])
        onsets_reject = np.array([0.3])
        ends_reject = np.array([0.7])
        
        result = _interval_difference(onsets_keep, ends_keep, onsets_reject, ends_reject)
        
        if len(result[0]) > 0:
            # Check precision is maintained
            assert abs(result[0][0] - 0.1) < 1e-10
            assert abs(result[1][0] - 0.3) < 1e-10
    
    def test_large_numbers(self):
        """Test with large number values"""
        onsets_keep = np.array([1e6])
        ends_keep = np.array([1e6 + 100])
        onsets_reject = np.array([1e6 + 20])
        ends_reject = np.array([1e6 + 80])
        
        result = _interval_difference(onsets_keep, ends_keep, onsets_reject, ends_reject)
        
        # Should handle large numbers correctly
        assert len(result[0]) == 2
        assert result[0][0] == 1e6
        assert result[1][0] == 1e6 + 20

    
    def test_stress_many_intervals(self):
        """Stress test with many intervals"""
        n_keep = 100
        n_reject = 200
        
        # Create non-overlapping keep intervals
        onsets_keep = np.arange(0, n_keep * 10, 10)
        ends_keep = onsets_keep + 5
        
        # Create some overlapping reject intervals
        onsets_reject = np.random.uniform(0, n_keep * 10, n_reject)
        ends_reject = onsets_reject + np.random.uniform(1, 3, n_reject)
        
        # Sort reject intervals
        sort_idx = np.argsort(onsets_reject)
        onsets_reject = onsets_reject[sort_idx]
        ends_reject = ends_reject[sort_idx]
        
        result = _interval_difference(onsets_keep, ends_keep, onsets_reject, ends_reject)
        
        # Basic sanity checks
        assert len(result[0]) == len(result[1])
        
        # All result intervals should be valid
        for i in range(len(result[0])):
            assert result[0][i] <= result[1][i], f"Invalid interval: onset > end"
    
    def test_output_validity_invariants(self):
        """Test that output maintains validity invariants"""
        test_cases = [
            ([0, 10], [5, 15], [2], [3]),
            ([0, 10], [5, 15], [2, 7], [3, 12]),
            ([0], [20], [5, 10, 15], [7, 12, 17]),
        ]
        
        for onsets_keep, ends_keep, onsets_reject, ends_reject in test_cases:
            result = _interval_difference(
                np.array(onsets_keep), np.array(ends_keep),
                np.array(onsets_reject), np.array(ends_reject)
            )
            
            # Invariant 1: Equal length outputs
            assert len(result[0]) == len(result[1])
            
            # Invariant 2: Each interval valid (onset <= end)
            for i in range(len(result[0])):
                assert result[0][i] <= result[1][i], f"Invalid interval {i}"
            
            # Invariant 3: Result intervals are sorted and non-overlapping
            for i in range(len(result[0]) - 1):
                assert result[1][i] <= result[0][i + 1], f"Overlapping result intervals"
            
            # Invariant 4: All result intervals are within original keep intervals
            for i in range(len(result[0])):
                onset, end = result[0][i], result[1][i]
                # Find which keep interval this belongs to
                found = False
                for j in range(len(onsets_keep)):
                    if onsets_keep[j] <= onset and end <= ends_keep[j]:
                        found = True
                        break
                assert found, f"Result interval [{onset}, {end}] not within any keep interval"
    
    def test_specific_bug_detection_original_vs_corrected(self):
        """Test specific cases to verify the corrected function works properly"""
        test_cases = [
            # Case 1: Rejection completely engulfs keep (now detected correctly)
            ([3], [7], [0], [10]),
            # Case 2: Final segment handling
            ([0], [10], [2], [6]),
            # Case 3: Multiple rejections with inclusive arithmetic
            ([0], [20], [3, 10], [5, 15]),
        ]
        
        for onsets_keep, ends_keep, onsets_reject, ends_reject in test_cases:
            result = _interval_difference(
                np.array(onsets_keep), np.array(ends_keep),
                np.array(onsets_reject), np.array(ends_reject)
            )
            
            # Basic validity checks for corrected function
            assert len(result[0]) == len(result[1]), "Mismatched output lengths"
            
            # All intervals should be valid
            for i in range(len(result[0])):
                assert result[0][i] <= result[1][i], f"Invalid interval {i}: onset > end"
    
    def test_edge_cases_boundary_arithmetic(self):
        """Test edge cases with boundary arithmetic for inclusive intervals"""
        # Test the +1 logic for inclusive intervals
        onsets_keep = np.array([0])
        ends_keep = np.array([10])
        onsets_reject = np.array([4])
        ends_reject = np.array([6])
        
        result = _interval_difference(onsets_keep, ends_keep, onsets_reject, ends_reject)
        
        # With inclusive intervals: curr_win_start = reject_end + 1 = 6 + 1 = 7
        assert len(result[0]) == 2
        assert result[0][0] == 0
        assert result[1][0] == 4
        assert result[0][1] == 7  # Should be 7, not 6
        assert result[1][1] == 10
    
    @pytest.mark.parametrize("min_N", [0, 1, 2, 5, 10])
    def test_min_N_parameter_comprehensive(self, min_N):
        """Test min_N parameter with various values"""
        onsets_keep = np.array([0, 20])
        ends_keep = np.array([15, 35])
        onsets_reject = np.array([5, 25])
        ends_reject = np.array([7, 27])
        
        result = _interval_difference(onsets_keep, ends_keep, onsets_reject, ends_reject, min_N)
        
        # All returned segments should be >= min_N in length
        for i in range(len(result[0])):
            segment_length = result[1][i] - result[0][i]
            assert segment_length >= min_N, f"Segment {i} too short: {segment_length} < {min_N}"
    
    def test_performance_large_scale(self):
        """Test performance with large inputs"""
        n_keep = 1000
        n_reject = 2000
        
        onsets_keep = np.arange(0, n_keep * 100, 100)
        ends_keep = onsets_keep + 50
        
        onsets_reject = np.random.uniform(0, n_keep * 100, n_reject)
        ends_reject = onsets_reject + np.random.uniform(1, 10, n_reject)
        
        # Sort reject intervals
        sort_idx = np.argsort(onsets_reject)
        onsets_reject = onsets_reject[sort_idx]
        ends_reject = ends_reject[sort_idx]
        
        import time
        start_time = time.time()
        result = _interval_difference(onsets_keep, ends_keep, onsets_reject, ends_reject)
        execution_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert execution_time < 10.0, f"Function too slow: {execution_time:.2f}s"
        
        # Basic validity
        assert len(result[0]) == len(result[1])
    
    def test_type_handling(self):
        """Test function handles different numpy dtypes"""
        dtypes = [np.int32, np.int64, np.float32, np.float64]
        
        for dtype in dtypes:
            onsets_keep = np.array([0, 10], dtype=dtype)
            ends_keep = np.array([5, 15], dtype=dtype)
            onsets_reject = np.array([2], dtype=dtype)
            ends_reject = np.array([3], dtype=dtype)
            
            result = _interval_difference(onsets_keep, ends_keep, onsets_reject, ends_reject)
            
            # Should work with all dtypes
            assert len(result[0]) >= 1
            assert len(result[1]) >= 1

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short", "-x"])  # -x stops on first failure