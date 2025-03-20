use std::cmp::{max, min};
use std::collections::BTreeSet;
use std::ops::Range;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioRange {
    /// Start position in samples (inclusive)
    pub start: u64,
    /// End position in samples (exclusive)
    pub end: u64,
    /// Priority for loading (higher values are loaded first)
    pub priority: u8,
}

impl AudioRange {
    /// Create a new audio range
    pub fn new(start: u64, end: u64, priority: u8) -> Self {
        assert!(start <= end, "Range start must be <= end");
        Self { start, end, priority }
    }

    /// Create a range representing the visible window
    pub fn visible_window(start_sample: u64, visible_samples: u64) -> Self {
        Self::new(start_sample, start_sample + visible_samples, 10)
    }

    /// Create a range for pre-buffering playback
    pub fn playback_buffer(current_pos: u64, buffer_samples: u64) -> Self {
        Self::new(current_pos, current_pos + buffer_samples, 5)
    }
    
    /// Create a range based on a time window
    pub fn from_time_window(start_time: f64, duration: f64, sample_rate: u32, channels: u16) -> Self {
        let samples_per_second = sample_rate as u64 * channels as u64;
        let start_sample = (start_time * samples_per_second as f64) as u64;
        let duration_samples = (duration * samples_per_second as f64) as u64;
        Self::new(start_sample, start_sample + duration_samples, 0)
    }

    /// Length of this range in samples
    pub fn len(&self) -> u64 {
        self.end - self.start
    }

    /// Check if this range is empty
    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }

    /// Check if this range contains a specific sample position
    pub fn contains(&self, position: u64) -> bool {
        position >= self.start && position < self.end
    }

    /// Check if this range overlaps with another range
    pub fn overlaps(&self, other: &AudioRange) -> bool {
        self.start < other.end && self.end > other.start
    }

    /// Get the intersection of this range with another
    pub fn intersection(&self, other: &AudioRange) -> Option<AudioRange> {
        if !self.overlaps(other) {
            return None;
        }
        
        let start = max(self.start, other.start);
        let end = min(self.end, other.end);
        let priority = max(self.priority, other.priority);
        
        Some(AudioRange::new(start, end, priority))
    }

    /// Get a new range that covers both this range and another
    pub fn union(&self, other: &AudioRange) -> AudioRange {
        let start = min(self.start, other.start);
        let end = max(self.end, other.end);
        let priority = max(self.priority, other.priority);
        
        AudioRange::new(start, end, priority)
    }

    /// Check if this range is a subrange of another
    pub fn is_subrange_of(&self, other: &AudioRange) -> bool {
        self.start >= other.start && self.end <= other.end
    }

    /// Align this range to a given chunk size
    pub fn align_to_chunk_size(&self, chunk_size: u64) -> AudioRange {
        let aligned_start = (self.start / chunk_size) * chunk_size;
        let aligned_end = ((self.end + chunk_size - 1) / chunk_size) * chunk_size;
        
        AudioRange::new(aligned_start, aligned_end, self.priority)
    }

    /// Convert to a standard Rust range
    pub fn as_range(&self) -> Range<u64> {
        self.start..self.end
    }
}

impl From<Range<u64>> for AudioRange {
    fn from(range: Range<u64>) -> Self {
        AudioRange::new(range.start, range.end, 0)
    }
}

impl fmt::Display for AudioRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}-{}) (priority: {})", self.start, self.end, self.priority)
    }
}

/// A collection of non-overlapping audio ranges.
/// Used to track multiple disjoint ranges that need to be loaded.
#[derive(Debug, Clone, Default)]
pub struct AudioRangeSet {
    // Use BTreeSet for ordered iteration (which is helpful for loading in order)
    ranges: BTreeSet<(u64, u64, u8)>,
}

impl AudioRangeSet {
    /// Create a new, empty range set
    pub fn new() -> Self {
        Self {
            ranges: BTreeSet::new(),
        }
    }
    
    /// Add a range to the set, merging with overlapping ranges
    pub fn add(&mut self, range: &AudioRange) {
        if range.is_empty() {
            return;
        }
        
        // Convert to simpler representation for storage
        let entry = (range.start, range.end, range.priority);
        
        // Find overlapping ranges
        let mut overlaps: Vec<(u64, u64, u8)> = self.ranges
            .iter()
            .filter(|&&(start, end, _)| start < range.end && end > range.start)
            .copied()
            .collect();
        
        // If no overlaps, just add the new range
        if overlaps.is_empty() {
            self.ranges.insert(entry);
            return;
        }
        
        // Remove overlapping ranges
        for overlap in &overlaps {
            self.ranges.remove(overlap);
        }
        
        // Add this range to the overlaps for merging
        overlaps.push(entry);
        
        // Merge overlapping ranges
        let mut merged_start = u64::MAX;
        let mut merged_end = 0;
        let mut max_priority = 0;
        
        for (start, end, priority) in overlaps {
            merged_start = merged_start.min(start);
            merged_end = merged_end.max(end);
            max_priority = max_priority.max(priority);
        }
        
        // Insert the merged range
        self.ranges.insert((merged_start, merged_end, max_priority));
    }
    
    /// Remove a range from the set, splitting existing ranges as needed
    pub fn remove(&mut self, range: &AudioRange) {
        if range.is_empty() {
            return;
        }
        
        // Find affected ranges
        let affected: Vec<(u64, u64, u8)> = self.ranges
            .iter()
            .filter(|&&(start, end, _)| start < range.end && end > range.start)
            .copied()
            .collect();
        
        for (start, end, priority) in affected {
            // Remove the affected range
            self.ranges.remove(&(start, end, priority));
            
            // Create new ranges for the parts that don't overlap
            if start < range.start {
                self.ranges.insert((start, range.start, priority));
            }
            
            if end > range.end {
                self.ranges.insert((range.end, end, priority));
            }
        }
    }
    
    /// Check if this set completely contains the given range
    pub fn contains(&self, range: &AudioRange) -> bool {
        if range.is_empty() {
            return true;
        }
        
        // Find all ranges that might contain parts of the query range
        let containing_ranges: Vec<(u64, u64, u8)> = self.ranges
            .iter()
            .filter(|&&(start, end, _)| start <= range.start && end >= range.end)
            .copied()
            .collect();
        
        !containing_ranges.is_empty()
    }
    
    /// Get ranges that are not yet loaded in the given range
    pub fn missing_ranges(&self, range: &AudioRange) -> Vec<AudioRange> {
        if range.is_empty() {
            return Vec::new();
        }
        
        let mut result = Vec::new();
        let mut current = range.start;
        
        // Handle the case of an empty set
        if self.ranges.is_empty() {
            result.push(range.clone());
            return result;
        }
        
        // Find all relevant ranges (that might overlap with our query)
        let relevant: Vec<(u64, u64, u8)> = self.ranges
            .iter()
            .filter(|&&(start, end, _)| start < range.end && end > range.start)
            .copied()
            .collect();
        
        // If no relevant ranges, the entire query range is missing
        if relevant.is_empty() {
            result.push(range.clone());
            return result;
        }
        
        // Sort by start position for easier processing
        let mut sorted = relevant;
        sorted.sort_by_key(|&(start, _, _)| start);
        
        // Process each range to find gaps
        for (start, end, _) in sorted {
            // If there's a gap before this range, add it to missing
            if current < start {
                result.push(AudioRange::new(current, start, range.priority));
            }
            
            // Move current position to end of this range
            current = current.max(end);
            
            // If we've covered the whole query range, we're done
            if current >= range.end {
                break;
            }
        }
        
        // If we haven't reached the end of the query range, add the final missing part
        if current < range.end {
            result.push(AudioRange::new(current, range.end, range.priority));
        }
        
        result
    }
    
    /// Get all ranges in this set
    pub fn get_ranges(&self) -> Vec<AudioRange> {
        self.ranges
            .iter()
            .map(|&(start, end, priority)| AudioRange::new(start, end, priority))
            .collect()
    }
    
    /// Get all ranges in this set, ordered by priority (highest first)
    pub fn get_ranges_by_priority(&self) -> Vec<AudioRange> {
        let mut ranges = self.get_ranges();
        ranges.sort_by(|a, b| b.priority.cmp(&a.priority));
        ranges
    }
    
    /// Clear all ranges from this set
    pub fn clear(&mut self) {
        self.ranges.clear();
    }
    
    /// Check if this set is empty
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }
    
    /// Get the total number of samples covered by this set
    pub fn total_samples(&self) -> u64 {
        self.ranges
            .iter()
            .map(|&(start, end, _)| end - start)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_range_basic() {
        let range = AudioRange::new(100, 200, 5);
        assert_eq!(range.len(), 100);
        assert!(!range.is_empty());
        assert!(range.contains(150));
        assert!(!range.contains(200));
    }

    #[test]
    fn test_audio_range_overlaps() {
        let range1 = AudioRange::new(100, 200, 5);
        let range2 = AudioRange::new(150, 250, 3);
        let range3 = AudioRange::new(300, 400, 1);
        
        assert!(range1.overlaps(&range2));
        assert!(range2.overlaps(&range1));
        assert!(!range1.overlaps(&range3));
    }

    #[test]
    fn test_audio_range_intersection() {
        let range1 = AudioRange::new(100, 200, 5);
        let range2 = AudioRange::new(150, 250, 3);
        
        let intersection = range1.intersection(&range2).unwrap();
        assert_eq!(intersection.start, 150);
        assert_eq!(intersection.end, 200);
        assert_eq!(intersection.priority, 5); // Higher priority
    }

    #[test]
    fn test_audio_range_union() {
        let range1 = AudioRange::new(100, 200, 5);
        let range2 = AudioRange::new(150, 250, 3);
        
        let union = range1.union(&range2);
        assert_eq!(union.start, 100);
        assert_eq!(union.end, 250);
        assert_eq!(union.priority, 5); // Higher priority
    }

    #[test]
    fn test_range_set_add() {
        let mut set = AudioRangeSet::new();
        
        // Add first range
        set.add(&AudioRange::new(100, 200, 5));
        let ranges = set.get_ranges();
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].start, 100);
        assert_eq!(ranges[0].end, 200);
        
        // Add non-overlapping range
        set.add(&AudioRange::new(300, 400, 3));
        let ranges = set.get_ranges();
        assert_eq!(ranges.len(), 2);
        
        // Add overlapping range that should merge
        set.add(&AudioRange::new(150, 350, 4));
        let ranges = set.get_ranges();
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].start, 100);
        assert_eq!(ranges[0].end, 400);
        assert_eq!(ranges[0].priority, 5); // Highest priority is kept
    }

    #[test]
    fn test_range_set_remove() {
        let mut set = AudioRangeSet::new();
        set.add(&AudioRange::new(100, 500, 5));
        
        // Remove from middle
        set.remove(&AudioRange::new(200, 300, 0));
        let ranges = set.get_ranges();
        assert_eq!(ranges.len(), 2);
        
        // Should have [100, 200) and [300, 500)
        let mut sorted = ranges;
        sorted.sort_by_key(|r| r.start);
        assert_eq!(sorted[0].start, 100);
        assert_eq!(sorted[0].end, 200);
        assert_eq!(sorted[1].start, 300);
        assert_eq!(sorted[1].end, 500);
    }

    #[test]
    fn test_range_set_missing_ranges() {
        let mut set = AudioRangeSet::new();
        set.add(&AudioRange::new(100, 200, 5));
        set.add(&AudioRange::new(300, 400, 3));
        
        // Check missing in middle
        let missing = set.missing_ranges(&AudioRange::new(0, 500, 1));
        assert_eq!(missing.len(), 3);
        
        // Should have [0, 100), [200, 300), and [400, 500)
        let mut sorted = missing;
        sorted.sort_by_key(|r| r.start);
        assert_eq!(sorted[0].start, 0);
        assert_eq!(sorted[0].end, 100);
        assert_eq!(sorted[1].start, 200);
        assert_eq!(sorted[1].end, 300);
        assert_eq!(sorted[2].start, 400);
        assert_eq!(sorted[2].end, 500);
    }
}