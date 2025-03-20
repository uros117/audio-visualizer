use std::collections::BTreeMap;
use std::ops::Bound::{Included, Excluded};
use std::sync::{Arc, RwLock};
use std::cmp::min;

/// Represents a chunk of audio data with reference counting for efficient sharing
#[derive(Clone, Debug)]
pub struct AudioChunk {
    /// Starting position in samples
    pub position: u64,
    /// Sample data (interleaved for multi-channel audio)
    pub samples: Arc<Vec<f32>>,
    /// Number of channels
    pub channels: u16,
    /// Timestamp when this chunk was last accessed
    pub last_accessed: std::time::Instant,
}

impl AudioChunk {
    /// Create a new audio chunk
    pub fn new(position: u64, samples: Vec<f32>, channels: u16) -> Self {
        Self {
            position,
            samples: Arc::new(samples),
            channels,
            last_accessed: std::time::Instant::now(),
        }
    }

    /// Get the length of this chunk in samples
    pub fn len(&self) -> u64 {
        self.samples.len() as u64
    }

    /// Update the last accessed timestamp
    pub fn touch(&mut self) {
        self.last_accessed = std::time::Instant::now();
    }
}

/// A storage system for audio data using a B-tree for efficient retrieval
pub struct AudioStorage {
    /// B-tree mapping from start position to audio chunks
    chunks: BTreeMap<u64, AudioChunk>,
    /// Maximum number of chunks to keep in memory
    max_chunks: usize,
    /// Total number of samples in memory
    samples_in_memory: u64,
    /// Maximum number of samples to keep in memory
    max_samples: u64,
    /// Sample rate of the audio
    sample_rate: u32,
    /// Number of channels in the audio
    channels: u16,
}

impl AudioStorage {
    /// Create a new audio storage with the given parameters
    pub fn new(sample_rate: u32, channels: u16, max_samples: u64) -> Self {
        Self {
            chunks: BTreeMap::new(),
            max_chunks: 1024, // Default, can be adjusted
            samples_in_memory: 0,
            max_samples,
            sample_rate,
            channels,
        }
    }
    
    /// Add a chunk of audio data to the storage
    pub fn add_chunk(&mut self, position: u64, samples: Vec<f32>) -> bool {
        // Make sure we're not adding a duplicate
        if self.chunks.contains_key(&position) {
            return false; // Already exists
        }
        
        let chunk_size = samples.len() as u64;
        
        // Create the new chunk
        let chunk = AudioChunk::new(position, samples, self.channels);
        
        // Check if we need to free up space
        if self.samples_in_memory + chunk_size > self.max_samples || 
           self.chunks.len() >= self.max_chunks {
            self.evict_chunks(chunk_size);
        }
        
        // Add the chunk
        self.chunks.insert(position, chunk);
        self.samples_in_memory += chunk_size;
        
        true
    }
    
    /// Evict chunks until we have enough space for the given size
    fn evict_chunks(&mut self, needed_size: u64) {
        // Create a vector of (position, last_accessed) pairs
        let mut chunk_ages: Vec<(u64, std::time::Instant)> = self.chunks
            .iter()
            .map(|(&pos, chunk)| (pos, chunk.last_accessed))
            .collect();
        
        // Sort by last accessed time (oldest first)
        chunk_ages.sort_by(|(_, a), (_, b)| a.cmp(b));
        
        // Evict chunks until we have enough space
        let mut freed = 0;
        for (pos, _) in chunk_ages {
            if self.samples_in_memory + needed_size - freed <= self.max_samples &&
               self.chunks.len() <= self.max_chunks {
                break;
            }
            
            if let Some(chunk) = self.chunks.remove(&pos) {
                freed += chunk.len();
            }
        }
        
        self.samples_in_memory -= freed;
    }
    
    /// Get audio samples for the given range
    pub fn get_samples(&mut self, start: u64, count: u64) -> Vec<f32> {
        let mut result = Vec::with_capacity(count as usize);
        let end = start + count;
        
        // Find all chunks that intersect with our range
        let range = self.chunks.range((Included(start), Excluded(end)));
        
        // Track how many samples we've collected
        let mut collected = 0;
        
        // Iterate over relevant chunks
        for (_, chunk) in range {
            // Touch the chunk to update its last accessed time
            let mut chunk = chunk.clone();
            chunk.touch();
            
            // Calculate offsets
            let chunk_start = chunk.position;
            let chunk_end = chunk_start + chunk.len();
            
            let overlap_start = start.max(chunk_start);
            let overlap_end = end.min(chunk_end);
            
            if overlap_start < overlap_end {
                // Calculate the offset into the chunk
                let chunk_offset = (overlap_start - chunk_start) as usize;
                // Calculate how many samples to take
                let take_count = (overlap_end - overlap_start) as usize;
                
                // Add samples to the result
                result.extend_from_slice(&chunk.samples[chunk_offset..chunk_offset + take_count]);
                
                // Update collected count
                collected += take_count as u64;
            }
        }
        
        // If we didn't collect enough samples, pad with zeros
        if collected < count {
            result.resize(count as usize, 0.0);
        }
        
        result
    }
    
    /// Check what ranges are missing in the given range
    pub fn missing_ranges(&self, start: u64, end: u64) -> Vec<(u64, u64)> {
        let mut missing = Vec::new();
        let mut current = start;
        
        // Find all chunks that intersect with our range
        let range = self.chunks.range((Included(start), Excluded(end)));
        
        // Convert iterator to vector for easier processing
        let chunks: Vec<(&u64, &AudioChunk)> = range.collect();
        
        // If no chunks, the entire range is missing
        if chunks.is_empty() {
            missing.push((start, end));
            return missing;
        }
        
        // Process each chunk to find gaps
        for (_, chunk) in chunks {
            let chunk_start = chunk.position;
            let chunk_end = chunk_start + chunk.len();
            
            // If there's a gap before this chunk, add it to missing
            if current < chunk_start {
                missing.push((current, chunk_start));
            }
            
            // Move current position to end of this chunk
            current = current.max(chunk_end);
            
            // If we've covered the whole range, we're done
            if current >= end {
                break;
            }
        }
        
        // If we haven't reached the end, add the final missing part
        if current < end {
            missing.push((current, end));
        }
        
        missing
    }
    
    /// Get the total number of samples in memory
    pub fn samples_in_memory(&self) -> u64 {
        self.samples_in_memory
    }
    
    /// Get the number of chunks in memory
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }
    
    /// Clear all chunks from memory
    pub fn clear(&mut self) {
        self.chunks.clear();
        self.samples_in_memory = 0;
    }
    
    /// Check if a specific position is loaded
    pub fn is_loaded(&self, position: u64) -> bool {
        // Find the largest key less than or equal to position
        if let Some((start, chunk)) = self.chunks.range(..=position).next_back() {
            // Check if position is within this chunk
            let end = start + chunk.len();
            position < end
        } else {
            false
        }
    }
    
    /// Get the proportion of a range that is loaded (0.0 to 1.0)
    pub fn loaded_proportion(&self, start: u64, end: u64) -> f32 {
        if start >= end {
            return 1.0;
        }
        
        let total_length = end - start;
        let missing_ranges = self.missing_ranges(start, end);
        
        let missing_samples: u64 = missing_ranges.iter()
            .map(|(s, e)| e - s)
            .sum();
        
        let loaded_samples = total_length - missing_samples;
        loaded_samples as f32 / total_length as f32
    }
}

/// A thread-safe wrapper around AudioStorage for concurrent access
pub struct ThreadSafeAudioStorage {
    storage: Arc<RwLock<AudioStorage>>,
}

impl ThreadSafeAudioStorage {
    /// Create a new thread-safe audio storage
    pub fn new(sample_rate: u32, channels: u16, max_samples: u64) -> Self {
        Self {
            storage: Arc::new(RwLock::new(AudioStorage::new(
                sample_rate, channels, max_samples
            ))),
        }
    }
    
    /// Add a chunk of audio data
    pub fn add_chunk(&self, position: u64, samples: Vec<f32>) -> Result<bool, String> {
        match self.storage.write() {
            Ok(mut storage) => Ok(storage.add_chunk(position, samples)),
            Err(_) => Err("Failed to acquire write lock on audio storage".to_string()),
        }
    }
    
    /// Get audio samples for the given range
    pub fn get_samples(&self, start: u64, count: u64) -> Result<Vec<f32>, String> {
        match self.storage.write() {
            Ok(mut storage) => Ok(storage.get_samples(start, count)),
            Err(_) => Err("Failed to acquire write lock on audio storage".to_string()),
        }
    }
    
    /// Check what ranges are missing in the given range
    pub fn missing_ranges(&self, start: u64, end: u64) -> Result<Vec<(u64, u64)>, String> {
        match self.storage.read() {
            Ok(storage) => Ok(storage.missing_ranges(start, end)),
            Err(_) => Err("Failed to acquire read lock on audio storage".to_string()),
        }
    }
    
    /// Clear all chunks from memory
    pub fn clear(&self) -> Result<(), String> {
        match self.storage.write() {
            Ok(mut storage) => {
                storage.clear();
                Ok(())
            },
            Err(_) => Err("Failed to acquire write lock on audio storage".to_string()),
        }
    }
    
    /// Get statistics about the storage
    pub fn stats(&self) -> Result<(u64, usize), String> {
        match self.storage.read() {
            Ok(storage) => Ok((storage.samples_in_memory(), storage.chunk_count())),
            Err(_) => Err("Failed to acquire read lock on audio storage".to_string()),
        }
    }
    
    /// Get the proportion of a range that is loaded (0.0 to 1.0)
    pub fn loaded_proportion(&self, start: u64, end: u64) -> Result<f32, String> {
        match self.storage.read() {
            Ok(storage) => Ok(storage.loaded_proportion(start, end)),
            Err(_) => Err("Failed to acquire read lock on audio storage".to_string()),
        }
    }
    
    /// Create a cloneable handle to this storage
    pub fn clone_handle(&self) -> Self {
        Self {
            storage: Arc::clone(&self.storage),
        }
    }
}

/// Integration with AudioLoader to load chunks into storage
pub struct AudioBufferManager {
    /// The audio storage
    storage: ThreadSafeAudioStorage,
    /// The audio loader (you'll need to adapt this to your loader)
    loader: Arc<RwLock<crate::audio_loader::AudioLoader>>,
    /// Currently pending load requests
    pending_requests: std::collections::HashSet<u64>,
    /// Chunk size for loading
    chunk_size: u64,
}

impl AudioBufferManager {
    /// Create a new audio buffer manager
    pub fn new(
        sample_rate: u32,
        channels: u16,
        max_samples: u64,
        loader: Arc<RwLock<crate::audio_loader::AudioLoader>>,
        chunk_size: u64,
    ) -> Self {
        Self {
            storage: ThreadSafeAudioStorage::new(sample_rate, channels, max_samples),
            loader,
            pending_requests: std::collections::HashSet::new(),
            chunk_size,
        }
    }
    
    /// Update the buffer manager (call this regularly)
    pub fn update(&mut self) -> Result<(), String> {
        // Check for completed chunk loads
        let responses = match self.loader.write() {
            Ok(mut loader) => loader.poll_responses(),
            Err(_) => return Err("Failed to acquire write lock on loader".to_string()),
        };
        
        for response in responses {
            match response {
                crate::audio_loader::LoaderResponse::ChunkLoaded(position, samples) => {
                    // Add to storage
                    let _ = self.storage.add_chunk(position, samples);
                    
                    // Remove from pending requests
                    self.pending_requests.remove(&position);
                },
                _ => {} // Ignore other response types
            }
        }
        
        Ok(())
    }
    
    /// Request loading of ranges that are missing
    pub fn ensure_loaded(&mut self, ranges: &[(u64, u64)]) -> Result<(), String> {
        for &(start, end) in ranges {
            // Find missing sub-ranges
            let missing = self.storage.missing_ranges(start, end)?;
            
            // Request loading for each missing range
            for (missing_start, missing_end) in missing {
                // Align to chunk boundaries
                let aligned_start = (missing_start / self.chunk_size) * self.chunk_size;
                let aligned_end = ((missing_end + self.chunk_size - 1) / self.chunk_size) * self.chunk_size;
                
                // Request each chunk
                for pos in (aligned_start..aligned_end).step_by(self.chunk_size as usize) {
                    // Skip if already pending
                    if self.pending_requests.contains(&pos) {
                        continue;
                    }
                    
                    // Request the chunk
                    match self.loader.write() {
                        Ok(mut loader) => {
                            loader.request_chunk(pos);
                            self.pending_requests.insert(pos);
                        },
                        Err(_) => return Err("Failed to acquire write lock on loader".to_string()),
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Get samples from storage, loading if necessary
    pub fn get_samples(&mut self, start: u64, count: u64) -> Result<Vec<f32>, String> {
        // Check if the range is fully loaded
        let missing = self.storage.missing_ranges(start, start + count)?;
        
        // If there are missing ranges, request them
        if !missing.is_empty() {
            self.ensure_loaded(&missing)?;
        }
        
        // Get the samples we have available
        self.storage.get_samples(start, count)
    }
    
    /// Get a cloneable handle to the storage
    pub fn storage_handle(&self) -> ThreadSafeAudioStorage {
        self.storage.clone_handle()
    }
    
    /// Get the number of pending requests
    pub fn pending_count(&self) -> usize {
        self.pending_requests.len()
    }

    /// Clear the storage
    pub fn clear(&mut self) -> Result<(), String> {
        self.storage.clear()
    }
}