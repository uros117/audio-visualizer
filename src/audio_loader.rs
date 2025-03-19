use std::fs::File;
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::collections::VecDeque;
use std::io::SeekFrom;
use std::io::prelude::*;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, Decoder};
use symphonia::core::formats::{FormatOptions, FormatReader, SeekMode, SeekTo};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::core::errors::Error as SymphoniaError;

// Keep existing command and response enums
#[derive(Debug, Clone)]
pub enum LoaderCommand {
    LoadChunk(u64), // Position to load (in samples)
    Stop,
}

#[derive(Debug, Clone)]
pub enum LoaderResponse {
    ChunkLoaded(u64, Vec<f32>), // Position and data
    LoadingError(String),
    Progress(f32), // Progress percentage
    FileInfo {
        sample_rate: u32,
        num_channels: u16,
        total_samples: u64,
    },
}

// New structure to represent a decoded audio frame
struct CachedFrame {
    // Starting sample position of this frame
    position: u64,
    // Decoded audio samples
    samples: Vec<f32>,
    // Number of frames (not samples) in this packet
    frame_count: usize,
}

// New caching decoder that maintains decoder state and frame cache
struct CachingDecoder {
    // File path for reopening if needed
    path: PathBuf,
    // Symphonia format reader
    format: Box<dyn FormatReader>,
    // Symphonia decoder
    decoder: Box<dyn Decoder>,
    // Track ID we're decoding
    track_id: u32,
    // Audio specification
    sample_rate: u32,
    num_channels: u16,
    // Current position in the stream (in frames)
    current_frame_pos: u64,
    // Frame cache - store recently decoded frames
    frame_cache: VecDeque<CachedFrame>,
    // Maximum number of frames to cache
    max_cached_frames: usize,
}

impl CachingDecoder {
    pub fn new(path: PathBuf) -> Result<(Self, u64), String> {
        // Open the file
        let file = match File::open(&path) {
            Ok(f) => f,
            Err(e) => return Err(format!("Error opening file: {:?}", e)),
        };

        // Configure format reader
        let mut hint = Hint::new();
        if let Some(ext) = path.extension() {
            if let Some(ext_str) = ext.to_str() {
                hint.with_extension(ext_str);
            }
        }

        let format_opts = FormatOptions {
            seek_index_fill_rate: 100,
            ..Default::default()
        };
        let metadata_opts = MetadataOptions::default();
        let decoder_opts = DecoderOptions::default();

        // Create media source stream
        let source = Box::new(file);
        let mss = MediaSourceStream::new(source, Default::default());

        // Probe for format
        let probed = match symphonia::default::get_probe()
            .format(&hint, mss, &format_opts, &metadata_opts) {
            Ok(p) => p,
            Err(e) => return Err(format!("Error probing format: {:?}", e)),
        };

        let mut format = probed.format;

        // Get the default track
        let track = match format.default_track() {
            Some(t) => t,
            None => return Err("No default track found".to_string()),
        };

        let track_id = track.id;
        let codec_params = track.codec_params.clone();

        // Get audio specs
        let sample_rate = codec_params.sample_rate.unwrap_or(44100);
        let num_channels = codec_params.channels.unwrap_or(symphonia::core::audio::Channels::empty()).count() as u16;

        // Calculate total samples
        let total_samples = if let Some(n_frames) = codec_params.n_frames {
            n_frames * num_channels as u64
        } else if let Some(duration) = codec_params.time_base.and_then(|tb|
            codec_params.n_frames.map(|frames| tb.calc_time(frames))
        ) {
            ((duration.seconds as f64 + duration.frac) * sample_rate as f64 * num_channels as f64) as u64
        } else {
            // Estimate based on file size if duration can't be determined
            match File::open(&path) {
                Ok(f) => match f.metadata() {
                    Ok(metadata) => {
                        let file_size = metadata.len();
                        // Rough estimate: bytes per sample ~= 2, plus overhead
                        file_size / 3 * num_channels as u64
                    },
                    Err(_) => 0,
                },
                Err(_) => 0,
            }
        };

        // Create decoder
        let decoder = match symphonia::default::get_codecs()
            .make(&codec_params, &decoder_opts) {
            Ok(d) => d,
            Err(e) => return Err(format!("Error creating decoder: {:?}", e)),
        };

        Ok((Self {
            path,
            format,
            decoder,
            track_id,
            sample_rate,
            num_channels,
            current_frame_pos: 0,
            frame_cache: VecDeque::with_capacity(100),
            max_cached_frames: 100,
        }, total_samples))
    }

    // Seek to a specific frame position
    pub fn seek_to_frame(&mut self, target_frame: u64) -> Result<u64, String> {
        // Check if the target frame is in our cache
        if let Some(cached_idx) = self.find_cached_frame(target_frame) {
            let frame = &self.frame_cache[cached_idx];
            self.current_frame_pos = frame.position / self.num_channels as u64;
            return Ok(self.current_frame_pos);
        }

        // Calculate timestamp for seeking
        let codec_params = match self.format.default_track() {
            Some(track) => track.codec_params.clone(),
            None => return Err("Track not found".to_string()),
        };

        let time_base = match codec_params.time_base {
            Some(tb) => tb,
            None => return Err("Time base not available".to_string()),
        };

        let timestamp = time_base.calc_time(target_frame);

        // Attempt to seek
        match self.format.seek(
            SeekMode::Accurate,
            SeekTo::Time {
                track_id: Some(self.track_id),
                time: timestamp
            },
        ) {
            Ok(seeked_to) => {
                // Reset decoder after seeking
                self.decoder.reset();

                // Clear cache after seeking
                self.frame_cache.clear();

                // Update current position
                self.current_frame_pos = seeked_to.actual_ts;

                Ok(self.current_frame_pos)
            },
            Err(e) => Err(format!("Error seeking: {:?}", e)),
        }
    }

    // Find a cached frame that contains the target frame
    fn find_cached_frame(&self, target_frame: u64) -> Option<usize> {
        let target_sample = target_frame * self.num_channels as u64;

        for (idx, frame) in self.frame_cache.iter().enumerate() {
            let frame_start_sample = frame.position;
            let frame_end_sample = frame_start_sample + (frame.frame_count * self.num_channels as usize) as u64;

            if target_sample >= frame_start_sample && target_sample < frame_end_sample {
                return Some(idx);
            }
        }

        None
    }

    // Decode next packet and add to cache
    fn decode_next_packet(&mut self) -> Result<&CachedFrame, String> {
        // Get next packet
        let packet = match self.format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(_)) => {
                return Err("End of file".to_string());
            },
            Err(e) => {
                return Err(format!("Error reading packet: {:?}", e));
            }
        };

        // Skip packets from other tracks
        if packet.track_id() != self.track_id {
            return self.decode_next_packet();
        }

        // Decode the packet
        match self.decoder.decode(&packet) {
            Ok(decoded) => {
                // Get decoded audio information
                let frame_count = decoded.frames();
                let spec = *decoded.spec();

                // Convert to sample buffer
                let mut sample_buf = SampleBuffer::<f32>::new(
                    decoded.capacity() as u64,
                    spec,
                );

                sample_buf.copy_interleaved_ref(decoded);
                let samples = sample_buf.samples().to_vec();

                // Calculate position of this frame in samples
                let position = self.current_frame_pos * self.num_channels as u64;

                // Create cached frame
                let cached_frame = CachedFrame {
                    position,
                    samples,
                    frame_count,
                };

                // Add to cache
                self.frame_cache.push_back(cached_frame);

                // Limit cache size
                while self.frame_cache.len() > self.max_cached_frames {
                    self.frame_cache.pop_front();
                }

                // Update current position
                self.current_frame_pos += frame_count as u64;

                // Return reference to the cached frame
                self.frame_cache.back().ok_or("Cache error".to_string())
            },
            Err(e) => {
                // If we hit a decoding error, try to skip and continue
                println!("Decoding error: {:?}, skipping packet", e);
                self.decode_next_packet()
            }
        }
    }

    // Get samples from a specific position, decoding frames as needed
    pub fn get_samples(&mut self, start_sample: u64, required_samples: usize) -> Result<Vec<f32>, String> {
        let mut result = Vec::with_capacity(required_samples);
        let start_frame = start_sample / self.num_channels as u64;

        // If we're not at the right position, seek
        if self.current_frame_pos != start_frame {
            // Check if we have the frame in cache first
            if self.find_cached_frame(start_frame).is_none() {
                // If not in cache, seek to target position
                self.seek_to_frame(start_frame)?;
            }
        }

        // Calculate how many samples we still need to collect
        let mut remaining_samples = required_samples;
        let mut current_sample = start_sample;

        // First, gather samples from cache if available
        while remaining_samples > 0 {
            // Try to find a cached frame that contains our current sample
            if let Some(cache_idx) = self.find_cached_frame(current_sample / self.num_channels as u64) {
                let frame = &self.frame_cache[cache_idx];
                let frame_start_sample = frame.position;

                // Calculate offset into the cached frame
                let offset = (current_sample - frame_start_sample) as usize;

                // Calculate how many samples we can take from this frame
                let available = frame.samples.len() - offset;
                let to_take = std::cmp::min(available, remaining_samples);

                // Copy samples to result
                result.extend_from_slice(&frame.samples[offset..offset + to_take]);

                // Update counters
                remaining_samples -= to_take;
                current_sample += to_take as u64;

                // If we took all available samples from this frame, we need to decode more
                if offset + to_take >= frame.samples.len() {
                    break;
                }
            } else {
                // Not in cache, need to decode more
                break;
            }
        }

        // If we still need more samples, decode more frames
        while remaining_samples > 0 {
            // Decode next frame
            match self.decode_next_packet() {
                Ok(frame) => {
                    // Calculate how many samples we want from this frame
                    let to_take = std::cmp::min(frame.samples.len(), remaining_samples);

                    // Add samples to result
                    result.extend_from_slice(&frame.samples[0..to_take]);

                    // Update counters
                    remaining_samples -= to_take;
                    current_sample += to_take as u64;
                },
                Err(e) => {
                    // If we hit the end of file, just return what we have
                    if e == "End of file" {
                        break;
                    }
                    return Err(e);
                }
            }
        }

        Ok(result)
    }
}

pub struct AudioLoader {
    thread: Option<thread::JoinHandle<()>>,
    command_sender: Option<mpsc::Sender<LoaderCommand>>,
    response_receiver: Option<mpsc::Receiver<LoaderResponse>>,
    is_running: bool,
    file_path: Option<PathBuf>,
    sample_rate: u32,
    num_channels: u16,
    total_samples: u64,
    chunk_size: u64,
}

impl AudioLoader {
    pub fn new() -> Self {
        Self {
            thread: None,
            command_sender: None,
            response_receiver: None,
            is_running: false,
            file_path: None,
            sample_rate: 44100,
            num_channels: 2,
            total_samples: 0,
            chunk_size: 1024,
        }
    }

    pub fn start(&mut self, file_path: PathBuf, chunk_size: u64) -> Result<(), String> {
        if self.is_running {
            return Err("Loader is already running".to_string());
        }

        // Create channels for communication
        let (cmd_tx, cmd_rx) = mpsc::channel::<LoaderCommand>();
        let (resp_tx, resp_rx) = mpsc::channel::<LoaderResponse>();

        self.file_path = Some(file_path.clone());
        self.chunk_size = chunk_size;

        // Create caching decoder to get file info
        let decoder_result = CachingDecoder::new(file_path.clone());

        let (decoder_info, total_samples) = match decoder_result {
            Ok((decoder, total_samples)) => {
                self.sample_rate = decoder.sample_rate;
                self.num_channels = decoder.num_channels;
                self.total_samples = total_samples;
                (
                    (decoder.sample_rate, decoder.num_channels),
                    total_samples
                )
            },
            Err(e) => return Err(e),
        };

        // Create and start the thread with the sample_rate and num_channels
        let (sample_rate, num_channels) = decoder_info;
        let file_path_clone = file_path.clone();
        let thread = thread::spawn(move || {
            Self::loader_thread(file_path_clone, chunk_size, sample_rate, num_channels, total_samples, cmd_rx, resp_tx);
        });

        self.thread = Some(thread);
        self.command_sender = Some(cmd_tx);
        self.response_receiver = Some(resp_rx);
        self.is_running = true;

        // Send file info immediately
        if let Some(sender) = &self.response_receiver {
            // Try to receive the file info that should be sent by the loader thread
            match sender.recv_timeout(std::time::Duration::from_millis(100)) {
                Ok(_) => {}, // File info received
                Err(_) => {} // Timeout or error, could handle differently
            }
        }

        Ok(())
    }

    // The improved loader thread with frame caching
    fn loader_thread(
        file_path: PathBuf,
        chunk_size: u64,
        sample_rate: u32,
        num_channels: u16,
        total_samples: u64,
        cmd_rx: mpsc::Receiver<LoaderCommand>,
        resp_tx: mpsc::Sender<LoaderResponse>
    ) {
        println!("Loader thread started");

        // Send initial file info
        let _ = resp_tx.send(LoaderResponse::FileInfo {
            sample_rate,
            num_channels,
            total_samples,
        });

        // Create our caching decoder
        let mut decoder = match CachingDecoder::new(file_path) {
            Ok((decoder, _)) => decoder,
            Err(e) => {
                println!("Error creating decoder: {}", e);
                let _ = resp_tx.send(LoaderResponse::LoadingError(e));
                return;
            }
        };

        // Process commands
        loop {
            match cmd_rx.recv() {
                Ok(cmd) => match cmd {
                    LoaderCommand::LoadChunk(position) => {
                        println!("Loading chunk at position: {}", position);

                        // Use the caching decoder to get samples
                        match decoder.get_samples(position, chunk_size as usize) {
                            Ok(samples) => {
                                // Send back the loaded chunk
                                let _ = resp_tx.send(LoaderResponse::ChunkLoaded(position, samples));
                            },
                            Err(e) => {
                                let _ = resp_tx.send(LoaderResponse::LoadingError(e));
                            }
                        }
                    },
                    LoaderCommand::Stop => {
                        println!("Loader thread stopping");
                        break;
                    }
                },
                Err(e) => {
                    println!("Error receiving command: {:?}", e);
                    break;
                }
            }
        }

        println!("Loader thread terminated");
    }

    // Keep existing methods
    pub fn stop(&mut self) {
        if let Some(sender) = &self.command_sender {
            let _ = sender.send(LoaderCommand::Stop);

            // Wait for thread to finish
            if let Some(thread) = self.thread.take() {
                let _ = thread.join();
            }

            self.command_sender = None;
            self.response_receiver = None;
            self.is_running = false;
        }
    }

    pub fn request_chunk(&mut self, position: u64) {
        if let Some(sender) = &self.command_sender {
            let _ = sender.send(LoaderCommand::LoadChunk(position));
        }
    }

    pub fn poll_responses(&mut self) -> Vec<LoaderResponse> {
        let mut responses = Vec::new();

        if let Some(receiver) = &self.response_receiver {
            // Collect all available responses
            while let Ok(response) = receiver.try_recv() {
                responses.push(response);
            }
        }

        responses
    }

    pub fn is_running(&self) -> bool {
        self.is_running
    }

    pub fn get_info(&self) -> (u32, u16, u64) {
        (self.sample_rate, self.num_channels, self.total_samples)
    }
}

impl Drop for AudioLoader {
    fn drop(&mut self) {
        self.stop();
    }
}