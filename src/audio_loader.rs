use std::fs::File;
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::core::formats::*;

#[derive(Debug, Clone)]
pub enum LoaderCommand {
    LoadChunk(u64), // Position to load
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
        
        // Get audio metadata to determine total samples
        let (sample_rate, num_channels, total_samples) = match self.get_file_info(&file_path) {
            Ok((sr, nc, ts)) => {
                self.sample_rate = sr;
                self.num_channels = nc;
                self.total_samples = ts;
                (sr, nc, ts)
            },
            Err(e) => return Err(e),
        };
        
        // Create and start the thread
        let file_path_clone = file_path.clone();
        let thread = thread::spawn(move || {
            Self::loader_thread(file_path_clone, chunk_size, sample_rate, num_channels, cmd_rx, resp_tx);
        });
        
        self.thread = Some(thread);
        self.command_sender = Some(cmd_tx);
        self.response_receiver = Some(resp_rx);
        self.is_running = true;
        
        // Send file info back immediately
        if let Some(sender) = &self.command_sender {
            let _ = sender.send(LoaderCommand::LoadChunk(0));
        }
        
        Ok(())
    }
    
    fn get_file_info(&self, path: &PathBuf) -> Result<(u32, u16, u64), String> {
        match File::open(path) {
            Ok(file) => {
                let hint = Hint::new();
                let source = Box::new(file);
                let mss = MediaSourceStream::new(source, Default::default());

                let format_opts = FormatOptions {
                    seek_index_fill_rate: 100,
                    ..Default::default()
                };
                let metadata_opts = MetadataOptions::default();

                match symphonia::default::get_probe().format(&hint, mss, &format_opts, &metadata_opts) {
                    Ok(probed) => {
                        let mut format = probed.format;

                        match format.default_track() {
                            Some(track) => {
                                let sample_rate = track.codec_params.sample_rate.unwrap_or(44100);
                                let num_channels = track.codec_params.channels.unwrap_or(symphonia::core::audio::Channels::empty()).count() as u16;
                                
                                // Estimate total duration
                                let total_samples = if let Some(n_frames) = track.codec_params.n_frames {
                                    n_frames * num_channels as u64
                                } else if let Some(duration) = track.codec_params.time_base.and_then(|tb|
                                    track.codec_params.n_frames.map(|frames| tb.calc_time(frames))
                                ) {
                                    ((duration.seconds as f64 + duration.frac) * sample_rate as f64 * num_channels as f64) as u64
                                } else {
                                    // If we can't determine duration, make a rough guess
                                    if let Ok(file) = File::open(path) {
                                        if let Ok(metadata) = file.metadata() {
                                            let file_size = metadata.len();
                                            // Rough estimate for common formats: bytes per sample ~= 2, plus overhead
                                            file_size / 3 * num_channels as u64
                                        } else {
                                            0
                                        }
                                    } else {
                                        0
                                    }
                                };
                                
                                Ok((sample_rate, num_channels, total_samples))
                            },
                            None => Err("No default audio track found".to_string()),
                        }
                    },
                    Err(e) => Err(format!("Error probing format: {:?}", e)),
                }
            },
            Err(e) => Err(format!("Error opening file: {:?}", e)),
        }
    }
    
    fn loader_thread(
        file_path: PathBuf,
        chunk_size: u64,
        sample_rate: u32,
        num_channels: u16,
        cmd_rx: mpsc::Receiver<LoaderCommand>,
        resp_tx: mpsc::Sender<LoaderResponse>
    ) {
        println!("Loader thread started");
        
        // Send initial file info
        let _ = resp_tx.send(LoaderResponse::FileInfo {
            sample_rate,
            num_channels,
            total_samples: 0, // Will be calculated later
        });
        
        // Process commands
        loop {
            match cmd_rx.recv() {
                Ok(cmd) => match cmd {
                    LoaderCommand::LoadChunk(position) => {
                        println!("Loading chunk at position: {}", position);
                        match Self::load_chunk(&file_path, position, chunk_size, sample_rate, num_channels) {
                            Ok(samples) => {
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
    
    fn load_chunk(
        path: &PathBuf,
        position: u64,
        chunk_size: u64,
        sample_rate: u32,
        channels: u16
    ) -> Result<Vec<f32>, String> {
        // Use Symphonia for all formats
        let file = match File::open(path) {
            Ok(f) => f,
            Err(e) => return Err(format!("Error opening file: {:?}", e)),
        };
        let source = Box::new(file);

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

        let mss = MediaSourceStream::new(source, Default::default());
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
        
        // Calculate position in frames (not samples)
        // Samples are interleaved, so we need to divide by channel count to get frame position
        let frame_pos = position / channels as u64;

        // Create decoder before seeking
        let mut decoder = match symphonia::default::get_codecs()
            .make(&codec_params, &decoder_opts) {
            Ok(d) => d,
            Err(e) => return Err(format!("Error creating decoder: {:?}", e)),
        };

        // Calculate timestamp for frame_pos
        let time_base = codec_params.time_base.unwrap_or_default();
        let timestamp = time_base.calc_time(frame_pos);
        
        // Attempt to seek
        let seek_result = format.seek(
            SeekMode::Accurate,
            SeekTo::Time {
                track_id: Some(track_id),
                time: timestamp
            },
        );

        match &seek_result {
            Ok(seeked_to) => {
                println!("Successfully seeked to: {:?}", seeked_to);
                // Reset decoder after seeking
                decoder.reset();
            },
            Err(e) => {
                println!("Error seeking: {:?}. Will read from beginning.", e);
            }
        };

        // After seeking, we need to:
        // 1. Decode packets until we reach our target sample
        // 2. Ensure correct alignment of sample position
        let target_frame = frame_pos;
        let mut current_frame: u64 = 0;
        
        if let Ok(seeked_to) = seek_result {
            // If we successfully seeked, update current frame position
            current_frame = seeked_to.actual_ts;
            println!("After seek, current frame position is: {}", current_frame);
        }

        // Whether we need to skip some frames to reach target
        let mut need_frame_skip = current_frame < target_frame;
        
        // Process audio packets and collect samples
        let mut samples = Vec::with_capacity(chunk_size as usize);
        
        'decode_loop: loop {
            let packet = match format.next_packet() {
                Ok(packet) => packet,
                Err(e) => {
                    println!("End of file or error: {:?}", e);
                    break 'decode_loop;
                },
            };

            if packet.track_id() != track_id {
                continue;
            }

            match decoder.decode(&packet) {
                Ok(decoded) => {
                    // Get number of frames in this decoded buffer
                    let frame_count = decoded.frames();
                    
                    // Update current frame counter
                    if need_frame_skip {
                        // We're still skipping to reach target
                        current_frame += frame_count as u64;
                        
                        if current_frame >= target_frame {
                            // We've reached our target frame
                            let frames_to_skip = frame_count as u64 - (current_frame - target_frame);
                            println!("Reached target frame. Need to skip {} frames in this buffer", frames_to_skip);
                            need_frame_skip = false;
                            
                            // Convert to a sample buffer with the right specifications
                            let mut sample_buf = SampleBuffer::<f32>::new(
                                decoded.capacity() as u64,
                                *decoded.spec(),
                            );
                            
                            // Copy and convert samples from the decoded buffer
                            sample_buf.copy_interleaved_ref(decoded);
                            let buffer_samples = sample_buf.samples();
                            
                            // Calculate how many samples to skip (frames * channels)
                            let samples_to_skip = (frames_to_skip as usize * channels as usize)
                                .min(buffer_samples.len());
                            
                            // Add the rest of the samples
                            samples.extend_from_slice(&buffer_samples[samples_to_skip..]);
                        }
                    } else {
                        // We're already at or past our target position, just collect samples
                        // Convert to a sample buffer with the right specifications
                        let mut sample_buf = SampleBuffer::<f32>::new(
                            decoded.capacity() as u64,
                            *decoded.spec(),
                        );
                        
                        // Copy and convert samples from the decoded buffer
                        sample_buf.copy_interleaved_ref(decoded);
                        let buffer_samples = sample_buf.samples();
                        
                        // Add samples to our output
                        samples.extend_from_slice(buffer_samples);
                    }
                    
                    // Check if we have enough samples
                    if samples.len() >= chunk_size as usize {
                        // Trim to exact chunk size
                        if samples.len() > chunk_size as usize {
                            samples.truncate(chunk_size as usize);
                        }
                        break 'decode_loop;
                    }
                },
                Err(e) => {
                    println!("Error decoding packet: {:?}", e);
                    continue;
                },
            }
        }
        
        println!("Loaded chunk at position {}, with {} samples", position, samples.len());
        Ok(samples)
    }
    
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