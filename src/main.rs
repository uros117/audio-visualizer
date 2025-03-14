use std::collections::VecDeque;
use std::fs::File;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use eframe::egui;
use egui::{Color32, Pos2, Rect, Stroke, StrokeKind, Ui, Vec2};
use fixed::FixedI64;
use fixed::types::I48F16;
use rfd::FileDialog;
use ringbuf::{HeapRb, traits::*};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::core::formats::*;

const FFT_SIZE: usize = 2048;
const BUFFER_SIZE: usize = 1024; // 1MB chunks for loading
struct AudioVisualizer {
    file_path: Option<PathBuf>,
    samples: Vec<f32>,
    spectrum: Vec<f32>,
    loaded_percentage: f32,
    chunk_start: u64,           // Integer sample position
    chunk_size: u64,            // Integer chunk size
    total_samples: u64,         // Integer total sample count
    sample_rate: u32,
    num_channels: u16,
    loading_thread: Option<thread::JoinHandle<()>>,
    ring_buffer: Option<Arc<Mutex<HeapRb<f32>>>>,
    is_loading: bool,
    visible_time_window: I48F16,  // Fixed-point time window
    zoom_sensitivity: I48F16,     // Fixed-point zoom sensitivity
    scroll_offset: I48F16,        // Fixed-point scroll position
    scroll_sensitivity: I48F16,   // Fixed-point scroll sensitivity

    chunks: VecDeque<Vec<f32>>,
    max_chunks: usize,            // Maximum number of chunks to keep in memory
    current_chunk_index: usize,   // Index of the "current" chunk in the circular buffer
    loading_direction: i32,       // Direction of loading: -1 = backward, 0 = none, 1 = forward
    pending_chunk_position: u64,  // Position of the chunk being loaded
}

impl Default for AudioVisualizer {
    fn default() -> Self {
        let ring_buffer = HeapRb::<f32>::new(BUFFER_SIZE * 2);

        let max_chunks = 10240;
        Self {
            file_path: None,
            samples: Vec::new(),
            spectrum: vec![0.0; FFT_SIZE / 2],
            loaded_percentage: 0.0,
            chunk_start: 0,
            chunk_size: BUFFER_SIZE as u64,
            total_samples: 0,
            sample_rate: 44100,
            num_channels: 2,
            loading_thread: None,
            ring_buffer: Some(Arc::new(Mutex::new(ring_buffer))),
            is_loading: false,
            // Using from_num for conversion from f64 to I48F16
            visible_time_window: I48F16::from_num(20.0), // 20 seconds
            zoom_sensitivity: I48F16::from_num(10.0),
            scroll_offset: I48F16::from_num(0.0),
            scroll_sensitivity: I48F16::from_num(0.2),
            max_chunks,            // Store up to 10 chunks
            current_chunk_index: 0,      // Start at the first chunk
            loading_direction: 0,        // No loading initially
            pending_chunk_position: 0,   // No pending load
            chunks: VecDeque::with_capacity(max_chunks),
        }
    }
}

impl AudioVisualizer {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self::default()
    }

    // Convert fixed-point to f32
    fn fixed_to_f32(&self, value: I48F16) -> f32 {
        value.to_num()
    }


    fn open_file(&mut self) {
        if let Some(path) = FileDialog::new()
            .add_filter("Audio", &["wav", "mp3", "flac", "aac", "ogg"])
            .pick_file()
        {
            println!("Opening file: {:?}", path);
            self.file_path = Some(path.clone());
            // Clear existing chunks
            self.chunks.clear();
            self.spectrum = vec![0.0; FFT_SIZE / 2];
            self.loaded_percentage = 0.0;
            self.chunk_start = 0;
            self.current_chunk_index = 0;
            self.loading_direction = 0;
            self.pending_chunk_position = 0;

            // Get audio metadata to determine total samples
            match File::open(&path) {
                Ok(file) => {
                    let hint = Hint::new();
                    let source = Box::new(file);
                    let mss = MediaSourceStream::new(source, Default::default());

                    let format_opts = FormatOptions {
                        seek_index_fill_rate: 100, // Build a complete seek index
                        ..Default::default()
                    };
                    let metadata_opts = MetadataOptions::default();
                    let decoder_opts = DecoderOptions::default();

                    match symphonia::default::get_probe().format(&hint, mss, &format_opts, &metadata_opts) {
                        Ok(probed) => {
                            let mut format = probed.format;

                            // Print available tracks for debugging
                            println!("Available tracks: {}", format.tracks().len());
                            for (i, track) in format.tracks().iter().enumerate() {
                                println!("  Track {}: codec={:?}, sample_rate={:?}, channels={:?}",
                                         i,
                                         track.codec_params.codec,
                                         track.codec_params.sample_rate,
                                         track.codec_params.channels);
                            }

                            match format.default_track() {
                                Some(track) => {
                                    self.sample_rate = track.codec_params.sample_rate.unwrap_or(44100);
                                    self.num_channels = track.codec_params.channels.unwrap_or(symphonia::core::audio::Channels::empty()).count() as u16;

                                    println!("Selected track: sample rate={}, channels={}", self.sample_rate, self.num_channels);

                                    // Estimate total duration
                                    if let Some(n_frames) = track.codec_params.n_frames {
                                        self.total_samples = n_frames * self.num_channels as u64;
                                        println!("Total samples from n_frames: {}", self.total_samples);
                                    } else if let Some(duration) = track.codec_params.time_base.and_then(|tb|
                                        // Instead of using format.duration(), use n_frames if available
                                        track.codec_params.n_frames.map(|frames| tb.calc_time(frames))
                                    ) {
                                        self.total_samples = ((duration.seconds as f64 + duration.frac) * self.sample_rate as f64 * self.num_channels as f64) as u64;
                                        println!("Total samples from duration: {}", self.total_samples);
                                    } else {
                                        // If we can't determine duration, make a rough guess
                                        if let Ok(file) = File::open(&path) {
                                            if let Ok(metadata) = file.metadata() {
                                                let file_size = metadata.len();
                                                // Rough estimate for common formats: bytes per sample ~= 2, plus overhead
                                                self.total_samples = file_size / 3 * self.num_channels as u64;
                                                println!("Estimated total samples from file size: {}", self.total_samples);
                                            }
                                        }
                                    }

                                    // Initialize the ring buffer with the proper size
                                    let ring_buffer = HeapRb::<f32>::new(BUFFER_SIZE * 2);
                                    self.ring_buffer = Some(Arc::new(Mutex::new(ring_buffer)));

                                    // Start loading the first chunk
                                    self.load_chunk_at_position(0);
                                },
                                None => {
                                    println!("No default track found");
                                }
                            }
                        },
                        Err(e) => {
                            println!("Error probing format: {:?}", e);
                        }
                    }
                },
                Err(e) => {
                    println!("Error opening file: {:?}", e);
                }
            }
        }
    }

    fn get_sample_at(&self, position: u64) -> Option<f32> {
        if self.chunks.is_empty() {
            return None;
        }

        // Calculate chunk offset from start
        if position < self.chunk_start {
            return None; // Before our buffer
        }

        let relative_pos = position - self.chunk_start;
        let chunk_index = (relative_pos / self.chunk_size) as usize;
        let sample_index = (relative_pos % self.chunk_size) as usize;

        if chunk_index >= self.chunks.len() {
            return None; // After our buffer
        }

        // Now get the sample from the correct chunk
        self.chunks.get(chunk_index).and_then(|chunk| {
            if sample_index < chunk.len() {
                Some(chunk[sample_index])
            } else {
                None
            }
        })
    }

    fn check_buffer_needs(&mut self) {
        if self.is_loading {
            return; // Already loading something
        }

        if self.file_path.is_none() {
            return; // No file loaded
        }

        // Convert visible window to samples
        let visible_samples_start = (self.scroll_offset.to_num::<f64>() * self.sample_rate as f64 * self.num_channels as f64) as u64;
        let visible_samples_end = ((self.scroll_offset + self.visible_time_window).to_num::<f64>() * self.sample_rate as f64 * self.num_channels as f64) as u64;

        // Check if we're near the start of our buffer
        if visible_samples_start < self.chunk_start + self.chunk_size {
            // Need to load backwards if possible
            if self.chunk_start > 0 {
                let new_start = self.chunk_start.saturating_sub(self.chunk_size);
                self.load_chunk_at_position(new_start);
                return;
            }
        }

        // Check if we're near the end of our buffer
        let buffer_end = self.chunk_start + (self.chunks.len() as u64 * self.chunk_size);
        if visible_samples_end > buffer_end.saturating_sub(self.chunk_size) {
            // Need to load forward if possible
            if buffer_end < self.total_samples {
                self.load_chunk_at_position(buffer_end);
                return;
            }
        }
    }

    // Revised method to load a chunk at a specific position
    fn load_chunk_at_position(&mut self, position: u64) {
        if self.is_loading {
            return;
        }

        println!("Requesting chunk at position: {}", position);

        // Ensure position is aligned to chunk boundaries
        let aligned_position = (position / self.chunk_size) * self.chunk_size;

        // Check if this chunk is already loaded
        let relative_pos = aligned_position.checked_sub(self.chunk_start);
        if let Some(rel_pos) = relative_pos {
            let chunk_index = (rel_pos / self.chunk_size) as usize;
            if chunk_index < self.chunks.len() {
                // Already loaded
                println!("Chunk already loaded at index {}", chunk_index);
                return;
            }
        }

        // Determine loading direction
        if relative_pos.is_some() && relative_pos.unwrap() >= self.chunks.len() as u64 * self.chunk_size {
            self.loading_direction = 1; // Forward
            println!("Loading forward");
        } else {
            self.loading_direction = -1; // Backward
            println!("Loading backward");
            // If loading backward, we'll need to adjust the buffer later
        }

        self.pending_chunk_position = aligned_position;
        self.start_loading_thread(aligned_position);
    }

    fn scale_width(&self, time_duration: I48F16, rect: Rect) -> f32 {
        rect.width() * (time_duration / self.visible_time_window).to_num::<f32>()
    }

    fn time_to_screen_x(&self, time: I48F16, rect: Rect) -> f32 {
        rect.left() + rect.width() * ((time - self.scroll_offset) / self.visible_time_window).to_num::<f32>()
    }

    fn get_chunk_duration(&self) -> I48F16 {
        I48F16::from_num(self.chunk_size) / I48F16::from_num(self.sample_rate)
    }

    fn draw_chunk_borders(&self, ui: &mut Ui, rect: Rect) {
        if self.sample_rate == 0 {
            return;
        }

        // Fixed-point comparison
        if self.visible_time_window >= I48F16::from_num(3.0) { return; }

        let painter = ui.painter();

        // No need for explicit conversion since we're using fixed-point directly
        let start_time = self.scroll_offset;
        let end_time = self.scroll_offset + self.visible_time_window;

        // Calculate duration of each chunk in seconds
        let chunk_duration = self.get_chunk_duration();

        // Calculate visible chunks using fixed-point math
        let visible_chunk_window = (self.visible_time_window / chunk_duration).ceil();
        let number_of_visible_chunks = visible_chunk_window.to_num::<u32>();

        let start_idx = (self.scroll_offset / chunk_duration).floor();
        let end_idx = start_idx + visible_chunk_window;

        // Drawing parameters
        let height = 20.0; // Height for time markers area
        let y_pos = rect.bottom() - height;

        // Draw time markers
        let mut current_x_pos: f32 = self.time_to_screen_x(start_idx * chunk_duration,rect);
        let mut current_index: u32 = start_idx.to_num();
        let chunk_width: f32 = self.scale_width(chunk_duration, rect);
        for i in 0..number_of_visible_chunks {
            let current_time = start_idx * I48F16::from_num(i) * chunk_duration;

            let chunk_rect = Rect::from_min_max(
                Pos2::new(current_x_pos, rect.top()),
                Pos2::new(current_x_pos + chunk_width, rect.bottom())
            );

            painter.rect_stroke(
                chunk_rect,
                1.0,
                Stroke::new(1.0, Color32::LIGHT_GREEN),
                StrokeKind::Inside,
            );

            painter.text(
                chunk_rect.center(),
                egui::Align2::CENTER_CENTER,
                current_index,
                egui::FontId::monospace(10.0),
                Color32::LIGHT_GREEN
            );

            current_x_pos += chunk_width;
            current_index += 1;
        }
    }

    fn draw_time_markers(&self, ui: &mut Ui, rect: Rect) {
        if self.sample_rate == 0 {
            return;
        }

        let painter = ui.painter();

        // No need for explicit conversion
        let start_time = self.scroll_offset;
        let end_time = self.scroll_offset + self.visible_time_window;

        // Drawing parameters
        let height = 20.0; // Height for time markers area
        let y_pos = rect.bottom() - height;

        // Calculate time range
        let time_range = end_time - start_time;

        // Determine appropriate time interval based on zoom level
        let interval = if self.visible_time_window < I48F16::from_num(1.0) {
            I48F16::from_num(0.1) // 100ms
        } else if self.visible_time_window < I48F16::from_num(5.0) {
            I48F16::from_num(0.5) // 500ms
        } else if self.visible_time_window < I48F16::from_num(30.0) {
            I48F16::from_num(1.0) // 1 second
        } else if self.visible_time_window < I48F16::from_num(120.0) {
            I48F16::from_num(5.0) // 5 seconds
        } else if self.visible_time_window < I48F16::from_num(300.0) {
            I48F16::from_num(10.0) // 10 seconds
        } else if self.visible_time_window < I48F16::from_num(600.0) {
            I48F16::from_num(30.0) // 30 seconds
        } else {
            I48F16::from_num(60.0) // 1 minute
        };

        let screen_space_interval: f32 = self.scale_width(interval, rect);

        // Calculate the first time marker (rounded to the nearest interval)
        let first_marker_time: I48F16 = (start_time / interval).ceil() * interval;
        let first_marker = self.time_to_screen_x(first_marker_time, rect);
        let last_marker_pos = self.time_to_screen_x(end_time, rect);

        // Draw background for time markers
        painter.rect_filled(
            Rect::from_min_max(
                Pos2::new(rect.left(), y_pos),
                Pos2::new(rect.right(), rect.bottom())
            ),
            0.0,
            Color32::from_gray(30)
        );

        // Draw time markers
        let mut current_x_pos: f32 = first_marker;
        let mut current_time: I48F16 = first_marker_time;
        while current_x_pos <= last_marker_pos {
            // Convert time to x-position
            let x_pos = current_x_pos;

            // Draw marker line
            painter.line_segment(
                [Pos2::new(x_pos, y_pos), Pos2::new(x_pos, y_pos + 5.0)],
                Stroke::new(1.0, Color32::from_gray(200))
            );

            // Format time as MM:SS.ms
            let minutes = (current_time / I48F16::from_num(60.0)).floor().to_num::<i32>();
            let seconds = (current_time % I48F16::from_num(60.0)).floor().to_num::<i32>();
            let ms = ((current_time % I48F16::from_num(1.0)) * I48F16::from_num(100.0)).floor().to_num::<i32>();

            let time_text = if interval < I48F16::from_num(1.0) {
                format!("{:02}:{:02}.{:02}", minutes, seconds, ms)
            } else {
                format!("{:02}:{:02}", minutes, seconds)
            };

            // Draw time text
            painter.text(
                Pos2::new(x_pos, y_pos + 8.0),
                egui::Align2::CENTER_TOP,
                time_text,
                egui::FontId::monospace(10.0),
                Color32::from_gray(200)
            );

            current_x_pos += screen_space_interval;
            current_time += interval;
        }
    }

    fn handle_scroll_input(&mut self, ui: &mut Ui, rect: Rect) {
        // Capture pointer hover
        let hover_pos = ui.input(|i| i.pointer.hover_pos());

        // Only process inputs if mouse is hovering over the waveform area
        if let Some(pos) = hover_pos {
            if rect.contains(pos) {
                // Get normalized x position (0.0 to 1.0) relative to the visible area
                let norm_x = (pos.x - rect.left()) / rect.width();

                // Handle scroll events
                ui.input(|i| {
                    // Get scroll delta
                    let scroll_delta = i.smooth_scroll_delta;

                    // Check if Alt key is pressed for zooming
                    let alt_pressed = i.modifiers.alt;

                    if alt_pressed && scroll_delta.y != 0.0 {
                        // Zoom in/out with proper fixed-point math
                        let old_zoom = self.visible_time_window;

                        // Convert scroll_delta to fixed-point
                        let scroll_delta_fixed = I48F16::from_num(scroll_delta.y);
                        let div_factor = I48F16::from_num(120.0);
                        let scale_factor = I48F16::from_num(0.1);

                        // Calculate delta factor
                        let delta_factor = scroll_delta_fixed / div_factor * scale_factor * self.zoom_sensitivity;

                        // Calculate zoom factor
                        let zoom_factor = I48F16::from_num(1.0) + delta_factor;

                        // Calculate new zoom with clamping
                        let new_zoom_tmp = self.visible_time_window * zoom_factor;
                        let min_zoom = I48F16::from_num(1.0); // 1 s
                        let max_zoom = I48F16::from_num(600.0); // 10 min

                        let new_zoom = if new_zoom_tmp < min_zoom {
                            min_zoom
                        } else if new_zoom_tmp > max_zoom {
                            max_zoom
                        } else {
                            new_zoom_tmp
                        };

                        // Adjust scroll position to keep the point under cursor at the same place
                        if old_zoom != new_zoom {
                            self.visible_time_window = new_zoom;
                            // Additional cursor position adjustment could be added here
                        }

                        return true; // Consumed event
                    } else if scroll_delta.y != 0.0 {
                        // Horizontal scrolling (time navigation) with fixed-point precision
                        let delta = I48F16::from_num(scroll_delta.y) / I48F16::from_num(120.0);
                        let scroll_amount = self.scroll_sensitivity * self.visible_time_window * delta;

                        let new_scroll = self.scroll_offset + scroll_amount;

                        // Update scroll offset with precise fixed-point addition
                        self.scroll_offset = if new_scroll >= 0 { new_scroll } else { FixedI64::ZERO };

                        return true; // Consumed event
                    }

                    false // Event not consumed
                });
            }
        }
    }


    fn update_chunk(&mut self) {
        if !self.is_loading {
            return;
        }

        // Get new samples from the ring buffer
        let mut new_samples = Vec::with_capacity(self.chunk_size as usize);
        let mut sample_count = 0;

        if let Some(rb) = &self.ring_buffer {
            if let Ok(mut rb_lock) = rb.lock() {
                // Extract all available samples
                sample_count = rb_lock.occupied_len();
                while let Some(sample) = rb_lock.try_pop() {
                    new_samples.push(sample);
                }
            }
        }

        if !new_samples.is_empty() {
            // Debug info
            println!("Got {} new samples (out of {} in buffer). First few: {:?}",
                     new_samples.len(),
                     sample_count,
                     &new_samples[0..new_samples.len().min(10)]);

            // Create a chunk even if it's not a full chunk
            if self.loading_direction > 0 {
                // Loading forward - append to the end
                println!("Adding chunk in forward direction, size: {}", new_samples.len());
                self.chunks.push_back(new_samples);

                // If we have too many chunks, remove from the front
                while self.chunks.len() > self.max_chunks {
                    self.chunks.pop_front();
                    self.chunk_start += self.chunk_size;
                }

            } else if self.loading_direction < 0 {
                // Loading backward - prepend to the front
                println!("Adding chunk in backward direction, size: {}", new_samples.len());
                self.chunks.push_front(new_samples);
                self.chunk_start = self.pending_chunk_position;

                // If we have too many chunks, remove from the back
                while self.chunks.len() > self.max_chunks {
                    self.chunks.pop_back();
                }
            }

            // Calculate loaded percentage
            let buffer_samples = self.chunks.len() as u64 * self.chunk_size;
            self.loaded_percentage = buffer_samples as f32 / self.total_samples as f32 * 100.0;

            println!("Updated chunks. Now have {} chunks, loaded {}%",
                     self.chunks.len(), self.loaded_percentage);
        }

        // Check if thread is done
        if let Some(handle) = &self.loading_thread {
            if handle.is_finished() {
                println!("Loading thread finished");
                self.loading_thread = None;

                // Reset the ring buffer
                if let Some(rb) = &self.ring_buffer {
                    if let Ok(mut rb_lock) = rb.lock() {
                        rb_lock.clear();
                    }
                }

                self.is_loading = false;
                self.loading_direction = 0;

                // Check if we need to load more chunks
                self.check_buffer_needs();
            }
        }
    }

    fn start_loading_thread(&mut self, position: u64) {
        if self.is_loading {
            return;
        }
    
        println!("Starting loading thread at position: {}", position);
    
        self.is_loading = true;
        let path = self.file_path.as_ref().unwrap().clone();
        let chunk_size = self.chunk_size;
        let rb = self.ring_buffer.clone().unwrap();
        let total_samples = self.total_samples;
        let channels = self.num_channels;
    
        let handle = thread::spawn(move || {
            // Use Symphonia for all formats
            let file = match File::open(&path) {
                Ok(f) => f,
                Err(e) => {
                    println!("Error opening file: {:?}", e);
                    return;
                }
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
                seek_index_fill_rate: 100, // Ensure seeking index is built comprehensively
                ..Default::default()
            };
            let metadata_opts = MetadataOptions::default();
            let decoder_opts = DecoderOptions::default();
    
            let mss = MediaSourceStream::new(source, Default::default());
            let probed = match symphonia::default::get_probe()
                .format(&hint, mss, &format_opts, &metadata_opts) {
                Ok(p) => p,
                Err(e) => {
                    println!("Error probing format: {:?}", e);
                    return;
                }
            };
    
            let mut format = probed.format;
    
            // Get the default track
            let track = match format.default_track() {
                Some(t) => t,
                None => {
                    println!("No default track found");
                    return;
                }
            };
    
            let track_id = track.id;
            let codec_params = track.codec_params.clone();
            let num_channels = codec_params.channels.unwrap_or(symphonia::core::audio::Channels::empty()).count() as u16;
            let sample_rate = codec_params.sample_rate.unwrap_or(44100);
    
            println!("Track info: codec={:?}, sample_rate={}, channels={}",
                     codec_params.codec, sample_rate, num_channels);
    
            // Calculate position in frames (not samples)
            // Samples are interleaved, so we need to divide by channel count to get frame position
            let frame_pos = position / channels as u64;
    
            // Now convert to time for the API
            let time_base = codec_params.time_base.unwrap_or_default();
            
            // Calculate timestamp for frame_pos
            let timestamp = time_base.calc_time(frame_pos);
            
            println!("Seeking to frame position {} (timestamp: {:?})", frame_pos, timestamp);
    
            // Create decoder before seeking
            let mut decoder = match symphonia::default::get_codecs()
                .make(&codec_params, &decoder_opts) {
                Ok(d) => d,
                Err(e) => {
                    println!("Error creating decoder: {:?}", e);
                    return;
                }
            };
    
            // Attempt to seek using accurate mode (always seeks to a position before the requested one)
            let seek_result = format.seek(
                symphonia::core::formats::SeekMode::Accurate,
                symphonia::core::formats::SeekTo::Time {
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
                let seeked_ts = seeked_to.actual_ts;

                // Use the actual timestamp directly
                current_frame = seeked_ts;
                println!("After seek, current frame position is: {}", current_frame);
            }
    
            // Whether we need to skip some frames to reach target
            let mut need_frame_skip = current_frame < target_frame;
            
            // Process audio packets and collect samples
            let mut buffer_samples = 0;
            
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
    
                // Get packet timestamp for frame alignment
                let packet_ts = packet.ts();
                
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
                                let samples = sample_buf.samples();
                                
                                // Calculate how many samples to skip (frames * channels)
                                let samples_to_skip = (frames_to_skip as usize * num_channels as usize)
                                    .min(samples.len());
                                
                                // Push the rest of the samples to the ring buffer
                                for i in samples_to_skip..samples.len() {
                                    if let Ok(mut rb_lock) = rb.lock() {
                                        // Make space if needed
                                        if rb_lock.is_full() {
                                            let _ = rb_lock.try_pop();
                                        }
    
                                        // Try to push the sample
                                        match rb_lock.try_push(samples[i]) {
                                            Ok(_) => {
                                                buffer_samples += 1;
                                                // Break if we've collected enough samples
                                                if buffer_samples >= chunk_size as usize {
                                                    println!("Buffer full with {} samples, breaking", buffer_samples);
                                                    break 'decode_loop;
                                                }
                                            },
                                            Err(e) => println!("Error pushing to ring buffer: {:?}", e)
                                        }
                                    } else {
                                        thread::sleep(Duration::from_millis(1));
                                    }
                                }
                            } else {
                                // Still need to skip this entire buffer
                                println!("Skipping buffer, current_frame={}, target_frame={}", current_frame, target_frame);
                                continue;
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
                            let samples = sample_buf.samples();
                            
                            if !samples.is_empty() {
                                // Debug - show first few samples to verify they're not all zero
                                println!("Sample buffer (len: {}). First few: {:?}",
                                        samples.len(),
                                        &samples[0..samples.len().min(10)]);
                                
                                // Push samples to the ring buffer
                                for sample in samples {
                                    if let Ok(mut rb_lock) = rb.lock() {
                                        // Make space if needed
                                        if rb_lock.is_full() {
                                            let _ = rb_lock.try_pop();
                                        }
    
                                        // Try to push the sample
                                        match rb_lock.try_push(*sample) {
                                            Ok(_) => {
                                                buffer_samples += 1;
                                                // Break if we've collected enough samples
                                                if buffer_samples >= chunk_size as usize {
                                                    println!("Buffer full with {} samples, breaking", buffer_samples);
                                                    break 'decode_loop;
                                                }
                                            },
                                            Err(e) => println!("Error pushing to ring buffer: {:?}", e)
                                        }
                                    } else {
                                        thread::sleep(Duration::from_millis(1));
                                    }
                                }
                            }
                        }
                    },
                    Err(e) => {
                        println!("Error decoding packet: {:?}", e);
                        continue;
                    },
                }
            }
    
            println!("Finished loading thread, collected {} samples", buffer_samples);
        });
    
        self.loading_thread = Some(handle);
    }

    fn draw_waveform(&self, ui: &mut Ui, rect: Rect) {
        if self.chunks.is_empty() {
            return;
        }

        let painter = ui.painter();

        // Calculate visible sample range
        let visible_start_time = self.scroll_offset;
        let visible_end_time = self.scroll_offset + self.visible_time_window;

        let visible_start_sample = (visible_start_time.to_num::<f64>() * self.sample_rate as f64 * self.num_channels as f64) as u64;
        let visible_end_sample = (visible_end_time.to_num::<f64>() * self.sample_rate as f64 * self.num_channels as f64) as u64;

        // Determine how many pixels per sample
        let samples_per_pixel = ((visible_end_sample - visible_start_sample) as f32 / rect.width()).max(1.0);
        let pixels_per_sample = 1.0 / samples_per_pixel;

        // Draw waveform
        let height = rect.height();
        let mid_y = rect.center().y;

        // Draw background
        painter.rect_filled(
            rect,
            0.0,
            Color32::from_gray(20)
        );

        // Draw centerline
        painter.line_segment(
            [Pos2::new(rect.left(), mid_y), Pos2::new(rect.right(), mid_y)],
            Stroke::new(1.0, Color32::from_gray(40))
        );

        // Calculate downsampling factor based on zoom level
        let downsample = samples_per_pixel.ceil() as usize;

        // Draw samples
        let mut last_x = rect.left();
        let mut last_max_y = mid_y;
        let mut last_min_y = mid_y;
        let channel_count = self.num_channels as usize;

        // Visualize chunk boundaries
        for i in 0..self.chunks.len() {
            let chunk_start_sample = self.chunk_start + (i as u64 * self.chunk_size);
            let chunk_end_sample = chunk_start_sample + self.chunk_size;

            if chunk_end_sample < visible_start_sample || chunk_start_sample > visible_end_sample {
                continue; // Skip chunks outside the visible range
            }

            // Calculate screen positions
            let chunk_start_time = I48F16::from_num(chunk_start_sample as f64 / (self.sample_rate as f64 * self.num_channels as f64));
            let chunk_end_time = I48F16::from_num(chunk_end_sample as f64 / (self.sample_rate as f64 * self.num_channels as f64));

            let chunk_start_x = self.time_to_screen_x(chunk_start_time, rect);
            let chunk_end_x = self.time_to_screen_x(chunk_end_time, rect);

            // Draw chunk background
            if i % 2 == 0 {
                painter.rect_filled(
                    Rect::from_min_max(
                        Pos2::new(chunk_start_x, rect.top()),
                        Pos2::new(chunk_end_x, rect.bottom())
                    ),
                    0.0,
                    Color32::from_rgb(30, 30, 40)
                );
            }
        }

        // Draw the actual waveform data
        for x in 0..rect.width() as usize {
            let start_sample = visible_start_sample + (x as f32 * samples_per_pixel) as u64;
            let end_sample = visible_start_sample + ((x + 1) as f32 * samples_per_pixel) as u64;

            // Find min/max in this range
            let mut max_val = -1.0f32;
            let mut min_val = 1.0f32;
            let mut has_sample = false;

            for s in (start_sample..end_sample).step_by(channel_count) {
                if let Some(sample) = self.get_sample_at(s) {
                    max_val = max_val.max(sample);
                    min_val = min_val.min(sample);
                    has_sample = true;
                }
            }

            if has_sample {
                // Map to screen coordinates
                let max_y = mid_y - (max_val * height * 0.5);
                let min_y = mid_y - (min_val * height * 0.5);

                // Draw vertical line for this column of pixels
                painter.line_segment(
                    [Pos2::new(rect.left() + x as f32, min_y), Pos2::new(rect.left() + x as f32, max_y)],
                    Stroke::new(1.0, Color32::from_rgb(100, 200, 100))
                );

                // Connect with previous point for smoother waveform
                if x > 0 {
                    painter.line_segment(
                        [Pos2::new(last_x, last_max_y), Pos2::new(rect.left() + x as f32, max_y)],
                        Stroke::new(1.0, Color32::from_rgb(100, 200, 100))
                    );

                    painter.line_segment(
                        [Pos2::new(last_x, last_min_y), Pos2::new(rect.left() + x as f32, min_y)],
                        Stroke::new(1.0, Color32::from_rgb(100, 200, 100))
                    );
                }

                last_x = rect.left() + x as f32;
                last_max_y = max_y;
                last_min_y = min_y;
            }
        }

        // Draw gaps in data if there are any
        let mut x = rect.left();
        while x < rect.right() {
            let time = self.scroll_offset + (I48F16::from_num((x - rect.left()) / rect.width()) * self.visible_time_window);
            let sample = (time.to_num::<f64>() * self.sample_rate as f64 * self.num_channels as f64) as u64;

            // Check if this sample is in a gap
            if self.get_sample_at(sample).is_none() {
                // Find the end of the gap
                let mut end_x = x;
                let mut end_time = time;
                let mut end_sample = sample;

                while end_x < rect.right() && self.get_sample_at(end_sample).is_none() {
                    end_x += 1.0;
                    end_time = self.scroll_offset + (I48F16::from_num((end_x - rect.left()) / rect.width()) * self.visible_time_window);
                    end_sample = (end_time.to_num::<f64>() * self.sample_rate as f64 * self.num_channels as f64) as u64;
                }

                // Draw the gap indicator
                if end_x > x {
                    painter.rect_filled(
                        Rect::from_min_max(
                            Pos2::new(x, rect.top()),
                            Pos2::new(end_x, rect.bottom())
                        ),
                        0.0,
                        Color32::from_rgba_premultiplied(150, 50, 50, 100)
                    );

                    // Draw "Loading..." text if the gap is wide enough
                    if end_x - x > 50.0 {
                        painter.text(
                            Pos2::new((x + end_x) / 2.0, mid_y),
                            egui::Align2::CENTER_CENTER,
                            "Loading...",
                            egui::FontId::proportional(14.0),
                            Color32::WHITE
                        );
                    }

                    x = end_x;
                } else {
                    x += 1.0;
                }
            } else {
                x += 1.0;
            }
        }
    }
}

impl eframe::App for AudioVisualizer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Update loading status
        self.update_chunk();

        // Check if we need to load more chunks based on the visible area
        self.check_buffer_needs();

        // Main app menu
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open...").clicked() {
                        self.open_file();
                        ui.close_menu();
                    }
                    if ui.button("Exit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
            });
        });

        // Status bar
        egui::TopBottomPanel::bottom("bottom_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if let Some(path) = &self.file_path {
                    ui.label(format!("File: {}", path.file_name().unwrap().to_string_lossy()));
                    ui.separator();
                    ui.label(format!("Sample rate: {} Hz", self.sample_rate));
                    ui.separator();
                    ui.label(format!("Channels: {}", self.num_channels));
                    ui.separator();
                    ui.label(format!("Loaded: {:.1}%", self.loaded_percentage));
                    ui.separator();
                    ui.label(format!("Buffer: {} chunks", self.chunks.len()));
                    ui.separator();
                    ui.label(format!("Position: {} samples", self.chunk_start));
                    ui.separator();
                    // Display fixed-point values with high precision
                    ui.label(format!("Time: {:.6} s", self.scroll_offset));

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Next Chunk").clicked() {
                            // Instead of loading next chunk directly, move to it
                            let next_pos = self.chunk_start + self.chunk_size * self.chunks.len() as u64;
                            if next_pos < self.total_samples {
                                self.load_chunk_at_position(next_pos);
                            }
                        }
                        if ui.button("Previous Chunk").clicked() {
                            // Load previous chunk if we're not at the beginning
                            if self.chunk_start > 0 {
                                let prev_pos = self.chunk_start.saturating_sub(self.chunk_size);
                                self.load_chunk_at_position(prev_pos);
                            }
                        }
                    });
                } else {
                    ui.label("No file loaded. Use File > Open to load an audio file.");
                }
            });
        });

        // Main content area
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.file_path.is_none() {
                ui.centered_and_justified(|ui| {
                    ui.heading("Open an audio file to start visualizing");
                });
                return;
            }

            // Request continuous repainting while loading
            if self.is_loading {
                ctx.request_repaint();
            }

            // Calculate layout
            let available_size = ui.available_size();
            let waveform_height = available_size.y - 20.0; // Reserve space for time markers

            // Create rect for waveform
            let waveform_rect = ui.allocate_rect(Rect::from_min_size(
                ui.cursor().min,
                Vec2::new(available_size.x, waveform_height)
            ), egui::Sense::hover());

            // Create rect for time markers
            let time_markers_rect = Rect::from_min_size(
                Pos2::new(waveform_rect.rect.left(), waveform_rect.rect.bottom()),
                Vec2::new(available_size.x, 20.0)
            );

            // Handle scroll input
            self.handle_scroll_input(ui, waveform_rect.rect);

            // Draw the waveform and time markers
            self.draw_waveform(ui, waveform_rect.rect);
            self.draw_chunk_borders(ui, waveform_rect.rect);
            self.draw_time_markers(ui, time_markers_rect);
        });

    }
}

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        ..Default::default()
    };

    eframe::run_native(
        "Audio Visualizer",
        native_options,
        Box::new(|cc| Ok(Box::new(AudioVisualizer::new(cc))))
    )
}