mod audio_loader;
mod waveform_widget;

use std::collections::VecDeque;
use std::path::PathBuf;

use eframe::egui;
use egui::{Vec2, ViewportBuilder};
use rfd::FileDialog;

use audio_loader::{AudioLoader, LoaderResponse};
use waveform_widget::WaveformWidget;

const FFT_SIZE: usize = 2048;
const BUFFER_SIZE: usize = 1024; // 1MB chunks for loading

struct AudioVisualizer {
    file_path: Option<PathBuf>,
    spectrum: Vec<f32>,
    loaded_percentage: f32,
    
    // Audio state
    chunk_start: u64,           // Integer sample position
    chunk_size: u64,            // Integer chunk size
    total_samples: u64,         // Integer total sample count
    sample_rate: u32,
    num_channels: u16,
    
    // Chunks storage
    chunks: VecDeque<Vec<f32>>,
    max_chunks: usize,
    
    // Loading state
    is_loading: bool,
    pending_chunks: Vec<u64>,
    
    // Loader
    audio_loader: AudioLoader,
    
    // Waveform widget
    waveform_widget: WaveformWidget,
}

impl Default for AudioVisualizer {
    fn default() -> Self {
        let max_chunks = 10240;
        Self {
            file_path: None,
            spectrum: vec![0.0; FFT_SIZE / 2],
            loaded_percentage: 0.0,
            chunk_start: 0,
            chunk_size: BUFFER_SIZE as u64,
            total_samples: 0,
            sample_rate: 44100,
            num_channels: 2,
            is_loading: false,
            max_chunks,
            pending_chunks: Vec::new(),
            chunks: VecDeque::with_capacity(max_chunks),
            audio_loader: AudioLoader::new(),
            waveform_widget: WaveformWidget::new(),
        }
    }
}

impl AudioVisualizer {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self::default()
    }

    fn open_file(&mut self) {
        if let Some(path) = FileDialog::new()
            .add_filter("Audio", &["wav", "mp3", "flac", "aac", "ogg"])
            .pick_file()
        {
            println!("Opening file: {:?}", path);
            
            // Stop any existing loader
            if self.audio_loader.is_running() {
                self.audio_loader.stop();
            }
            
            // Clear existing chunks
            self.chunks.clear();
            self.spectrum = vec![0.0; FFT_SIZE / 2];
            self.loaded_percentage = 0.0;
            self.chunk_start = 0;
            self.pending_chunks.clear();
            
            // Start the loader
            match self.audio_loader.start(path.clone(), BUFFER_SIZE as u64) {
                Ok(()) => {
                    println!("Loader started successfully");
                    self.file_path = Some(path);
                    
                    // Get file info from loader
                    let (sample_rate, num_channels, total_samples) = self.audio_loader.get_info();
                    self.sample_rate = sample_rate;
                    self.num_channels = num_channels;
                    self.total_samples = total_samples;
                    
                    // Update the waveform widget with the audio info
                    self.waveform_widget.set_audio_info(sample_rate, num_channels, total_samples);
                    
                    // Request the first chunk
                    self.load_chunk_at_position(0);
                },
                Err(e) => {
                    println!("Error starting loader: {}", e);
                }
            }
        }
    }

    // Request chunks based on current view
    fn check_buffer_needs(&mut self) {
        if !self.audio_loader.is_running() {
            return; // No loader running
        }

        // Get current scroll position and visible window from widget
        let scroll_offset = self.waveform_widget.scroll_offset();
        let visible_time_window = self.waveform_widget.visible_time_window();

        // Convert visible window to samples
        let visible_samples_start = (scroll_offset.to_num::<f64>() * self.sample_rate as f64 * self.num_channels as f64) as u64;
        let visible_samples_end = ((scroll_offset + visible_time_window).to_num::<f64>() * self.sample_rate as f64 * self.num_channels as f64) as u64;

        // Check if we're near the start of our buffer
        if visible_samples_start < self.chunk_start + self.chunk_size {
            // Need to load backwards if possible
            if self.chunk_start > 0 {
                let new_start = self.chunk_start.saturating_sub(self.chunk_size);
                self.load_chunk_at_position(new_start);
            }
        }

        // Check if we're near the end of our buffer
        let buffer_end = self.chunk_start + (self.chunks.len() as u64 * self.chunk_size);
        if visible_samples_end > buffer_end.saturating_sub(self.chunk_size) {
            // Need to load forward if possible
            if buffer_end < self.total_samples {
                self.load_chunk_at_position(buffer_end);
            }
        }
    }

    fn load_chunk_at_position(&mut self, position: u64) {
        if !self.audio_loader.is_running() {
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
        
        // Check if this chunk is already pending
        if self.pending_chunks.contains(&aligned_position) {
            println!("Chunk already pending at position {}", aligned_position);
            return;
        }

        // Request the chunk from the loader
        self.audio_loader.request_chunk(aligned_position);
        self.pending_chunks.push(aligned_position);
        self.is_loading = true;
    }

    fn update_chunks(&mut self) {
        if !self.audio_loader.is_running() {
            return;
        }
        
        // Get responses from the loader
        let responses = self.audio_loader.poll_responses();
        let mut chunks_updated = false;
        
        for response in responses {
            match response {
                LoaderResponse::ChunkLoaded(position, samples) => {
                    println!("Received chunk at position: {}, size: {}", position, samples.len());
                    
                    // Determine if this should be prepended or appended
                    if position < self.chunk_start {
                        // Prepend to the front
                        println!("Prepending chunk at position {}", position);
                        self.chunks.push_front(samples);
                        self.chunk_start = position;
                        
                        // If we have too many chunks, remove from the back
                        while self.chunks.len() > self.max_chunks {
                            self.chunks.pop_back();
                        }
                        
                        chunks_updated = true;
                    } else {
                        // Check if this is an insert in the middle
                        let relative_pos = position - self.chunk_start;
                        let chunk_index = (relative_pos / self.chunk_size) as usize;
                        
                        if chunk_index < self.chunks.len() {
                            // Replace existing chunk
                            println!("Replacing chunk at index {}", chunk_index);
                            if let Some(chunk) = self.chunks.get_mut(chunk_index) {
                                *chunk = samples;
                            }
                            chunks_updated = true;
                        } else if chunk_index == self.chunks.len() {
                            // Append to the end
                            println!("Appending chunk at end");
                            self.chunks.push_back(samples);
                            
                            // If we have too many chunks, remove from the front
                            while self.chunks.len() > self.max_chunks {
                                self.chunks.pop_front();
                                self.chunk_start += self.chunk_size;
                            }
                            
                            chunks_updated = true;
                        } else {
                            println!("Warning: received out-of-order chunk at position {}", position);
                            // Could handle this better, but for now just append
                            self.chunks.push_back(samples);
                            chunks_updated = true;
                        }
                    }
                    
                    // Remove this position from pending list
                    if let Some(index) = self.pending_chunks.iter().position(|&p| p == position) {
                        self.pending_chunks.remove(index);
                    }
                    
                    // Update loaded percentage
                    let buffer_samples = self.chunks.len() as u64 * self.chunk_size;
                    self.loaded_percentage = buffer_samples as f32 / self.total_samples as f32 * 100.0;
                },
                LoaderResponse::LoadingError(e) => {
                    println!("Loading error: {}", e);
                    // Could clear pending chunks here
                },
                LoaderResponse::Progress(progress) => {
                    // Update loading progress
                    println!("Loading progress: {}%", progress);
                },
                LoaderResponse::FileInfo { sample_rate, num_channels, total_samples } => {
                    println!("Received file info: sample_rate={}, channels={}, total_samples={}",
                             sample_rate, num_channels, total_samples);
                    self.sample_rate = sample_rate;
                    self.num_channels = num_channels;
                    self.total_samples = total_samples;
                    
                    // Update the waveform widget with audio info
                    self.waveform_widget.set_audio_info(sample_rate, num_channels, total_samples);
                }
            }
        }
        
        // Update waveform widget with new chunks if needed
        if chunks_updated {
            self.waveform_widget.set_chunks(self.chunks.clone(), self.chunk_start, self.chunk_size);
        }
        
        // Update loading status
        self.is_loading = !self.pending_chunks.is_empty();
    }
}

impl eframe::App for AudioVisualizer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Update chunks from the loader
        self.update_chunks();

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
                    ui.label(format!("Time: {:.6} s", self.waveform_widget.scroll_offset()));

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Next Chunk").clicked() {
                            let next_pos = self.chunk_start + self.chunk_size * self.chunks.len() as u64;
                            if next_pos < self.total_samples {
                                self.load_chunk_at_position(next_pos);
                            }
                        }
                        if ui.button("Previous Chunk").clicked() {
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

            // Use our custom waveform widget
            ui.add(&mut self.waveform_widget);
        });
    }
}

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: ViewportBuilder {
            inner_size: Some(Vec2 { x: 1000.0, y: 400.0}),
            ..Default::default()
        },
        ..Default::default()
    };

    eframe::run_native(
        "Audio Visualizer",
        native_options,
        Box::new(|cc| Ok(Box::new(AudioVisualizer::new(cc))))
    )
}