mod audio_loader;
mod waveform_widget;
mod audio_range;
mod audio_storage;

use std::sync::RwLock;
use std::{collections::VecDeque, sync::Arc};
use std::path::PathBuf;

use audio_storage::AudioBufferManager;
use eframe::egui;
use egui::{Vec2, ViewportBuilder};
use rfd::FileDialog;

use audio_loader::{AudioLoader, LoaderResponse};
use waveform_widget::WaveformWidget;

const FFT_SIZE: usize = 2048;
const BUFFER_SIZE: usize = 1024; // 1MB chunks for loading

struct AudioVisualizer {
    file_path: Option<PathBuf>,
    loaded_percentage: f32,
    
    // Audio state
    chunk_start: u64,           // Integer sample position
    chunk_size: u64,            // Integer chunk size
    total_samples: u64,         // Integer total sample count
    sample_rate: u32,
    num_channels: u16,
    

    // Audio storage
    buffer_manager: Option<audio_storage::AudioBufferManager>,
    
    // Loading state
    is_loading: bool,
    
    // Loader
    audio_loader: Arc<RwLock<AudioLoader>>,
    
    // Waveform widget
    waveform_widget: WaveformWidget,

    // Visible range tracking
    visible_range: audio_range::AudioRange,
    
    // Pre-buffer range for playback (when playing)
    playing: bool,
    play_position: u64,
    playback_range: Option<audio_range::AudioRange>,
}

impl Default for AudioVisualizer {
    fn default() -> Self {
        Self {
            file_path: None,
            loaded_percentage: 0.0,
            chunk_start: 0,
            chunk_size: BUFFER_SIZE as u64,
            total_samples: 0,
            sample_rate: 44100,
            num_channels: 2,
            is_loading: false,
            audio_loader: Arc::new(RwLock::new(AudioLoader::new())),
            waveform_widget: WaveformWidget::new(),
            buffer_manager: None,
            visible_range: audio_range::AudioRange { start: 0, end: 10, priority: 0 },
            playing: false,
            play_position: 0,
            playback_range: None,
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
            {
                let mut loader = self.audio_loader.write().unwrap();
                if loader.is_running() {
                    loader.stop();
                }
            }
            
            // Clear existing chunks
            if let Some(buff) = &mut self.buffer_manager {
                buff.clear();
            }

            self.loaded_percentage = 0.0;
            self.chunk_start = 0;

            // Reset the playback
            self.playing = false;
            self.play_position = 0;
            self.playback_range = None;
            
            // Start the loader
            {
                let mut loader = self.audio_loader.write().unwrap();
                match loader.start(path.clone(), BUFFER_SIZE as u64) {
                    Ok(()) => {
                        println!("Loader started successfully");
                        self.file_path = Some(path);
                        
                        // Get file info from loader
                        let (sample_rate, num_channels, total_samples) = loader.get_info();
                        self.sample_rate = sample_rate;
                        self.num_channels = num_channels;
                        self.total_samples = total_samples;
                        
                        // Update the waveform widget with the audio info
                        self.waveform_widget.set_audio_info(sample_rate, num_channels, total_samples);
                        
                        // Create the buffer manager
                        let max_samples = 60 * sample_rate as u64 * num_channels as u64; // Store up to 60 seconds
                        self.buffer_manager = Some(audio_storage::AudioBufferManager::new(
                            sample_rate,
                            num_channels,
                            max_samples,
                            self.audio_loader.clone(),
                            BUFFER_SIZE as u64
                        ));
                        
                        // Initialize the visible range
                        let visible_samples = (self.waveform_widget.visible_time_window().to_num::<f64>() 
                            * sample_rate as f64 * num_channels as f64) as u64;
                        self.visible_range = audio_range::AudioRange::new(0, visible_samples, 10);
                    },
                    Err(e) => {
                        println!("Error starting loader: {}", e);
                    }
                }
            }
        }
    }

    // Request chunks based on current view
    fn check_buffer_needs(&mut self) {
        if self.buffer_manager.is_none() {
            return;
        }
        
        // Update buffer manager to receive any loaded chunks
        if let Some(buffer_manager) = &mut self.buffer_manager {
            let _ = buffer_manager.update();
        }
        
        // Get current scroll position and visible window from widget
        let scroll_offset = self.waveform_widget.scroll_offset();
        let visible_time_window = self.waveform_widget.visible_time_window();
        
        // Convert visible window to samples
        let visible_samples_start = (scroll_offset.to_num::<f64>() * self.sample_rate as f64 * self.num_channels as f64) as u64;
        let visible_samples_end = ((scroll_offset + visible_time_window).to_num::<f64>() * self.sample_rate as f64 * self.num_channels as f64) as u64;
        
        // Update the visible range
        self.visible_range = audio_range::AudioRange::new(visible_samples_start, visible_samples_end, 10);
        
        // Calculate a pre-buffer range for playback if playing
        if self.playing {
            let buffer_duration = 5; // 5 seconds ahead
            let play_end = self.play_position + (buffer_duration * self.sample_rate as u64 * self.num_channels as u64);
            self.playback_range = Some(audio_range::AudioRange::new(self.play_position, play_end, 5));
        } else {
            self.playback_range = None;
        }
        
        // Create a list of ranges to ensure are loaded
        let mut ranges_to_load = Vec::new();
        
        // Always load the visible range
        ranges_to_load.push((self.visible_range.start, self.visible_range.end));
        
        // Add playback range if playing
        if let Some(ref range) = self.playback_range {
            ranges_to_load.push((range.start, range.end));
        }
        
        // Ensure all ranges are loaded
        if let Some(buffer_manager) = &mut self.buffer_manager {
            if let Err(e) = buffer_manager.ensure_loaded(&ranges_to_load) {
                println!("Error ensuring ranges are loaded: {}", e);
            }
            
            // Update loading status
            self.is_loading = buffer_manager.pending_count() > 0;
            
            // Update loaded percentage for the visible range
            if let Ok(proportion) = buffer_manager.storage_handle().loaded_proportion(
                self.visible_range.start, self.visible_range.end
            ) {
                self.loaded_percentage = proportion * 100.0;
            }
        }
    }

    fn update_waveform_data(&mut self) {
        if let Some(buffer_manager) = &mut self.buffer_manager {
            // Get the visible range
            let visible_start = self.visible_range.start;
            let visible_len = self.visible_range.len();
            
            // Get samples for the visible range
            match buffer_manager.storage_handle().get_samples(visible_start, visible_len) {
                Ok(samples) => {
                    // Check if we need to create fake chunks for the waveform widget
                    // or if we need to modify the waveform widget to accept a single buffer
                    
                    // Option 1: Create fake chunks
                    let chunk_size = BUFFER_SIZE as u64;
                    let mut chunks = VecDeque::new();
                    
                    for i in 0..(visible_len + chunk_size - 1) / chunk_size {
                        let start_idx = (i * chunk_size) as usize;
                        let end_idx = ((i + 1) * chunk_size).min(visible_len) as usize;
                        
                        if start_idx < samples.len() {
                            let chunk_samples = if end_idx <= samples.len() {
                                samples[start_idx..end_idx].to_vec()
                            } else {
                                // Pad with zeros if needed
                                let mut chunk = samples[start_idx..].to_vec();
                                chunk.resize(end_idx - start_idx, 0.0);
                                chunk
                            };
                            
                            chunks.push_back(chunk_samples);
                        }
                    }
                    
                    // Set the chunks in the waveform widget
                    self.waveform_widget.set_chunks(chunks, visible_start, chunk_size);
                    
                    // Option 2: Modify waveform_widget.rs to accept a single buffer
                    // This would be more efficient but requires changing the widget
                },
                Err(e) => {
                    println!("Error getting samples for waveform: {}", e);
                }
            }
        }
    }
        
}

impl eframe::App for AudioVisualizer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        
        // Check if we need to load more chunks based on the visible area
        self.check_buffer_needs();

        // Update waveform data
        self.update_waveform_data();

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
                    ui.label(format!("Position: {} samples", self.chunk_start));
                    ui.separator();
                    // Display fixed-point values with high precision
                    ui.label(format!("Time: {:.6} s", self.waveform_widget.scroll_offset()));

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        // TODO: add some buttons
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