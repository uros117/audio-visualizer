use std::collections::VecDeque;
use egui::{Color32, Pos2, Rect, Stroke, StrokeKind, Ui, Vec2, Response};
use fixed::types::I48F16;

/// A widget for visualizing audio waveforms with time markers and scrolling/zooming capabilities.
pub struct WaveformWidget {
    // Audio data
    chunks: VecDeque<Vec<f32>>,
    chunk_start: u64,
    chunk_size: u64,
    total_samples: u64,
    
    // Audio metadata
    sample_rate: u32,
    num_channels: u16,
    
    // View settings
    visible_time_window: I48F16,
    zoom_sensitivity: I48F16,
    scroll_offset: I48F16,
    scroll_sensitivity: I48F16,
    
    // Visual settings
    waveform_color: Color32,
    background_color: Color32,
    chunk_alt_color: Color32,
    chunk_border_color: Color32,
    loading_color: Color32,
    centerline_color: Color32,
    
    // Interaction state
    dragging: bool,
    drag_start_pos: Option<Pos2>,
    drag_start_offset: I48F16,
}

impl Default for WaveformWidget {
    fn default() -> Self {
        Self {
            chunks: VecDeque::new(),
            chunk_start: 0,
            chunk_size: 1024,
            total_samples: 0,
            sample_rate: 44100,
            num_channels: 2,
            visible_time_window: I48F16::from_num(20.0), // 20 seconds
            zoom_sensitivity: I48F16::from_num(10.0),
            scroll_offset: I48F16::from_num(0.0),
            scroll_sensitivity: I48F16::from_num(0.2),
            waveform_color: Color32::from_rgb(100, 200, 100),
            background_color: Color32::from_gray(20),
            chunk_alt_color: Color32::from_rgb(30, 30, 40),
            chunk_border_color: Color32::LIGHT_GREEN,
            loading_color: Color32::from_rgba_premultiplied(150, 50, 50, 100),
            centerline_color: Color32::from_gray(40),
            dragging: false,
            drag_start_pos: None,
            drag_start_offset: I48F16::from_num(0.0),
        }
    }
}

impl WaveformWidget {
    /// Create a new waveform widget
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set audio data chunks
    pub fn set_chunks(&mut self, chunks: VecDeque<Vec<f32>>, chunk_start: u64, chunk_size: u64) {
        self.chunks = chunks;
        self.chunk_start = chunk_start;
        self.chunk_size = chunk_size;
    }
    
    /// Set audio metadata
    pub fn set_audio_info(&mut self, sample_rate: u32, num_channels: u16, total_samples: u64) {
        self.sample_rate = sample_rate;
        self.num_channels = num_channels;
        self.total_samples = total_samples;
    }
    
    /// Set the visible time window (zoom level)
    pub fn set_visible_time_window(&mut self, window: f64) -> &mut Self {
        self.visible_time_window = I48F16::from_num(window);
        self
    }
    
    /// Set the scroll offset (position in time)
    pub fn set_scroll_offset(&mut self, offset: f64) -> &mut Self {
        self.scroll_offset = I48F16::from_num(offset);
        self
    }
    
    /// Get the current scroll offset as a fixed-point value
    pub fn scroll_offset(&self) -> I48F16 {
        self.scroll_offset
    }
    
    /// Get the current visible time window as a fixed-point value
    pub fn visible_time_window(&self) -> I48F16 {
        self.visible_time_window
    }
    
    /// Set custom colors
    pub fn with_colors(
        &mut self, 
        waveform: Color32, 
        background: Color32, 
        chunk_alt: Color32,
        chunk_border: Color32
    ) -> &mut Self {
        self.waveform_color = waveform;
        self.background_color = background;
        self.chunk_alt_color = chunk_alt;
        self.chunk_border_color = chunk_border;
        self
    }
    
    /// Get a sample at a specific position
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
    
    // Helper functions for coordinate conversion
    fn scale_width(&self, time_duration: I48F16, rect: Rect) -> f32 {
        rect.width() * (time_duration / self.visible_time_window).to_num::<f32>()
    }

    fn time_to_screen_x(&self, time: I48F16, rect: Rect) -> f32 {
        rect.left() + rect.width() * ((time - self.scroll_offset) / self.visible_time_window).to_num::<f32>()
    }

    fn get_chunk_duration(&self) -> I48F16 {
        I48F16::from_num(self.chunk_size) / I48F16::from_num(self.sample_rate * self.num_channels as u32)
    }
    
    /// Draw chunk borders to show data organization
    fn draw_chunk_borders(&self, ui: &mut Ui, rect: Rect) {
        if self.sample_rate == 0 {
            return;
        }

        // Fixed-point comparison
        if self.visible_time_window >= I48F16::from_num(3.0) { return; }

        let painter = ui.painter();

        // Calculate duration of each chunk in seconds
        let chunk_duration = self.get_chunk_duration();

        // Calculate visible chunks using fixed-point math
        let visible_chunk_window = (self.visible_time_window / chunk_duration).ceil();
        let number_of_visible_chunks = visible_chunk_window.to_num::<u32>();

        let start_idx = (self.scroll_offset / chunk_duration).floor();

        // Draw time markers
        let mut current_x_pos: f32 = self.time_to_screen_x(start_idx * chunk_duration, rect);
        let mut current_index: u32 = start_idx.to_num();
        let chunk_width: f32 = self.scale_width(chunk_duration, rect);
        
        for _ in 0..number_of_visible_chunks {
            let chunk_rect = Rect::from_min_max(
                Pos2::new(current_x_pos, rect.top()),
                Pos2::new(current_x_pos + chunk_width, rect.bottom())
            );

            painter.rect_stroke(
                chunk_rect,
                1.0,
                Stroke::new(1.0, self.chunk_border_color),
                StrokeKind::Inside,
            );

            painter.text(
                chunk_rect.center(),
                egui::Align2::CENTER_CENTER,
                current_index,
                egui::FontId::monospace(10.0),
                self.chunk_border_color
            );

            current_x_pos += chunk_width;
            current_index += 1;
        }
    }
    
    /// Draw time markers
    fn draw_time_markers(&self, ui: &mut Ui, rect: Rect) {
        if self.sample_rate == 0 {
            return;
        }

        let painter = ui.painter();

        // Drawing parameters
        let height = 20.0; // Height for time markers area
        let y_pos = rect.bottom() - height;

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
        let first_marker_time: I48F16 = (self.scroll_offset / interval).ceil() * interval;
        let first_marker = self.time_to_screen_x(first_marker_time, rect);
        let last_marker_pos = self.time_to_screen_x(self.scroll_offset + self.visible_time_window, rect);

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
            // Draw marker line
            painter.line_segment(
                [Pos2::new(current_x_pos, y_pos), Pos2::new(current_x_pos, y_pos + 5.0)],
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
                Pos2::new(current_x_pos, y_pos + 8.0),
                egui::Align2::CENTER_TOP,
                time_text,
                egui::FontId::monospace(10.0),
                Color32::from_gray(200)
            );

            current_x_pos += screen_space_interval;
            current_time += interval;
        }
    }
    
    /// Handle input events (scrolling, zooming)
    fn handle_input(&mut self, ui: &Ui, response: &Response, rect: Rect) -> bool {
        let mut changed = false;
        
        // Handle scroll wheel for navigation/zoom
        if response.hovered() {
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
                        changed = true;
                    }
                } else if scroll_delta.y != 0.0 {
                    // Horizontal scrolling (time navigation) with fixed-point precision
                    let delta = I48F16::from_num(scroll_delta.y) / I48F16::from_num(120.0);
                    let scroll_amount = self.scroll_sensitivity * self.visible_time_window * delta;

                    let new_scroll = self.scroll_offset + scroll_amount;

                    // Update scroll offset with precise fixed-point addition
                    self.scroll_offset = if new_scroll >= I48F16::ZERO { new_scroll } else { I48F16::ZERO };
                    changed = true;
                }
            });
        }
        
        // Handle dragging for seeking
        let pointer_pos = ui.input(|i| i.pointer.interact_pos());
        let mouse_pressed = ui.input(|i| i.pointer.primary_pressed());
        let mouse_released = ui.input(|i| i.pointer.primary_released());
        
        if mouse_pressed && response.hovered() && !self.dragging {
            // Start dragging
            self.dragging = true;
            self.drag_start_pos = pointer_pos;
            self.drag_start_offset = self.scroll_offset;
            changed = true;
        } else if mouse_released && self.dragging {
            // Stop dragging
            self.dragging = false;
            changed = true;
        } else if self.dragging {
            // Handle dragging
            if let (Some(drag_start), Some(current_pos)) = (self.drag_start_pos, pointer_pos) {
                // Calculate the drag delta in screen pixels
                let delta_pixels = current_pos.x - drag_start.x;
                
                // Convert to time delta
                let time_per_pixel = self.visible_time_window / I48F16::from_num(rect.width());
                let time_delta = time_per_pixel * I48F16::from_num(-delta_pixels);
                
                // Update scroll position
                let new_offset = self.drag_start_offset + time_delta;
                if new_offset >= I48F16::ZERO {
                    self.scroll_offset = new_offset;
                    changed = true;
                } else {
                    self.scroll_offset = I48F16::ZERO;
                    changed = true;
                }
            }
        }
        
        changed
    }
    
    /// Draw the actual waveform data
    fn draw_waveform(&self, ui: &mut Ui, rect: Rect) {
        if self.chunks.is_empty() || self.sample_rate == 0 {
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

        // Draw waveform
        let height = rect.height();
        let mid_y = rect.center().y;

        // Draw background
        painter.rect_filled(rect, 0.0, self.background_color);

        // Draw centerline
        painter.line_segment(
            [Pos2::new(rect.left(), mid_y), Pos2::new(rect.right(), mid_y)],
            Stroke::new(1.0, self.centerline_color)
        );

        // Visualize chunk boundaries (alternating background colors)
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
                    self.chunk_alt_color
                );
            }
        }

        // Draw the actual waveform data
        let mut last_x = rect.left();
        let mut last_max_y = mid_y;
        let mut last_min_y = mid_y;
        let channel_count = self.num_channels as usize;

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
                    Stroke::new(1.0, self.waveform_color)
                );

                // Connect with previous point for smoother waveform
                if x > 0 {
                    painter.line_segment(
                        [Pos2::new(last_x, last_max_y), Pos2::new(rect.left() + x as f32, max_y)],
                        Stroke::new(1.0, self.waveform_color)
                    );

                    painter.line_segment(
                        [Pos2::new(last_x, last_min_y), Pos2::new(rect.left() + x as f32, min_y)],
                        Stroke::new(1.0, self.waveform_color)
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
                        self.loading_color
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

/// Implementation of the Widget trait for WaveformWidget
impl egui::Widget for &mut WaveformWidget {
    fn ui(self, ui: &mut Ui) -> Response {
        // Determine size for the widget
        let available_size = ui.available_size();
        let waveform_height = available_size.y - 20.0; // Reserve space for time markers
        
        // Allocate space for the waveform area
        let (waveform_rect, mut waveform_response) = ui.allocate_exact_size(
            Vec2::new(available_size.x, waveform_height),
            egui::Sense::click_and_drag()
        );
        
        // Allocate space for the time markers
        let time_markers_rect = Rect::from_min_size(
            Pos2::new(waveform_rect.left(), waveform_rect.bottom()),
            Vec2::new(available_size.x, 20.0)
        );
        
        let (_time_rect, _time_response) = ui.allocate_exact_size(
            Vec2::new(available_size.x, 20.0),
            egui::Sense::hover()
        );
        
        // Handle interaction (scrolling, zooming, etc.)
        let changed = self.handle_input(ui, &waveform_response, waveform_rect);
        
        // Paint if visible
        if ui.is_rect_visible(waveform_rect) || ui.is_rect_visible(time_markers_rect) {
            // Draw the waveform and markers
            self.draw_waveform(ui, waveform_rect);
            self.draw_chunk_borders(ui, waveform_rect);
            self.draw_time_markers(ui, time_markers_rect);
        }
        
        // Report if the widget state changed
        if changed {
            waveform_response.mark_changed();
            waveform_response
        } else {
            waveform_response
        }
    }
}

/// The builder pattern implementation for creating a waveform widget
pub fn waveform_widget(chunks: VecDeque<Vec<f32>>, chunk_start: u64, chunk_size: u64) -> WaveformWidget {
    let mut widget = WaveformWidget::new();
    widget.set_chunks(chunks, chunk_start, chunk_size);
    widget
}