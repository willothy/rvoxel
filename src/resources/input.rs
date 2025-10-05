use bevy_ecs::prelude::*;
use std::collections::HashSet;
use winit::keyboard::KeyCode;

// Input state - tracks keyboard/mouse
#[derive(Resource, Default)]
pub struct InputState {
    pub keys_pressed: HashSet<KeyCode>,
    pub mouse_delta: (f32, f32),
    pub cursor_locked: bool,
}

impl InputState {
    pub fn is_key_pressed(&self, key: KeyCode) -> bool {
        self.keys_pressed.contains(&key)
    }

    pub fn reset_frame(&mut self) {
        self.mouse_delta = (0.0, 0.0);
    }
}
