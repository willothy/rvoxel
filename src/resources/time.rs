use bevy_ecs::prelude::*;

// Time - tracks delta time between frames
#[derive(Resource)]
pub struct Time {
    pub delta: f32,
    pub elapsed: f32,
    last_update: std::time::Instant,
}

impl Default for Time {
    fn default() -> Self {
        Self {
            delta: 0.0,
            elapsed: 0.0,
            last_update: std::time::Instant::now(),
        }
    }
}

impl Time {
    pub fn update(&mut self) {
        let now = std::time::Instant::now();
        self.delta = now.duration_since(self.last_update).as_secs_f32();
        self.elapsed += self.delta;
        self.last_update = now;
    }
}
