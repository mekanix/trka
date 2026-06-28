use avian3d::prelude::*;
use bevy::prelude::*;
use bevy::window::{PresentMode, Window, WindowPlugin};
use maolan_engine::{
    client::Client,
    kind::Kind,
    message::{Action, Message as EngineMessage},
};
use std::{
    f32::consts::TAU,
    io::Write,
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
    time::Duration,
};
use tracing_subscriber::{
    fmt::{writer::MakeWriterExt, Layer as FmtLayer},
    prelude::*,
};

const AUDIO_SAMPLE_RATE: i32 = 48_000;
const AUDIO_DEVICE: &str = "/dev/dsp5";
const ENGINE_HUM_WAV: &str = "engine_hum.wav";

fn main() {
    init_logging();

    let speed = Arc::new(AtomicU32::new(0));
    let speed_for_audio = speed.clone();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to create tokio runtime");
        rt.block_on(run_audio_engine(speed_for_audio));
    });

    App::new()
        .add_plugins(
            DefaultPlugins
                .build()
                .disable::<bevy::log::LogPlugin>()
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "trka".into(),
                        present_mode: PresentMode::AutoVsync,
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_plugins(PhysicsPlugins::default())
        .insert_resource(EngineSpeed(speed))
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (vehicle_controls, camera_follow, update_engine_speed),
        )
        .run();
}

fn init_logging() {
    if let Some(level) = parse_log_level() {
        let layer = FmtLayer::new().with_writer(std::io::stderr.with_max_level(level));
        tracing_subscriber::registry().with(layer).init();
    }
}

fn parse_log_level() -> Option<tracing::Level> {
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == "--log-level") {
        if pos + 1 < args.len() {
            match args[pos + 1].as_str() {
                "none" => None,
                "info" => Some(tracing::Level::INFO),
                "warning" => Some(tracing::Level::WARN),
                "error" => Some(tracing::Level::ERROR),
                "debug" => Some(tracing::Level::DEBUG),
                _other => None,
            }
        } else {
            None
        }
    } else {
        None
    }
}

#[derive(Resource)]
struct EngineSpeed(Arc<AtomicU32>);

#[derive(Component)]
struct Vehicle;

#[derive(Component)]
struct VehicleController {
    throttle: f32,
    steering: f32,
}

impl Default for VehicleController {
    fn default() -> Self {
        Self {
            throttle: 0.0,
            steering: 0.0,
        }
    }
}

const THROTTLE_RATE: f32 = 8.0;
const BRAKE_RATE: f32 = 12.0;
const COAST_RATE: f32 = 4.0;
const MAX_SPEED: f32 = 35.0;
const MAX_REVERSE_SPEED: f32 = -10.0;
const STEER_RATE: f32 = 2.0;
const STEER_RETURN_RATE: f32 = 3.0;
const MAX_YAW_RATE: f32 = 1.5;
const SPEED_CONTROL_GAIN: f32 = 2.0;
const TURN_GAIN: f32 = 80.0;
const LATERAL_GRIP: f32 = 6.0;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 12.0, 25.0).looking_at(Vec3::new(20.0, 0.0, 0.0), Vec3::Y),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 5000.0,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.6, 0.4, 0.0)),
    ));

    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(300.0, 300.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.15, 0.45, 0.15),
            ..default()
        })),
        RigidBody::Static,
        Collider::cuboid(300.0, 0.5, 300.0),
        Transform::from_xyz(0.0, -0.25, 0.0),
    ));

    spawn_elliptical_fence(&mut commands, &mut meshes, &mut materials);

    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0).mesh())),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.85, 0.1, 0.1),
            ..default()
        })),
        Transform::from_xyz(30.0, 0.75, 0.0),
        Vehicle,
        VehicleController::default(),
        RigidBody::Dynamic,
        Collider::cuboid(1.0, 1.0, 1.0),
        Mass(12.0),
        Friction::new(0.7),
        Restitution::new(0.0),
        LinearDamping(0.4),
        AngularDamping(0.6),
        LockedAxes::new()
            .lock_translation_y()
            .lock_rotation_x()
            .lock_rotation_z(),
    ));
}

fn spawn_elliptical_fence(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
) {
    let a = 30.0;
    let b = 20.0;
    let track_width = 7.0;
    let fence_height = 1.0;
    let fence_thickness = 0.5;
    let segments = 64;

    let fence_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.85, 0.85, 0.85),
        ..default()
    });
    let fence_mesh = meshes.add(Cuboid::new(1.0, 1.0, 1.0).mesh());

    let a_outer = a + track_width * 0.5;
    let b_outer = b + track_width * 0.5;
    let a_inner = a - track_width * 0.5;
    let b_inner = b - track_width * 0.5;

    for (a_ring, b_ring) in [(a_outer, b_outer), (a_inner, b_inner)] {
        for i in 0..segments {
            let t0 = TAU * i as f32 / segments as f32;
            let t1 = TAU * (i + 1) as f32 / segments as f32;
            let t = (t0 + t1) * 0.5;

            let x = a_ring * t.cos();
            let z = b_ring * t.sin();
            let dx = -a_ring * t.sin();
            let dz = b_ring * t.cos();
            let angle = dx.atan2(dz);

            let arc = ((dx * dx + dz * dz).sqrt()) * (t1 - t0);
            let depth = arc + 0.15;

            commands.spawn((
                Mesh3d(fence_mesh.clone()),
                MeshMaterial3d(fence_material.clone()),
                Transform {
                    translation: Vec3::new(x, fence_height * 0.5, z),
                    rotation: Quat::from_rotation_y(angle),
                    scale: Vec3::new(fence_thickness, fence_height, depth),
                },
                RigidBody::Static,
                Collider::cuboid(fence_thickness, fence_height, depth),
                Friction::new(0.9),
            ));
        }
    }
}

fn vehicle_controls(
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut query: Query<(&Transform, Forces, &Mass, &mut VehicleController), With<Vehicle>>,
) {
    let Some((transform, mut forces, mass, mut controller)) = query.iter_mut().next() else {
        return;
    };

    let dt = time.delta_secs();
    let forward = transform.forward().as_vec3();
    let speed = forces.linear_velocity().dot(forward);
    let yaw_rate = forces.angular_velocity().y;

    let accelerating = keyboard.pressed(KeyCode::KeyW) || keyboard.pressed(KeyCode::ArrowUp);
    let braking = keyboard.pressed(KeyCode::KeyS) || keyboard.pressed(KeyCode::ArrowDown);
    let left = keyboard.pressed(KeyCode::KeyA) || keyboard.pressed(KeyCode::ArrowLeft);
    let right = keyboard.pressed(KeyCode::KeyD) || keyboard.pressed(KeyCode::ArrowRight);

    // Throttle builds up and down with inertia.
    if accelerating {
        controller.throttle += THROTTLE_RATE * dt;
    } else if braking {
        controller.throttle -= BRAKE_RATE * dt;
    } else {
        if controller.throttle > 0.0 {
            controller.throttle = (controller.throttle - COAST_RATE * dt).max(0.0);
        } else if controller.throttle < 0.0 {
            controller.throttle = (controller.throttle + COAST_RATE * dt).min(0.0);
        }
    }
    controller.throttle = controller.throttle.clamp(MAX_REVERSE_SPEED, MAX_SPEED);

    // Steering input builds up and returns to center with inertia.
    let steer_input = (right as i32 - left as i32) as f32;
    if steer_input != 0.0 {
        controller.steering += steer_input * STEER_RATE * dt;
    } else {
        if controller.steering > 0.0 {
            controller.steering = (controller.steering - STEER_RETURN_RATE * dt).max(0.0);
        } else if controller.steering < 0.0 {
            controller.steering = (controller.steering + STEER_RETURN_RATE * dt).min(0.0);
        }
    }
    controller.steering = controller.steering.clamp(-1.0, 1.0);

    // Apply engine/brake force to reach the target speed.
    let speed_error = controller.throttle - speed;
    forces.apply_force(forward * speed_error * SPEED_CONTROL_GAIN * mass.0);

    // Lateral grip pulls the velocity to follow the vehicle's heading,
    // so steering actually changes the direction of travel instead of just
    // spinning the cube.
    let lateral_velocity = forces.linear_velocity() - forward * speed;
    forces.apply_force(-lateral_velocity * LATERAL_GRIP * mass.0);

    // The yaw target is reached through torque, so the rotation has inertia.
    // The torque direction is flipped when reversing so A/D always map to the
    // left/right side of the screen as seen from the chase camera.
    let drive_direction = if speed < 0.0 { -1.0 } else { 1.0 };
    let target_yaw_rate = -controller.steering * MAX_YAW_RATE * drive_direction;
    let yaw_error = target_yaw_rate - yaw_rate;
    forces.apply_torque(Vec3::Y * yaw_error * TURN_GAIN);
}

fn camera_follow(
    vehicle: Query<&Transform, With<Vehicle>>,
    mut camera: Query<&mut Transform, (With<Camera3d>, Without<Vehicle>)>,
) {
    let Some(vehicle_transform) = vehicle.iter().next() else {
        return;
    };
    let Some(mut camera_transform) = camera.iter_mut().next() else {
        return;
    };

    let back = -vehicle_transform.forward().as_vec3();
    let target_pos = vehicle_transform.translation + back * 12.0 + Vec3::Y * 6.0;
    camera_transform.translation = camera_transform.translation.lerp(target_pos, 0.1);
    camera_transform.look_at(vehicle_transform.translation + Vec3::Y * 0.5, Vec3::Y);
}

fn update_engine_speed(
    vehicle: Query<&LinearVelocity, With<Vehicle>>,
    engine_speed: Res<EngineSpeed>,
) {
    if let Some(velocity) = vehicle.iter().next() {
        let speed = velocity.0.length();
        engine_speed.0.store(speed.to_bits(), Ordering::Relaxed);
    }
}

async fn run_audio_engine(speed: Arc<AtomicU32>) {
    if let Err(err) = generate_engine_wav(ENGINE_HUM_WAV, 1.0, AUDIO_SAMPLE_RATE as u32) {
        eprintln!("Failed to generate engine sound: {err}");
        return;
    }

    let client = Client::default();
    let mut rx = client.subscribe().await;

    let _ = client
        .send(EngineMessage::Request(Action::OpenAudioDevice {
            device: AUDIO_DEVICE.to_string(),
            input_device: None,
            sample_rate_hz: AUDIO_SAMPLE_RATE,
            bits: 32,
            exclusive: false,
            period_frames: 2048,
            realtime_frames: 128,
            low_watermark_frames: 512,
            nperiods: 2,
            sync_mode: false,
            hybrid_enabled: false,
            actual_period_frames: 0,
            input_channels: 0,
            output_channels: 0,
            bytes_per_frame: 0,
        }))
        .await;

    let hw_ready = wait_for_message(&mut rx, |msg| {
        matches!(msg, EngineMessage::Response(Ok(Action::HWInfo { .. })))
    })
    .await;

    if !hw_ready {
        eprintln!("Audio device '{AUDIO_DEVICE}' not ready; continuing without engine audio");
        return;
    }

    let track_name = "engine".to_string();
    let samples = AUDIO_SAMPLE_RATE as usize;

    let _ = client
        .send(EngineMessage::Request(Action::AddTrack {
            name: track_name.clone(),
            audio_ins: 2,
            audio_outs: 2,
            midi_ins: 0,
            midi_outs: 0,
        }))
        .await;

    let _ = client
        .send(EngineMessage::Request(Action::AddClip {
            name: ENGINE_HUM_WAV.to_string(),
            track_name: track_name.clone(),
            start: 0,
            length: samples,
            offset: 0,
            input_channel: 0,
            muted: false,
            peaks_file: None,
            kind: Kind::Audio,
            fade_enabled: true,
            fade_in_samples: 240,
            fade_out_samples: 240,
            source_name: Some(ENGINE_HUM_WAV.to_string()),
            source_offset: None,
            source_length: None,
            preview_name: None,
            pitch_correction_points: Vec::new(),
            pitch_correction_frame_likeness: None,
            pitch_correction_inertia_ms: None,
            pitch_correction_formant_compensation: None,
            plugin_graph_json: None,
        }))
        .await;

    let _ = client
        .send(EngineMessage::Request(Action::SetLoopEnabled(true)))
        .await;
    let _ = client
        .send(EngineMessage::Request(Action::SetLoopRange(Some((
            0, samples,
        )))))
        .await;
    let _ = client
        .send(EngineMessage::Request(Action::SetClipPlaybackEnabled(true)))
        .await;
    let _ = client.send(EngineMessage::Request(Action::Play)).await;

    let mut interval = tokio::time::interval(Duration::from_millis(100));
    loop {
        interval.tick().await;
        let speed_f32 = f32::from_bits(speed.load(Ordering::Relaxed));
        let gain = (0.1 + (speed_f32 / 30.0).clamp(0.0, 1.0) * 0.9).clamp(0.0, 1.0);
        let _ = client
            .send(EngineMessage::Request(Action::TrackLevel(
                track_name.clone(),
                gain,
            )))
            .await;
    }
}

async fn wait_for_message<F>(
    rx: &mut tokio::sync::mpsc::Receiver<EngineMessage>,
    mut predicate: F,
) -> bool
where
    F: FnMut(&EngineMessage) -> bool,
{
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    loop {
        let timeout = deadline.saturating_duration_since(tokio::time::Instant::now());
        if timeout.is_zero() {
            return false;
        }
        match tokio::time::timeout(timeout, rx.recv()).await {
            Ok(Some(msg)) => {
                if predicate(&msg) {
                    return true;
                }
            }
            Ok(None) => return false,
            Err(_) => return false,
        }
    }
}

fn generate_engine_wav(path: &str, duration: f32, sample_rate: u32) -> std::io::Result<()> {
    let total_samples = (duration * sample_rate as f32) as usize;
    let mut samples = Vec::with_capacity(total_samples);

    for i in 0..total_samples {
        let t = i as f32 / sample_rate as f32;
        let mut sample = 0.0_f32;
        sample += (t * 80.0 * TAU).sin() * 0.45;
        sample += (t * 160.0 * TAU).sin() * 0.25;
        sample += (t * 240.0 * TAU).sin() * 0.12;
        sample += (((t * 40.0 * TAU).sin() + 1.0) * 0.5 - 0.5) * 0.08;

        let fade_in = (i as f32 / 240.0_f32).min(1.0);
        let fade_out = ((total_samples - i) as f32 / 240.0_f32).min(1.0);
        sample *= fade_in * fade_out;

        samples.push(sample);
    }

    let mut file = std::fs::File::create(path)?;
    let channels: u16 = 1;
    let bits: u16 = 32;
    let audio_format: u16 = 3;
    let byte_rate = sample_rate * channels as u32 * (bits as u32 / 8);
    let block_align = channels * (bits / 8);
    let data_size = (samples.len() * 4) as u32;
    let riff_size = 36 + data_size;

    file.write_all(b"RIFF")?;
    file.write_all(&riff_size.to_le_bytes())?;
    file.write_all(b"WAVE")?;

    file.write_all(b"fmt ")?;
    file.write_all(&16_u32.to_le_bytes())?;
    file.write_all(&audio_format.to_le_bytes())?;
    file.write_all(&channels.to_le_bytes())?;
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&block_align.to_le_bytes())?;
    file.write_all(&bits.to_le_bytes())?;

    file.write_all(b"data")?;
    file.write_all(&data_size.to_le_bytes())?;
    for sample in samples {
        file.write_all(&sample.to_le_bytes())?;
    }

    Ok(())
}
