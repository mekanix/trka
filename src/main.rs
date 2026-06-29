use avian3d::prelude::*;
use bevy::asset::RenderAssetUsages;
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy::tasks::{block_on, AsyncComputeTaskPool};
use bevy::tasks::futures_lite::future::poll_once;
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

mod osm;
mod track;
mod world;

use osm::{Building, OsmData, RoadSegment};
use track::{build_building_mesh, build_fences, build_track_mesh, save_track, Track};
use world::{fetch_natural_earth_land, fetch_urban_areas, ContinentPolygon};

const AUDIO_SAMPLE_RATE: i32 = 48_000;
const AUDIO_DEVICE: &str = "/dev/dsp5";
const ENGINE_HUM_WAV: &str = "engine_hum.wav";
const TRACK_FILE: &str = "track.json";
const DEFAULT_BBOX: &str = "40.748,-73.985,40.753,-73.978";
const OSM_SCALE: f32 = 0.3;
const WORLD_SIZE: f32 = 2000.0;

#[derive(Clone, Copy, Default, Eq, PartialEq, Debug, Hash, States)]
enum AppState {
    #[default]
    WorldMap,
    Loading,
    MapSelect,
    Drive,
}

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
        .init_state::<AppState>()
        .insert_resource(EngineSpeed(speed))
        .insert_resource(Route(Vec::new()))
        .insert_resource(SelectedRegion::default())
        .insert_resource(WorldContinents::default())
        .insert_resource(WorldUrbanAreas::default())
        .insert_resource(WorldMapDragState::default())
        .add_systems(Startup, setup)
        .add_systems(OnEnter(AppState::WorldMap), enter_world_map)
        .add_systems(OnExit(AppState::WorldMap), exit_world_map)
        .add_systems(
            Update,
            (
                world_map_zoom,
                world_map_drag,
                check_world_map_fetch,
                check_urban_fetch,
                draw_map_outlines,
            )
                .run_if(in_state(AppState::WorldMap)),
        )
        .add_systems(OnEnter(AppState::Loading), (enter_loading, start_osm_fetch))
        .add_systems(Update, check_osm_fetch.run_if(in_state(AppState::Loading)))
        .add_systems(OnEnter(AppState::MapSelect), enter_map_select)
        .add_systems(OnExit(AppState::MapSelect), exit_map_select)
        .add_systems(
            Update,
            (map_select_input, map_select_zoom, update_camera_overview)
                .run_if(in_state(AppState::MapSelect)),
        )
        .add_systems(OnEnter(AppState::Drive), enter_drive)
        .add_systems(OnExit(AppState::Drive), exit_drive)
        .add_systems(
            Update,
            (vehicle_controls, camera_follow, update_engine_speed, drive_input)
                .run_if(in_state(AppState::Drive)),
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

fn parse_bbox(region: &SelectedRegion) -> String {
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == "--bbox") {
        if pos + 1 < args.len() {
            return normalize_bbox(&args[pos + 1]).unwrap_or_else(|| args[pos + 1].clone());
        }
    }
    if let Some(bbox) = &region.bbox {
        return bbox.clone();
    }
    DEFAULT_BBOX.to_string()
}

fn parse_osm_file() -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == "--osm-file") {
        if pos + 1 < args.len() {
            return Some(args[pos + 1].clone());
        }
    }
    None
}

/// Accept Overpass format `south,west,north,east` (lat,lon,lat,lon) or
/// common OSM URL format `west,south,east,north` (lon,lat,lon,lat).
fn normalize_bbox(input: &str) -> Option<String> {
    let parts: Vec<&str> = input.split(',').collect();
    if parts.len() != 4 {
        return None;
    }
    let values: Vec<f64> = parts.iter().filter_map(|p| p.parse().ok()).collect();
    if values.len() != 4 {
        return None;
    }

    // If the first value is a longitude (outside [-90, 90]), swap to lat,lon order.
    if values[0].abs() > 90.0 {
        Some(format!(
            "{},{},{},{}",
            values[1], values[0], values[3], values[2]
        ))
    } else {
        Some(input.to_string())
    }
}

#[derive(Resource)]
struct EngineSpeed(Arc<AtomicU32>);

#[derive(Resource)]
struct Route(Vec<Vec3>);

#[derive(Resource)]
struct OsmRoads(Vec<RoadSegment>);

#[derive(Resource)]
struct OsmBuildings(Vec<Building>);

#[derive(Resource, Default)]
struct SelectedRegion {
    bbox: Option<String>,
}

#[derive(Resource, Default)]
struct WorldContinents(Vec<ContinentPolygon>);

#[derive(Resource, Default)]
struct WorldUrbanAreas(Vec<ContinentPolygon>);

#[derive(Resource, Default)]
struct WorldMapDragState {
    last_cursor: Option<Vec2>,
}

#[derive(Component)]
struct LoadingText;

#[derive(Component)]
struct WorldMapText;

#[derive(Component)]
struct ContinentVisual;

#[derive(Component)]
struct RoadSegmentVisual;

#[derive(Component)]
struct BuildingVisual;

#[derive(Component)]
struct RouteMarker;

#[derive(Component)]
struct RouteLine;

#[derive(Component)]
struct TrackRoad;

#[derive(Component)]
struct TrackFence;

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

#[derive(Component)]
struct OsmFetchTask(bevy::tasks::Task<Result<OsmData, String>>);

#[derive(Component)]
struct WorldMapFetchTask(bevy::tasks::Task<Result<Vec<ContinentPolygon>, String>>);

#[derive(Component)]
struct UrbanFetchTask(bevy::tasks::Task<Result<Vec<ContinentPolygon>, String>>);

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
        Projection::from(PerspectiveProjection {
            far: 10000.0,
            ..default()
        }),
        Transform::from_xyz(0.0, 1500.0, 0.1).looking_at(Vec3::ZERO, -Vec3::Z),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 5000.0,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.6, 0.4, 0.0)),
    ));

    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(WORLD_SIZE, WORLD_SIZE))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.15, 0.35, 0.55),
            ..default()
        })),
        RigidBody::Static,
        Collider::cuboid(WORLD_SIZE, 0.5, WORLD_SIZE),
        Transform::from_xyz(0.0, -0.25, 0.0),
    ));

}

fn start_osm_fetch(mut commands: Commands, region: Res<SelectedRegion>) {
    let pool = AsyncComputeTaskPool::get();

    let task = if let Some(path) = parse_osm_file() {
        pool.spawn(async move { osm::load_osm_file(&path) })
    } else {
        let bbox = parse_bbox(&region);
        pool.spawn(async move { osm::fetch_osm_roads(&bbox) })
    };

    commands.spawn(OsmFetchTask(task));
}

fn enter_loading(mut commands: Commands) {
    commands.spawn((
        Text::new("Loading OpenStreetMap roads..."),
        TextColor(Color::srgb(1.0, 1.0, 1.0)),
        TextFont {
            font_size: FontSize::Px(24.0),
            ..default()
        },
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(20.0),
            left: Val::Px(20.0),
            ..default()
        },
        LoadingText,
    ));
}

fn enter_world_map(mut commands: Commands) {
    let pool = AsyncComputeTaskPool::get();
    let land_task = pool.spawn(async move { fetch_natural_earth_land() });
    let urban_task = pool.spawn(async move { fetch_urban_areas() });
    commands.spawn(WorldMapFetchTask(land_task));
    commands.spawn(UrbanFetchTask(urban_task));

    commands.spawn((
        Text::new("Loading world map..."),
        TextColor(Color::srgb(1.0, 1.0, 1.0)),
        TextFont {
            font_size: FontSize::Px(24.0),
            ..default()
        },
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(20.0),
            left: Val::Px(20.0),
            ..default()
        },
        WorldMapText,
    ));
}

fn exit_world_map(
    mut commands: Commands,
    continents: Query<Entity, With<ContinentVisual>>,
    ui: Query<Entity, With<WorldMapText>>,
) {
    for entity in continents.iter().chain(ui.iter()) {
        commands.entity(entity).despawn();
    }
}

fn check_world_map_fetch(
    mut commands: Commands,
    mut tasks: Query<(Entity, &mut WorldMapFetchTask)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut camera: Query<&mut Transform, With<Camera3d>>,
    ui: Query<Entity, With<WorldMapText>>,
    mut continents_res: ResMut<WorldContinents>,
) {
    for (entity, mut task) in tasks.iter_mut() {
        if let Some(result) = block_on(poll_once(&mut task.0)) {
            commands.entity(entity).despawn();

            let material = materials.add(StandardMaterial {
                base_color: Color::srgb(0.75, 0.65, 0.45),
                emissive: Color::srgb(0.4, 0.35, 0.25).into(),
                perceptual_roughness: 0.9,
                double_sided: true,
                unlit: true,
                ..default()
            });

            let help_text = match &result {
                Ok(polygons) => {
                    continents_res.0 = polygons.clone();
                    let mut min = Vec3::splat(f32::INFINITY);
                    let mut max = Vec3::splat(f32::NEG_INFINITY);
                    let mut rendered = 0;
                    for polygon in polygons {
                        if polygon.points.len() < 3 {
                            continue;
                        }
                        for &p in &polygon.points {
                            min = min.min(p);
                            max = max.max(p);
                        }
                        commands.spawn((
                            Mesh3d(meshes.add(build_continent_mesh(polygon))),
                            MeshMaterial3d(material.clone()),
                            Transform::default(),
                            ContinentVisual,
                        ));
                        rendered += 1;
                    }
                    eprintln!(
                        "Rendered {} continent polygons; bounds: min={:?} max={:?}",
                        rendered, min, max
                    );
                    format!(
                        "World Map\n\
                        Continents: {}\n\
                        Click anywhere to select a region\n\
                        Scroll: zoom in/out",
                        rendered
                    )
                }
                Err(e) => {
                    eprintln!("{e}");
                    format!(
                        "World Map\n\
                        Failed to load continents.\n\
                        {e}\n\
                        Download ne_110m_land.geojson manually or check network."
                    )
                }
            };

            for loading in ui.iter() {
                commands.entity(loading).despawn();
            }

            commands.spawn((
                Text::new(help_text),
                TextColor(Color::srgb(1.0, 1.0, 1.0)),
                TextFont {
                    font_size: FontSize::Px(18.0),
                    ..default()
                },
                Node {
                    position_type: PositionType::Absolute,
                    top: Val::Px(20.0),
                    left: Val::Px(20.0),
                    ..default()
                },
                WorldMapText,
            ));

            if let Ok(mut transform) = camera.single_mut() {
                *transform = Transform::from_xyz(0.0, 1500.0, 0.1)
                    .looking_at(Vec3::ZERO, -Vec3::Z);
            }
        }
    }
}

fn check_urban_fetch(
    mut commands: Commands,
    mut tasks: Query<(Entity, &mut UrbanFetchTask)>,
    mut urban_areas: ResMut<WorldUrbanAreas>,
) {
    for (entity, mut task) in tasks.iter_mut() {
        if let Some(result) = block_on(poll_once(&mut task.0)) {
            commands.entity(entity).despawn();
            match result {
                Ok(polygons) => {
                    eprintln!("Loaded {} urban area polygons", polygons.len());
                    urban_areas.0 = polygons;
                }
                Err(e) => {
                    eprintln!("Urban areas load failed: {e}");
                }
            }
        }
    }
}

fn draw_map_outlines(
    continents: Res<WorldContinents>,
    urban_areas: Res<WorldUrbanAreas>,
    camera: Query<&Transform, With<Camera3d>>,
    mut gizmos: Gizmos,
) {
    let continent_color = Color::srgb(0.9, 0.8, 0.2);
    for polygon in &continents.0 {
        for window in polygon.points.windows(2) {
            gizmos.line(window[0] + Vec3::Y * 1.0, window[1] + Vec3::Y * 1.0, continent_color);
        }
        if let (Some(first), Some(last)) = (polygon.points.first(), polygon.points.last()) {
            gizmos.line(*last + Vec3::Y * 1.0, *first + Vec3::Y * 1.0, continent_color);
        }
    }

    const URBAN_ZOOM_HEIGHT: f32 = 1500.0;
    let Ok(transform) = camera.single() else {
        return;
    };
    if transform.translation.y > URBAN_ZOOM_HEIGHT {
        return;
    }

    static mut URBAN_DIAGNOSTICS_SHOWN: bool = false;
    if !urban_areas.0.is_empty() && !unsafe { URBAN_DIAGNOSTICS_SHOWN } {
        unsafe { URBAN_DIAGNOSTICS_SHOWN = true };
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for polygon in &urban_areas.0 {
            for &p in &polygon.points {
                min = min.min(p);
                max = max.max(p);
            }
        }
        eprintln!(
            "Drawing {} urban areas at camera height {}; bounds: min={:?} max={:?}",
            urban_areas.0.len(),
            transform.translation.y,
            min,
            max
        );
    }

    let urban_color = Color::srgb(0.95, 0.95, 1.0);
    let marker_color = Color::srgb(1.0, 0.1, 0.1);
    let marker_radius = (5.0 + (1500.0 - transform.translation.y) * 0.02).max(5.0);
    let circle_rotation = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);

    for polygon in &urban_areas.0 {
        for window in polygon.points.windows(2) {
            gizmos.line(window[0] + Vec3::Y * 5.0, window[1] + Vec3::Y * 5.0, urban_color);
        }
        if let (Some(first), Some(last)) = (polygon.points.first(), polygon.points.last()) {
            gizmos.line(*last + Vec3::Y * 5.0, *first + Vec3::Y * 5.0, urban_color);
        }

        let center = polygon_centroid(polygon);
        gizmos.circle(
            Isometry3d::new(center + Vec3::Y * 5.0, circle_rotation),
            marker_radius,
            marker_color,
        );
    }
}

fn polygon_centroid(polygon: &ContinentPolygon) -> Vec3 {
    let mut sum = Vec3::ZERO;
    for &p in &polygon.points {
        sum += p;
    }
    sum / polygon.points.len().max(1) as f32
}

fn build_continent_mesh(polygon: &ContinentPolygon) -> Mesh {
    let n = polygon.points.len();
    let flat: Vec<f64> = polygon
        .points
        .iter()
        .flat_map(|p| [p.x as f64, p.z as f64])
        .collect();
    let tri_indices = earcutr::earcut(&flat, &[], 2).unwrap_or_default();

    let mut vertices: Vec<[f32; 3]> = Vec::with_capacity(n);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(n);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(n);

    for p in &polygon.points {
        vertices.push([p.x, 0.5, p.z]);
        normals.push([0.0, 1.0, 0.0]);
        uvs.push([0.0, 0.0]);
    }

    // Duplicate every triangle with reversed winding so both faces render
    // regardless of material double-sided settings.
    let triangle_count = tri_indices.len() / 3;
    let mut indices = Vec::with_capacity(tri_indices.len() * 2);
    for tri in tri_indices.chunks(3) {
        let a = tri[0] as u32;
        let b = tri[1] as u32;
        let c = tri[2] as u32;
        indices.push(a);
        indices.push(b);
        indices.push(c);
        indices.push(a);
        indices.push(c);
        indices.push(b);
    }

    eprintln!(
        "continent mesh: {} points, {} earcut triangles -> {} total triangles",
        n,
        triangle_count,
        indices.len() / 3
    );

    Mesh::new(
        bevy::render::mesh::PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vertices)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_indices(bevy::render::mesh::Indices::U32(indices))
}

#[cfg(test)]
mod main_tests {
    use super::*;

    #[test]
    fn continent_mesh_has_triangles() {
        let polygon = ContinentPolygon {
            points: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(100.0, 0.0, 0.0),
                Vec3::new(100.0, 0.0, 50.0),
                Vec3::new(0.0, 0.0, 50.0),
            ],
        };
        let mesh = build_continent_mesh(&polygon);
        let indices = mesh.indices().expect("mesh should have indices");
        assert!(
            indices.len() >= 6,
            "expected at least two triangles for a continent rectangle"
        );
    }
}

fn world_map_zoom(
    mut scroll_events: MessageReader<MouseWheel>,
    mut camera: Query<&mut Transform, With<Camera3d>>,
) {
    let Ok(mut transform) = camera.single_mut() else {
        return;
    };

    const ZOOM_SPEED: f32 = 8.0;
    const MIN_HEIGHT: f32 = 50.0;
    const MAX_HEIGHT: f32 = 1500.0;

    for event in scroll_events.read() {
        let delta = match event.unit {
            MouseScrollUnit::Line => event.y,
            MouseScrollUnit::Pixel => event.y * 0.01,
        };
        transform.translation.y -= delta * ZOOM_SPEED;
        transform.translation.y = transform.translation.y.clamp(MIN_HEIGHT, MAX_HEIGHT);
    }
}

fn world_map_drag(
    mut drag_state: ResMut<WorldMapDragState>,
    mouse: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    camera: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
    mut camera_local: Query<&mut Transform, With<Camera3d>>,
) {
    let Ok(window) = windows.single() else {
        drag_state.last_cursor = None;
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        drag_state.last_cursor = None;
        return;
    };

    if mouse.just_released(MouseButton::Left) || !mouse.pressed(MouseButton::Left) {
        drag_state.last_cursor = None;
        return;
    }

    let Ok((camera, camera_global)) = camera.single() else {
        return;
    };
    let Ok(mut transform) = camera_local.single_mut() else {
        return;
    };

    if let Some(last) = drag_state.last_cursor {
        let ray_current = camera.viewport_to_world(camera_global, cursor).ok();
        let ray_last = camera.viewport_to_world(camera_global, last).ok();
        if let (Some(rc), Some(rl)) = (ray_current, ray_last) {
            let t_current = -rc.origin.y / rc.direction.y;
            let t_last = -rl.origin.y / rl.direction.y;
            if t_current.is_finite() && t_last.is_finite() && t_current > 0.0 && t_last > 0.0 {
                let current_hit = rc.origin + rc.direction * t_current;
                let last_hit = rl.origin + rl.direction * t_last;
                let delta = current_hit - last_hit;
                transform.translation.x -= delta.x;
                transform.translation.z -= delta.z;
            }
        }
    }

    drag_state.last_cursor = Some(cursor);
}

fn check_osm_fetch(
    mut commands: Commands,
    mut tasks: Query<(Entity, &mut OsmFetchTask)>,
    mut next_state: ResMut<NextState<AppState>>,
    loading_query: Query<Entity, With<LoadingText>>,
) {
    for (entity, mut task) in tasks.iter_mut() {
        if let Some(result) = block_on(poll_once(&mut task.0)) {
            commands.entity(entity).despawn();

            let mut data = result.unwrap_or_else(|e| {
                eprintln!("OSM load failed: {e}");
                OsmData {
                    roads: Vec::new(),
                    buildings: Vec::new(),
                }
            });

            // Scale OSM meter coordinates down to fit the play area.
            for segment in &mut data.roads {
                for point in &mut segment.points {
                    point.x *= OSM_SCALE;
                    point.z *= OSM_SCALE;
                }
            }
            for building in &mut data.buildings {
                for point in &mut building.footprint {
                    point.x *= OSM_SCALE;
                    point.z *= OSM_SCALE;
                }
                building.height *= OSM_SCALE;
            }

            eprintln!("Loaded {} roads and {} buildings", data.roads.len(), data.buildings.len());
            commands.insert_resource(OsmRoads(data.roads));
            commands.insert_resource(OsmBuildings(data.buildings));

            if let Ok(loading) = loading_query.single() {
                commands.entity(loading).despawn();
            }

            next_state.set(AppState::MapSelect);
        }
    }
}

fn enter_map_select(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    roads: Res<OsmRoads>,
    buildings: Res<OsmBuildings>,
    route: Res<Route>,
    mut camera: Query<&mut Transform, With<Camera3d>>,
) {
    let has_roads = !roads.0.is_empty();

    if has_roads {
        let road_material = materials.add(StandardMaterial {
            base_color: Color::srgb(0.6, 0.6, 0.65),
            ..default()
        });
        let road_mesh_base = meshes.add(Cuboid::new(1.0, 0.15, 1.0).mesh());

        for segment in &roads.0 {
            for window in segment.points.windows(2) {
                let a = window[0];
                let b = window[1];
                let delta = b - a;
                let length = delta.length();
                if length < 0.01 {
                    continue;
                }
                let mid = (a + b) * 0.5;
                let angle = delta.z.atan2(delta.x);
                let width = 2.0;

                commands.spawn((
                    Mesh3d(road_mesh_base.clone()),
                    MeshMaterial3d(road_material.clone()),
                    Transform {
                        translation: mid + Vec3::Y * 0.075,
                        rotation: Quat::from_rotation_y(-angle),
                        scale: Vec3::new(length, 1.0, width),
                    },
                    RoadSegmentVisual,
                ));
            }
        }
    }

    let building_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.35, 0.35, 0.38),
        perceptual_roughness: 0.9,
        double_sided: true,
        ..default()
    });

    for building in &buildings.0 {
        if building.footprint.len() < 3 {
            continue;
        }

        commands.spawn((
            Mesh3d(meshes.add(build_building_mesh(building))),
            MeshMaterial3d(building_material.clone()),
            Transform::default(),
            BuildingVisual,
        ));
    }

    spawn_route_markers(&mut commands, &mut meshes, &mut materials, &route.0);
    spawn_route_lines(&mut commands, &mut meshes, &mut materials, &route.0);

    frame_camera_on_data(&roads, &buildings, &mut camera);

    let road_count = roads.0.len();
    let building_count = buildings.0.len();

    let help_text = if has_roads {
        format!(
            "Map Editor\n\
            Roads: {road_count}  Buildings: {building_count}\n\
            Click a road to add a marker\n\
            Scroll: zoom in/out  Home: reset view\n\
            Backspace: remove last marker\n\
            C: clear all markers\n\
            Enter: drive the track\n\
            M: back to world map\n\
            A track must be created from OSM roads before driving"
        )
    } else {
        "No OSM roads loaded.\n\
        Check your network, or run with:\n\
        --osm-file path/to/map.osm\n\
        or --bbox \"min_lat,min_lon,max_lat,max_lon\"".to_string()
    };

    commands.spawn((
        Text::new(help_text),
        TextColor(Color::srgb(1.0, 1.0, 1.0)),
        TextFont {
            font_size: FontSize::Px(18.0),
            ..default()
        },
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(20.0),
            left: Val::Px(20.0),
            ..default()
        },
        LoadingText,
    ));
}

fn exit_map_select(
    mut commands: Commands,
    roads: Query<Entity, With<RoadSegmentVisual>>,
    buildings: Query<Entity, With<BuildingVisual>>,
    markers: Query<Entity, With<RouteMarker>>,
    lines: Query<Entity, With<RouteLine>>,
    ui: Query<Entity, With<LoadingText>>,
) {
    for entity in roads
        .iter()
        .chain(buildings.iter())
        .chain(markers.iter())
        .chain(lines.iter())
        .chain(ui.iter())
    {
        commands.entity(entity).despawn();
    }
}

#[allow(clippy::too_many_arguments)]
fn map_select_input(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mouse: Res<ButtonInput<MouseButton>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window>,
    camera: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
    mut camera_transform: Query<&mut Transform, With<Camera3d>>,
    roads: Res<OsmRoads>,
    buildings: Res<OsmBuildings>,
    mut route: ResMut<Route>,
    markers: Query<Entity, With<RouteMarker>>,
    lines: Query<Entity, With<RouteLine>>,
    mut next_state: ResMut<NextState<AppState>>,
) {
    if keyboard.just_pressed(KeyCode::Enter) || keyboard.just_pressed(KeyCode::NumpadEnter) {
        let track = Track::new(route.0.clone());
        if track.is_valid() {
            let _ = save_track(&track, TRACK_FILE);
            next_state.set(AppState::Drive);
        }
        return;
    }

    if keyboard.just_pressed(KeyCode::Home) {
        frame_camera_on_data(&roads, &buildings, &mut camera_transform);
        return;
    }

    if keyboard.just_pressed(KeyCode::KeyM) {
        next_state.set(AppState::WorldMap);
        return;
    }

    if keyboard.just_pressed(KeyCode::Backspace) {
        route.0.pop();
        refresh_route_visuals(
            &mut commands,
            &mut meshes,
            &mut materials,
            &route.0,
            &markers,
            &lines,
        );
        return;
    }

    if keyboard.just_pressed(KeyCode::KeyC) {
        route.0.clear();
        refresh_route_visuals(
            &mut commands,
            &mut meshes,
            &mut materials,
            &route.0,
            &markers,
            &lines,
        );
        return;
    }

    if !mouse.just_pressed(MouseButton::Left) {
        return;
    }

    let Ok(window) = windows.single() else {
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    let Ok((camera, camera_transform)) = camera.single() else {
        return;
    };

    let Ok(ray) = camera.viewport_to_world(camera_transform, cursor) else {
        return;
    };

    let t = -ray.origin.y / ray.direction.y;
    if t < 0.0 || !t.is_finite() {
        return;
    }
    let hit = ray.origin + ray.direction * t;

    let snap = if roads.0.is_empty() {
        hit
    } else {
        osm::nearest_point_on_segments(&roads.0, hit)
    };

    route.0.push(snap);
    refresh_route_visuals(
        &mut commands,
        &mut meshes,
        &mut materials,
        &route.0,
        &markers,
        &lines,
    );
}

fn refresh_route_visuals(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    points: &[Vec3],
    markers: &Query<Entity, With<RouteMarker>>,
    lines: &Query<Entity, With<RouteLine>>,
) {
    for entity in markers.iter().chain(lines.iter()) {
        commands.entity(entity).despawn();
    }
    spawn_route_markers(commands, meshes, materials, points);
    spawn_route_lines(commands, meshes, materials, points);
}

fn spawn_route_markers(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    points: &[Vec3],
) {
    let marker_material = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.8, 0.0),
        ..default()
    });
    let marker_mesh = meshes.add(Sphere::new(0.3).mesh().ico(8).unwrap());

    for &point in points {
        commands.spawn((
            Mesh3d(marker_mesh.clone()),
            MeshMaterial3d(marker_material.clone()),
            Transform::from_translation(point + Vec3::Y * 0.3),
            RouteMarker,
        ));
    }
}

fn spawn_route_lines(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    points: &[Vec3],
) {
    if points.len() < 2 {
        return;
    }

    let line_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.9, 0.7, 0.0),
        ..default()
    });
    let line_mesh_base = meshes.add(Cuboid::new(1.0, 0.05, 1.0).mesh());

    for window in points.windows(2) {
        let a = window[0];
        let b = window[1];
        let delta = b - a;
        let length = delta.length();
        if length < 0.01 {
            continue;
        }
        let mid = (a + b) * 0.5;
        let angle = delta.z.atan2(delta.x);

        commands.spawn((
            Mesh3d(line_mesh_base.clone()),
            MeshMaterial3d(line_material.clone()),
            Transform {
                translation: mid + Vec3::Y * 0.1,
                rotation: Quat::from_rotation_y(-angle),
                scale: Vec3::new(length, 1.0, 0.2),
            },
            RouteLine,
        ));
    }
}

fn frame_camera_on_data(
    roads: &OsmRoads,
    buildings: &OsmBuildings,
    camera: &mut Query<&mut Transform, With<Camera3d>>,
) {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    let mut has_points = false;

    for segment in &roads.0 {
        for &point in &segment.points {
            min = min.min(point);
            max = max.max(point);
            has_points = true;
        }
    }
    for building in &buildings.0 {
        for &point in &building.footprint {
            min = min.min(point);
            max = max.max(point);
            has_points = true;
        }
    }

    if has_points {
        let center = (min + max) * 0.5;
        let size = (max - min).max_element();
        let height = (size * 1.5 + 50.0).min(1500.0);

        if let Ok(mut transform) = camera.single_mut() {
            *transform = Transform::from_xyz(center.x, height, center.z + 0.1)
                .looking_at(center, -Vec3::Z);
        }
    }
}

fn update_camera_overview(
    roads: Res<OsmRoads>,
    buildings: Res<OsmBuildings>,
    mut camera: Query<&mut Transform, With<Camera3d>>,
) {
    if roads.is_changed() || buildings.is_changed() {
        frame_camera_on_data(&roads, &buildings, &mut camera);
    }
}

fn map_select_zoom(
    mut scroll_events: MessageReader<MouseWheel>,
    mut camera: Query<&mut Transform, With<Camera3d>>,
) {
    let Ok(mut transform) = camera.single_mut() else {
        return;
    };

    const ZOOM_SPEED: f32 = 8.0;
    const MIN_HEIGHT: f32 = 10.0;
    const MAX_HEIGHT: f32 = 1500.0;

    for event in scroll_events.read() {
        let delta = match event.unit {
            MouseScrollUnit::Line => event.y,
            MouseScrollUnit::Pixel => event.y * 0.01,
        };
        transform.translation.y -= delta * ZOOM_SPEED;
        transform.translation.y = transform.translation.y.clamp(MIN_HEIGHT, MAX_HEIGHT);
    }
}

fn enter_drive(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    route: Res<Route>,
    buildings: Res<OsmBuildings>,
) {
    let track = Track::new(route.0.clone());
    if !track.is_valid() {
        eprintln!("Cannot drive: no valid track. Mark at least 3 road points.");
        return;
    }

    let road_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.25, 0.25, 0.25),
        ..default()
    });

    commands.spawn((
        Mesh3d(meshes.add(build_track_mesh(&track))),
        MeshMaterial3d(road_material),
        RigidBody::Static,
        Collider::trimesh_from_mesh(&build_track_mesh(&track)).unwrap_or(Collider::cuboid(1.0, 0.1, 1.0)),
        Transform::default(),
        TrackRoad,
    ));

    let fence_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.85, 0.85, 0.85),
        ..default()
    });
    let fence_mesh = meshes.add(Cuboid::new(1.0, 1.0, 1.0).mesh());

    let segments = track.control_points.len().max(3) * 10;
    let (left_fences, right_fences) = build_fences(&track, segments);
    for fence in left_fences.into_iter().chain(right_fences) {
        commands.spawn((
            Mesh3d(fence_mesh.clone()),
            MeshMaterial3d(fence_material.clone()),
            fence.transform,
            RigidBody::Static,
            Collider::cuboid(fence.size.x, fence.size.y, fence.size.z),
            Friction::new(0.9),
            TrackFence,
        ));
    }

    let building_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.35, 0.35, 0.38),
        perceptual_roughness: 0.9,
        double_sided: true,
        ..default()
    });

    for building in &buildings.0 {
        if building.footprint.len() < 3 {
            continue;
        }

        commands.spawn((
            Mesh3d(meshes.add(build_building_mesh(building))),
            MeshMaterial3d(building_material.clone()),
            Transform::default(),
            BuildingVisual,
        ));
    }

    let start_pos = track.control_points.first().copied().unwrap_or(Vec3::ZERO);
    let next_pos = track.control_points.get(1).copied().unwrap_or(start_pos + Vec3::X);
    let forward = (next_pos - start_pos).normalize_or(Vec3::X);
    let rotation = Quat::from_rotation_arc(Vec3::Z, forward);

    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0).mesh())),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.85, 0.1, 0.1),
            ..default()
        })),
        Transform::from_translation(start_pos + Vec3::Y * 0.75).with_rotation(rotation),
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

    commands.spawn((
        Text::new("Drive mode\nWASD / Arrows: drive\nR: return to editor\nM: world map"),
        TextColor(Color::srgb(1.0, 1.0, 1.0)),
        TextFont {
            font_size: FontSize::Px(18.0),
            ..default()
        },
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(20.0),
            left: Val::Px(20.0),
            ..default()
        },
        LoadingText,
    ));
}

fn exit_drive(
    mut commands: Commands,
    roads: Query<Entity, With<TrackRoad>>,
    fences: Query<Entity, With<TrackFence>>,
    buildings: Query<Entity, With<BuildingVisual>>,
    vehicles: Query<Entity, With<Vehicle>>,
    ui: Query<Entity, With<LoadingText>>,
) {
    for entity in roads
        .iter()
        .chain(fences.iter())
        .chain(buildings.iter())
        .chain(vehicles.iter())
        .chain(ui.iter())
    {
        commands.entity(entity).despawn();
    }
}

fn drive_input(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut next_state: ResMut<NextState<AppState>>,
) {
    if keyboard.just_pressed(KeyCode::KeyR) {
        next_state.set(AppState::MapSelect);
    }
    if keyboard.just_pressed(KeyCode::KeyM) {
        next_state.set(AppState::WorldMap);
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

    let speed_error = controller.throttle - speed;
    forces.apply_force(forward * speed_error * SPEED_CONTROL_GAIN * mass.0);

    let lateral_velocity = forces.linear_velocity() - forward * speed;
    forces.apply_force(-lateral_velocity * LATERAL_GRIP * mass.0);

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
