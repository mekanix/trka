use crate::osm::Building;
use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use serde::{Deserialize, Serialize};

pub const DEFAULT_TRACK_WIDTH: f32 = 7.0;
pub const DEFAULT_FENCE_HEIGHT: f32 = 1.0;
pub const DEFAULT_FENCE_THICKNESS: f32 = 0.5;

#[derive(Resource, Clone, Debug, Default, Serialize, Deserialize)]
pub struct Track {
    pub control_points: Vec<Vec3>,
    pub width: f32,
}

impl Track {
    pub fn new(control_points: Vec<Vec3>) -> Self {
        Self {
            control_points,
            width: DEFAULT_TRACK_WIDTH,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.control_points.len() >= 3
    }
}

/// Closed-loop Catmull-Rom spline evaluation.
/// `t` is in [0, 1] around the whole loop.
pub fn evaluate_spline(points: &[Vec3], t: f32) -> Vec3 {
    let n = points.len();
    if n == 0 {
        return Vec3::ZERO;
    }
    if n == 1 {
        return points[0];
    }

    let segments = n;
    let t = t.clamp(0.0, 1.0 - f32::EPSILON) * segments as f32;
    let i = t.floor() as usize % n;
    let local_t = t - i as f32;

    let p0 = points[(i + n - 1) % n];
    let p1 = points[i];
    let p2 = points[(i + 1) % n];
    let p3 = points[(i + 2) % n];

    catmull_rom(p0, p1, p2, p3, local_t)
}

pub fn evaluate_spline_tangent(points: &[Vec3], t: f32) -> Vec3 {
    let n = points.len();
    if n < 2 {
        return Vec3::Z;
    }

    let segments = n;
    let t = t.clamp(0.0, 1.0 - f32::EPSILON) * segments as f32;
    let i = t.floor() as usize % n;
    let local_t = t - i as f32;

    let p0 = points[(i + n - 1) % n];
    let p1 = points[i];
    let p2 = points[(i + 1) % n];
    let p3 = points[(i + 2) % n];

    catmull_rom_tangent(p0, p1, p2, p3, local_t)
}

fn catmull_rom(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32) -> Vec3 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

fn catmull_rom_tangent(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32) -> Vec3 {
    let t2 = t * t;
    0.5 * ((-p0 + p2)
        + 2.0 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t
        + 3.0 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t2)
}

pub fn build_track_mesh(track: &Track) -> Mesh {
    let samples = track.control_points.len().max(3) * 20;
    let mut vertices: Vec<[f32; 3]> = Vec::with_capacity(samples * 2);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(samples * 2);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(samples * 2);
    let mut indices: Vec<u32> = Vec::with_capacity(samples * 6);

    let half_width = track.width * 0.5;
    let mut accumulated_distance = 0.0;
    let mut previous_center = None;

    for i in 0..=samples {
        let t = (i as f32 / samples as f32) % 1.0;
        let center = evaluate_spline(&track.control_points, t);
        let tangent = evaluate_spline_tangent(&track.control_points, t);
        let forward = tangent.normalize_or(Vec3::Z);
        let right = forward.cross(Vec3::Y).normalize_or(Vec3::X);

        let left = center - right * half_width;
        let right_pos = center + right * half_width;

        if let Some(prev) = previous_center {
            accumulated_distance += center.distance(prev);
        }
        previous_center = Some(center);

        let v = accumulated_distance * 0.1;

        vertices.push(left.into());
        vertices.push(right_pos.into());
        normals.push([0.0, 1.0, 0.0]);
        normals.push([0.0, 1.0, 0.0]);
        uvs.push([0.0, v]);
        uvs.push([1.0, v]);

        if i < samples {
            let base = i as u32 * 2;
            indices.push(base);
            indices.push(base + 1);
            indices.push(base + 2);
            indices.push(base + 2);
            indices.push(base + 1);
            indices.push(base + 3);
        }
    }

    Mesh::new(
        bevy::render::mesh::PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vertices)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_indices(bevy::render::mesh::Indices::U32(indices))
}

pub struct FenceSpawn {
    pub transform: Transform,
    pub size: Vec3,
}

pub fn build_fences(track: &Track, segments: usize) -> (Vec<FenceSpawn>, Vec<FenceSpawn>) {
    let half_width = track.width * 0.5;
    let mut left_fences = Vec::with_capacity(segments);
    let mut right_fences = Vec::with_capacity(segments);

    for i in 0..segments {
        let t0 = i as f32 / segments as f32;
        let t1 = ((i + 1) as f32 / segments as f32) % 1.0;
        let t = (t0 + t1) * 0.5;

        let center = evaluate_spline(&track.control_points, t);
        let tangent = evaluate_spline_tangent(&track.control_points, t);
        let forward = tangent.normalize_or(Vec3::Z);
        let right = forward.cross(Vec3::Y).normalize_or(Vec3::X);

        let angle = forward.z.atan2(forward.x);

        let p0 = evaluate_spline(&track.control_points, t0);
        let p1 = evaluate_spline(&track.control_points, t1);
        let arc = p0.distance(p1);
        let depth = arc + 0.15;

        let left_pos = center - right * (half_width + DEFAULT_FENCE_THICKNESS * 0.5);
        let right_pos = center + right * (half_width + DEFAULT_FENCE_THICKNESS * 0.5);

        left_fences.push(FenceSpawn {
            transform: Transform {
                translation: left_pos + Vec3::Y * (DEFAULT_FENCE_HEIGHT * 0.5),
                rotation: Quat::from_rotation_y(-angle),
                scale: Vec3::new(DEFAULT_FENCE_THICKNESS, DEFAULT_FENCE_HEIGHT, depth),
            },
            size: Vec3::new(DEFAULT_FENCE_THICKNESS, DEFAULT_FENCE_HEIGHT, depth),
        });
        right_fences.push(FenceSpawn {
            transform: Transform {
                translation: right_pos + Vec3::Y * (DEFAULT_FENCE_HEIGHT * 0.5),
                rotation: Quat::from_rotation_y(-angle),
                scale: Vec3::new(DEFAULT_FENCE_THICKNESS, DEFAULT_FENCE_HEIGHT, depth),
            },
            size: Vec3::new(DEFAULT_FENCE_THICKNESS, DEFAULT_FENCE_HEIGHT, depth),
        });
    }

    (left_fences, right_fences)
}

pub fn save_track(track: &Track, path: &str) -> Result<(), String> {
    let json = serde_json::to_string_pretty(track).map_err(|e| e.to_string())?;
    std::fs::write(path, json).map_err(|e| e.to_string())
}

pub fn build_building_mesh(building: &Building) -> Mesh {
    // Use the parsed OSM height, but keep a small minimum so degenerate data still renders.
    let height = building.height.max(0.1);
    let n = building.footprint.len();

    let mut vertices: Vec<[f32; 3]> = Vec::with_capacity(n * 2 + n * 4);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(n * 2 + n * 4);
    let mut indices: Vec<u32> = Vec::with_capacity(n * 12);

    // Triangulate the footprint polygon (in the XZ plane).
    let flat: Vec<f64> = building
        .footprint
        .iter()
        .flat_map(|p| [p.x as f64, p.z as f64])
        .collect();
    let tri_indices = earcutr::earcut(&flat, &[], 2).unwrap_or_default();

    // Determine winding so side-wall normals point outward.
    let signed_area: f32 = building
        .footprint
        .windows(2)
        .map(|w| w[0].x * w[1].z - w[1].x * w[0].z)
        .sum();
    let is_ccw = signed_area > 0.0;

    // Bottom face vertices.
    for p in &building.footprint {
        vertices.push([p.x, 0.0, p.z]);
        normals.push([0.0, -1.0, 0.0]);
    }
    // Top face vertices.
    for p in &building.footprint {
        vertices.push([p.x, height, p.z]);
        normals.push([0.0, 1.0, 0.0]);
    }

    let bottom_offset = 0u32;
    let top_offset = n as u32;

    // Bottom face triangles.
    for tri in tri_indices.chunks(3) {
        indices.push(bottom_offset + tri[0] as u32);
        indices.push(bottom_offset + tri[2] as u32);
        indices.push(bottom_offset + tri[1] as u32);
    }
    // Top face triangles.
    for tri in tri_indices.chunks(3) {
        indices.push(top_offset + tri[0] as u32);
        indices.push(top_offset + tri[1] as u32);
        indices.push(top_offset + tri[2] as u32);
    }

    // Side walls: for each footprint edge, add a quad.
    for i in 0..n {
        let j = (i + 1) % n;
        let bi = building.footprint[i];
        let bj = building.footprint[j];
        let ti = Vec3::new(bi.x, height, bi.z);
        let tj = Vec3::new(bj.x, height, bj.z);

        let d = bj - bi;
        let mut normal = Vec3::new(d.z, 0.0, -d.x).normalize_or(Vec3::Z);
        if !is_ccw {
            normal = -normal;
        }

        let wall_base = vertices.len() as u32;
        vertices.push([bi.x, 0.0, bi.z]);
        vertices.push([bj.x, 0.0, bj.z]);
        vertices.push([tj.x, tj.y, tj.z]);
        vertices.push([ti.x, ti.y, ti.z]);
        for _ in 0..4 {
            normals.push([normal.x, normal.y, normal.z]);
        }

        indices.push(wall_base);
        indices.push(wall_base + 1);
        indices.push(wall_base + 2);
        indices.push(wall_base);
        indices.push(wall_base + 3);
        indices.push(wall_base + 2);
    }

    Mesh::new(
        bevy::render::mesh::PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vertices)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(bevy::render::mesh::Indices::U32(indices))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_simple_building_mesh() {
        let building = Building {
            footprint: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(10.0, 0.0, 0.0),
                Vec3::new(10.0, 0.0, 5.0),
                Vec3::new(0.0, 0.0, 5.0),
            ],
            height: 5.0,
        };
        let mesh = build_building_mesh(&building);
        let indices = mesh.indices().expect("mesh should have indices");
        assert!(
            indices.len() >= 6,
            "expected at least two triangles for a rectangle"
        );
    }
}
