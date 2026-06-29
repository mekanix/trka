use bevy::prelude::*;
use quick_xml::events::Event;
use quick_xml::Reader;
use std::collections::HashMap;

const EARTH_RADIUS_M: f64 = 6_371_000.0;
const OVERPASS_URL: &str = "https://overpass-api.de/api/interpreter";
const DEFAULT_BUILDING_HEIGHT_M: f32 = 8.0;
const METERS_PER_LEVEL: f32 = 3.0;

#[derive(Debug, Clone)]
pub struct RoadSegment {
    pub points: Vec<Vec3>,
}

#[derive(Debug, Clone)]
pub struct Building {
    pub footprint: Vec<Vec3>,
    pub height: f32,
}

pub struct OsmData {
    pub roads: Vec<RoadSegment>,
    pub buildings: Vec<Building>,
}

pub fn load_osm_file(path: &str) -> Result<OsmData, String> {
    let xml = std::fs::read_to_string(path).map_err(|e| format!("Failed to read {path}: {e}"))?;
    parse_osm(&xml)
}

pub fn fetch_osm_roads(bbox: &str) -> Result<OsmData, String> {
    let query = format!(
        r#"[out:xml];
(
  way["highway"]({bbox});
  way["building"]({bbox});
);
(._;>;);
out body;"#
    );

    let response = ureq::post(OVERPASS_URL)
        .set("User-Agent", "trka-racing-game/0.1")
        .send_form(&[("data", &query)])
        .map_err(|e| format!("Overpass request failed: {e}"))?;

    let body = response
        .into_string()
        .map_err(|e| format!("Failed to read Overpass response: {e}"))?;

    parse_osm(&body)
}

#[derive(Debug, Clone, Copy)]
enum WayKind {
    Highway,
    Building { height: f32 },
}

fn parse_osm(xml: &str) -> Result<OsmData, String> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);

    let mut buf = Vec::new();
    let mut nodes: HashMap<i64, (f64, f64)> = HashMap::new();
    let mut road_refs: Vec<(i64, Vec<i64>)> = Vec::new();
    let mut building_refs: Vec<(i64, Vec<i64>, f32)> = Vec::new();

    let mut current_way: Option<(i64, Vec<i64>)> = None;
    let mut current_kind: Option<WayKind> = None;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Empty(ref e)) | Ok(Event::Start(ref e)) => match e.name().as_ref() {
                b"node" => {
                    let mut id: Option<i64> = None;
                    let mut lat: Option<f64> = None;
                    let mut lon: Option<f64> = None;
                    for attr in e.attributes().flatten() {
                        match attr.key.as_ref() {
                            b"id" => id = attr.unescape_value().ok().and_then(|v| v.parse().ok()),
                            b"lat" => lat = attr.unescape_value().ok().and_then(|v| v.parse().ok()),
                            b"lon" => lon = attr.unescape_value().ok().and_then(|v| v.parse().ok()),
                            _ => {}
                        }
                    }
                    if let (Some(id), Some(lat), Some(lon)) = (id, lat, lon) {
                        nodes.insert(id, (lat, lon));
                    }
                }
                b"way" => {
                    let id = e
                        .attributes()
                        .flatten()
                        .find(|a| a.key.as_ref() == b"id")
                        .and_then(|a| a.unescape_value().ok())
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0);
                    current_way = Some((id, Vec::new()));
                    current_kind = None;
                }
                b"nd" => {
                    if let Some((_, ref mut refs)) = current_way {
                        if let Some(node_ref) = e
                            .attributes()
                            .flatten()
                            .find(|a| a.key.as_ref() == b"ref")
                            .and_then(|a| a.unescape_value().ok())
                            .and_then(|v| v.parse().ok())
                        {
                            refs.push(node_ref);
                        }
                    }
                }
                b"tag" if current_way.is_some() => {
                    let key = e
                        .attributes()
                        .flatten()
                        .find(|a| a.key.as_ref() == b"k")
                        .and_then(|a| a.unescape_value().ok());
                    let value = e
                        .attributes()
                        .flatten()
                        .find(|a| a.key.as_ref() == b"v")
                        .and_then(|a| a.unescape_value().ok());

                    match key.as_deref() {
                        Some("highway") => {
                            current_kind = Some(WayKind::Highway);
                        }
                        Some("building") => {
                            current_kind = Some(WayKind::Building {
                                height: DEFAULT_BUILDING_HEIGHT_M,
                            });
                        }
                        Some("height") => {
                            if let Some(WayKind::Building { .. }) = current_kind {
                                if let Some(height) = value.as_ref().and_then(|v| parse_height(v)) {
                                    current_kind = Some(WayKind::Building { height });
                                }
                            }
                        }
                        Some("building:levels") => {
                            if let Some(levels) = value.as_ref().and_then(|v| v.parse::<f32>().ok()) {
                                current_kind = Some(WayKind::Building {
                                    height: levels * METERS_PER_LEVEL,
                                });
                            }
                        }
                        _ => {}
                    }
                }
                _ => {}
            },
            Ok(Event::End(ref e)) if e.name().as_ref() == b"way" => {
                if let Some((id, refs)) = current_way.take() {
                    match current_kind {
                        Some(WayKind::Highway) if refs.len() >= 2 => {
                            road_refs.push((id, refs));
                        }
                        Some(WayKind::Building { height }) if refs.len() >= 3 => {
                            building_refs.push((id, refs, height));
                        }
                        _ => {}
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(format!("XML parse error: {e}")),
            _ => {}
        }
        buf.clear();
    }

    let center = compute_center(&nodes);
    let mut roads = Vec::new();
    for (_id, refs) in road_refs {
        let mut points = Vec::with_capacity(refs.len());
        for node_ref in refs {
            if let Some(&(lat, lon)) = nodes.get(&node_ref) {
                let pos = latlon_to_local(lat, lon, center.0, center.1);
                points.push(pos);
            }
        }
        if points.len() >= 2 {
            roads.push(RoadSegment { points });
        }
    }

    let mut buildings = Vec::new();
    for (_id, refs, height) in building_refs {
        let mut points = Vec::with_capacity(refs.len());
        for node_ref in refs {
            if let Some(&(lat, lon)) = nodes.get(&node_ref) {
                let pos = latlon_to_local(lat, lon, center.0, center.1);
                points.push(pos);
            }
        }
        // Drop the repeated closing node if present.
        if points.len() >= 4 && points.first() == points.last() {
            points.pop();
        }
        if points.len() >= 3 {
            buildings.push(Building { footprint: points, height });
        }
    }

    Ok(OsmData { roads, buildings })
}

fn parse_height(value: &str) -> Option<f32> {
    let trimmed = value.trim();
    // Take the leading numeric part, ignoring units like "m" or "ft".
    let numeric: String = trimmed
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
        .collect();
    numeric.parse().ok()
}

fn compute_center(nodes: &HashMap<i64, (f64, f64)>) -> (f64, f64) {
    let mut lat_sum = 0.0;
    let mut lon_sum = 0.0;
    let count = nodes.len() as f64;
    for &(lat, lon) in nodes.values() {
        lat_sum += lat;
        lon_sum += lon;
    }
    (lat_sum / count, lon_sum / count)
}

fn latlon_to_local(lat: f64, lon: f64, center_lat: f64, center_lon: f64) -> Vec3 {
    let center_lat_rad = center_lat.to_radians();
    let cos_lat = center_lat_rad.cos();

    let x = (lon - center_lon).to_radians() * cos_lat * EARTH_RADIUS_M;
    let z = (lat - center_lat).to_radians() * EARTH_RADIUS_M;

    Vec3::new(x as f32, 0.0, -(z as f32))
}

pub fn nearest_point_on_segments(segments: &[RoadSegment], pos: Vec3) -> Vec3 {
    let mut best = Vec3::ZERO;
    let mut best_dist_sq = f32::INFINITY;

    for segment in segments {
        for window in segment.points.windows(2) {
            let a = window[0];
            let b = window[1];
            let ab = b - a;
            let ap = pos - a;
            let t = ap.dot(ab) / ab.dot(ab).max(f32::EPSILON);
            let t = t.clamp(0.0, 1.0);
            let projected = a + ab * t;
            let d = pos.distance_squared(projected);
            if d < best_dist_sq {
                best_dist_sq = d;
                best = projected;
            }
        }
    }

    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_sample_osm() {
        let data = load_osm_file("sample.osm").expect("failed to parse sample.osm");
        assert!(!data.roads.is_empty(), "expected at least one road segment");
        let total_points: usize = data.roads.iter().map(|s| s.points.len()).sum();
        assert!(total_points >= 4, "expected at least 4 road points");
        assert!(!data.buildings.is_empty(), "expected at least one building");
    }
}
