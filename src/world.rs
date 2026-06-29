use bevy::prelude::*;
use geojson::{GeoJson, Geometry, Value};

const NATURAL_EARTH_LAND_URL: &str =
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_land.geojson";
const NATURAL_EARTH_URBAN_URL: &str =
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_urban_areas.geojson";
const LAND_CACHE_FILE: &str = "ne_110m_land.geojson";
const URBAN_CACHE_FILE: &str = "ne_50m_urban_areas.geojson";
/// Degrees per world unit. The world (360° longitude) fits in roughly 2000 units.
pub const WORLD_PROJECTION_SCALE: f64 = 5.56;

#[derive(Debug, Clone)]
pub struct ContinentPolygon {
    pub points: Vec<Vec3>,
}

pub fn fetch_natural_earth_land() -> Result<Vec<ContinentPolygon>, String> {
    fetch_geojson_polygons(NATURAL_EARTH_LAND_URL, LAND_CACHE_FILE)
}

pub fn fetch_urban_areas() -> Result<Vec<ContinentPolygon>, String> {
    fetch_geojson_polygons(NATURAL_EARTH_URBAN_URL, URBAN_CACHE_FILE)
}

fn fetch_geojson_polygons(url: &str, cache_file: &str) -> Result<Vec<ContinentPolygon>, String> {
    let geojson_text = if let Ok(cached) = std::fs::read_to_string(cache_file) {
        cached
    } else {
        let response = ureq::get(url)
            .set("User-Agent", "trka-racing-game/0.1")
            .call()
            .map_err(|e| format!("Failed to download {url}: {e}"))?;

        let mut reader = response.into_reader();
        let mut file = std::fs::File::create(cache_file)
            .map_err(|e| format!("Failed to create cache file {cache_file}: {e}"))?;
        std::io::copy(&mut reader, &mut file)
            .map_err(|e| format!("Failed to write cache file {cache_file}: {e}"))?;

        std::fs::read_to_string(cache_file)
            .map_err(|e| format!("Failed to read cached {cache_file}: {e}"))?
    };

    parse_land_geojson(&geojson_text)
}

fn parse_land_geojson(text: &str) -> Result<Vec<ContinentPolygon>, String> {
    let geojson = text.parse::<GeoJson>().map_err(|e| e.to_string())?;

    let mut polygons = Vec::new();

    match geojson {
        GeoJson::FeatureCollection(collection) => {
            for feature in collection.features {
                if let Some(geometry) = feature.geometry {
                    extract_polygons_from_geometry(&geometry, &mut polygons);
                }
            }
        }
        GeoJson::Feature(feature) => {
            if let Some(geometry) = feature.geometry {
                extract_polygons_from_geometry(&geometry, &mut polygons);
            }
        }
        GeoJson::Geometry(geometry) => {
            extract_polygons_from_geometry(&geometry, &mut polygons);
        }
    }

    Ok(polygons)
}

fn extract_polygons_from_geometry(
    geometry: &Geometry,
    out: &mut Vec<ContinentPolygon>,
) {
    match &geometry.value {
        Value::Polygon(rings) => {
            if let Some(ring) = rings.first() {
                out.push(ring_to_polygon(ring));
            }
        }
        Value::MultiPolygon(polygons) => {
            for polygon in polygons {
                if let Some(ring) = polygon.first() {
                    out.push(ring_to_polygon(ring));
                }
            }
        }
        _ => {}
    }
}

fn ring_to_polygon(ring: &[Vec<f64>]) -> ContinentPolygon {
    let mut points: Vec<Vec3> = ring.iter().map(|pos| project_pos(pos)).collect();
    if points.len() >= 4 && points.first() == points.last() {
        points.pop();
    }
    ContinentPolygon { points }
}

fn project_pos(pos: &[f64]) -> Vec3 {
    if pos.len() < 2 {
        return Vec3::ZERO;
    }
    let lon = pos[0];
    let lat = pos[1];
    project_lon_lat(lon, lat)
}

pub fn project_lon_lat(lon: f64, lat: f64) -> Vec3 {
    Vec3::new(
        (lon * WORLD_PROJECTION_SCALE) as f32,
        0.0,
        -(lat * WORLD_PROJECTION_SCALE) as f32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_cached_natural_earth() {
        let polygons = fetch_natural_earth_land().expect("failed to load cached world data");
        assert!(!polygons.is_empty(), "expected at least one continent polygon");
        let total_points: usize = polygons.iter().map(|p| p.points.len()).sum();
        assert!(total_points > 100, "expected many continent points");
    }
}
