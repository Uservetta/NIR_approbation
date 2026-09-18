
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import folium
from folium import plugins
from streamlit_folium import st_folium
import plotly.graph_objects as go

from shapely.geometry import (
    Polygon, Point, LineString, shape, mapping, box
)
from shapely.ops import transform, unary_union, nearest_points
from shapely.validation import explain_validity
from pyproj import CRS, Transformer


# -----------------------------
# Configuration
# -----------------------------
st.set_page_config(
    page_title="COD Concept Generator — НИР-2",
    page_icon="▦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
NORMATIVES_PATH = BASE_DIR / "normatives.xlsx"

LAYER_BUILDINGS = "01_buildings_layer"
LAYER_TRANSPORT = "02_transport_layer"
LAYER_INFRA = "03_infrastructure_layer"
LAYER_RESTRICTIONS = "04_restrictions_layer"

# Palette is only for presentation. Semantic data are stored in GeoJSON attributes.
PALETTE = {
    "dc_building": "#4F46E5",
    "power_substation_gpp": "#F59E0B",
    "fuel_complex": "#EF4444",
    "kpp": "#10B981",
    "parking": "#94A3B8",
    "internal_roads": "#475569",
    "pedestrian_walkways": "#CBD5E1",
    "power": "#F97316",
    "telecom": "#06B6D4",
    "water": "#3B82F6",
    "restriction": "#EF4444",
    "site": "#0F172A",
    "usable": "#22C55E",
}

st.markdown("""
<style>
    .stApp { background: #F5F7FB; }
    .block-container { padding-top: 1.25rem; max-width: 1500px; }
    h1, h2, h3 { letter-spacing: -0.02em; }
    .hero {
        background: linear-gradient(135deg,#111827 0%,#1E293B 65%,#334155 100%);
        color: white; padding: 28px 32px; border-radius: 18px;
        margin-bottom: 18px; box-shadow: 0 10px 35px rgba(15,23,42,.12);
    }
    .hero h1 { color:white; margin:0 0 6px 0; font-size: 2.0rem; }
    .hero p { color:#CBD5E1; margin:0; }
    .small-note { color:#64748B; font-size:.86rem; }
    .status {
        padding: 12px 14px; border-radius: 12px; background:#FFFFFF;
        border:1px solid #E2E8F0; margin-bottom:10px;
    }
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #E2E8F0;
        padding: 14px; border-radius: 14px;
        box-shadow: 0 4px 16px rgba(15,23,42,.04);
    }
    .stButton button, .stDownloadButton button { border-radius:10px; }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Normative library
# -----------------------------
@st.cache_data
def load_normatives():
    if not NORMATIVES_PATH.exists():
        raise FileNotFoundError("Не найден normatives.xlsx рядом с app.py")
    df = pd.read_excel(NORMATIVES_PATH, sheet_name="Табл")
    return df


def norm_row(norms, element):
    r = norms[norms["element_name"].astype(str) == element]
    if r.empty:
        raise KeyError(f"В normatives.xlsx отсутствует элемент {element}")
    return r.iloc[0].to_dict()


# -----------------------------
# CRS / geometry helpers
# -----------------------------
def utm_crs_for_geom(geom_wgs84):
    c = geom_wgs84.centroid
    zone = int((c.x + 180) / 6) + 1
    return CRS.from_epsg(32600 + zone) if c.y >= 0 else CRS.from_epsg(32700 + zone)


def project_geom(geom, src, dst):
    tr = Transformer.from_crs(src, dst, always_xy=True)
    return transform(tr.transform, geom)


def inverse_project_geom(geom, src, dst):
    tr = Transformer.from_crs(src, dst, always_xy=True)
    return transform(tr.transform, geom)


def parse_geojson_polygon(raw_bytes):
    obj = json.loads(raw_bytes.decode("utf-8"))
    geoms = []
    if obj.get("type") == "FeatureCollection":
        for f in obj.get("features", []):
            if f.get("geometry"):
                geoms.append(shape(f["geometry"]))
    elif obj.get("type") == "Feature":
        geoms.append(shape(obj["geometry"]))
    else:
        geoms.append(shape(obj))

    polys = [g for g in geoms if g.geom_type in ("Polygon", "MultiPolygon")]
    if not polys:
        raise ValueError("Файл не содержит Polygon/MultiPolygon.")
    g = unary_union(polys)
    if g.geom_type == "MultiPolygon":
        g = max(g.geoms, key=lambda x: x.area)
    return g


def parse_geojson_any(raw_bytes):
    obj = json.loads(raw_bytes.decode("utf-8"))
    geoms = []
    if obj.get("type") == "FeatureCollection":
        for f in obj.get("features", []):
            if f.get("geometry"):
                geoms.append((shape(f["geometry"]), f.get("properties", {})))
    elif obj.get("type") == "Feature":
        geoms.append((shape(obj["geometry"]), obj.get("properties", {})))
    else:
        geoms.append((shape(obj), {}))
    return geoms


def make_rect(center, area, aspect=1.5):
    """Rectangle centered at center; long side / short side = aspect."""
    area = max(float(area), 1.0)
    w = math.sqrt(area / aspect)
    l = area / w
    return box(center.x - l/2, center.y - w/2, center.x + l/2, center.y + w/2)


def oriented_rect(center, area, angle_deg, aspect=1.5):
    p = make_rect(Point(0, 0), area, aspect)
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    def rot(x, y, z=None):
        return (x*ca-y*sa, x*sa+y*ca)
    p = transform(rot, p)
    return translate_geom(p, center.x, center.y)


def translate_geom(g, dx, dy):
    from shapely.affinity import translate
    return translate(g, xoff=dx, yoff=dy)


def rotate_geom(g, angle, origin):
    from shapely.affinity import rotate
    return rotate(g, angle, origin=origin, use_radians=False)


def boundary_point_near(poly, target):
    p, q = nearest_points(poly.boundary, target)
    return p


def farthest_boundary_point(poly, p):
    coords = list(poly.boundary.coords)
    pts = [Point(x, y) for x, y in coords]
    return max(pts, key=lambda q: q.distance(p))


def point_on_boundary(poly, fraction):
    return poly.boundary.interpolate(float(fraction) * poly.boundary.length)


# -----------------------------
# Optional OSM predictive GIS
# -----------------------------
def overpass_query(site_wgs84):
    minx, miny, maxx, maxy = site_wgs84.bounds
    # Small context around the site. Keep query compact for MVP.
    pad = 0.0015
    bbox = f"{miny-pad},{minx-pad},{maxy+pad},{maxx+pad}"
    q = f"""
    [out:json][timeout:20];
    (
      way["power"="line"]({bbox});
      way["power"="minor_line"]({bbox});
      way["industrial"]({bbox});
      way["landuse"="industrial"]({bbox});
    );
    out geom;
    """
    r = requests.post("https://overpass-api.de/api/interpreter", data=q, timeout=30)
    r.raise_for_status()
    return r.json()


def predictive_restrictions(site_wgs84, local_crs):
    restrictions = []
    try:
        data = overpass_query(site_wgs84)
    except Exception as e:
        return restrictions, f"OSM/Overpass недоступен: {e}"

    src = CRS.from_epsg(4326)
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        geom = el.get("geometry", [])
        if len(geom) < 2:
            continue
        coords = [(p["lon"], p["lat"]) for p in geom]
        line = LineString(coords)
        line_local = project_geom(line, src, local_crs)

        power = str(tags.get("voltage", ""))
        if tags.get("power") in ("line", "minor_line"):
            # Method description explicitly models 110 kV lines as a 20 m half-width keep-out.
            if "110" in power or "110000" in power:
                restrictions.append((
                    line_local.buffer(20.0),
                    {"restriction_type": "LEP_110kV", "source": "OSM/Overpass", "buffer_m": 20.0}
                ))
            continue

        if tags.get("industrial") or tags.get("landuse") == "industrial":
            # Predictive 50 m protection mask around nearby industrial objects.
            restrictions.append((
                line_local.buffer(50.0),
                {"restriction_type": "SZZ_predictive", "source": "OSM/Overpass", "buffer_m": 50.0}
            ))

    # Clip masks to the site vicinity later.
    return restrictions, None


# -----------------------------
# Entry points
# -----------------------------
def auto_entry_points(site):
    # UDS: first boundary point; in MVP it represents the closest hypothetical city-road connection.
    uds = point_on_boundary(site, 0.08)

    # Power A: near one end of boundary; B: maximum separation by chord.
    power_a = point_on_boundary(site, 0.25)
    power_b = farthest_boundary_point(site, power_a)

    # Telecom: opposite boundary anchors.
    telecom_a = point_on_boundary(site, 0.42)
    telecom_b = farthest_boundary_point(site, telecom_a)

    # Water: another boundary point, nearest to a transport corridor in full GIS version.
    water = point_on_boundary(site, 0.75)
    return {
        "entry_point_uds": uds,
        "entry_point_power_A": power_a,
        "entry_point_power_B": power_b,
        "entry_point_telecom_A": telecom_a,
        "entry_point_telecom_B": telecom_b,
        "entry_point_water": water,
    }


# -----------------------------
# Scenario / parameter stage
# -----------------------------
def calculate_parameters(site, usable, user_power_mw, norms):
    dc = norm_row(norms, "dc_building")
    fuel = norm_row(norms, "fuel_complex")
    sub = norm_row(norms, "power_substation_gpp")
    kpp = norm_row(norms, "kpp")
    parking = norm_row(norms, "parking")

    # Source-of-truth is the machine-readable library.
    dc_step = float(dc["area_coef"])
    if user_power_mw is None:
        # Report text: max power is limited by the useful footprint and then capped at 20 MW.
        # For the MVP we use the machine-readable dc area step.
        max_power_kw = usable.area / dc_step
        power_mw = min(max_power_kw / 1000.0, 20.0)
        scenario = "Максимальный потенциал"
    else:
        power_mw = float(user_power_mw)
        scenario = "Заданная мощность"

    pit_kw = power_mw * 1000.0
    s_dc = pit_kw * dc_step
    s_fuel = s_dc * float(fuel["area_coef"]) if fuel["calc_method"] == "by_dc_area" else float(fuel["area_coef"])
    s_sub = s_dc * float(sub["area_coef"]) if sub["calc_method"] == "by_dc_area" else float(sub["area_coef"])
    s_parking = s_dc * float(parking["area_coef"]) if parking["calc_method"] == "by_dc_area" else float(parking["area_coef"])
    s_kpp = float(kpp["area_coef"])

    # Method text: chiller 30% + ABK/self-needs 5%.
    p_total_mw = power_mw * 1.35
    pue = p_total_mw / power_mw if power_mw else np.nan
    n_racks = int(round(pit_kw / 5.0))

    s_build = s_dc + s_fuel + s_sub + s_kpp + s_parking
    k_build = s_build / site.area * 100.0

    return {
        "scenario": scenario,
        "power_mw": power_mw,
        "p_total_mw": p_total_mw,
        "pue": pue,
        "n_racks": n_racks,
        "s_dc": s_dc,
        "s_fuel": s_fuel,
        "s_sub": s_sub,
        "s_parking": s_parking,
        "s_kpp": s_kpp,
        "s_build": s_build,
        "k_build": k_build,
    }


# -----------------------------
# Layout / iterative buffer placement
# -----------------------------
def fit_inside(poly, container):
    """Translate polygon to container and then pull it toward centroid until contained."""
    if container.contains(poly):
        return poly
    c0 = poly.centroid
    c1 = container.centroid
    dx, dy = c1.x-c0.x, c1.y-c0.y
    candidate = translate_geom(poly, dx, dy)
    if container.contains(candidate):
        return candidate

    # Binary-search interpolation from original center to container centroid.
    lo, hi = 0.0, 1.0
    best = None
    for _ in range(50):
        t = (lo+hi)/2
        cand = translate_geom(poly, dx*t, dy*t)
        if container.contains(cand):
            best = cand
            lo = t
        else:
            hi = t
    return best if best is not None else candidate


def place_touching_facade(base, area, offset, side, angle):
    minx, miny, maxx, maxy = base.bounds
    if side == "left":
        cx = minx - offset
        cy = (miny+maxy)/2
    elif side == "right":
        cx = maxx + offset
        cy = (miny+maxy)/2
    elif side == "top":
        cx = (minx+maxx)/2
        cy = maxy + offset
    else:
        cx = (minx+maxx)/2
        cy = miny - offset
    p = oriented_rect(Point(cx, cy), area, angle, aspect=1.5)
    return p


def _required_gap(norms, name_a, name_b):
    """Conservative separation: max of the two table-driven safety offsets."""
    vals = []
    for name in (name_a, name_b):
        if name in norms["element_name"].values:
            vals.append(float(norm_row(norms, name)["buffer_m"]))
    return max(vals) if vals else 0.0


def _candidate_rects(container, area, angle, aspect=1.5, desired=None, steps=45):
    """Generate deterministic candidate rectangles over the usable polygon."""
    minx, miny, maxx, maxy = container.bounds
    xs = np.linspace(minx, maxx, steps)
    ys = np.linspace(miny, maxy, steps)
    candidates = []
    pts = [(x, y) for x in xs for y in ys]
    if desired is not None:
        pts.sort(key=lambda xy: Point(*xy).distance(desired))
    for x, y in pts:
        g = oriented_rect(Point(x, y), area, angle, aspect=aspect)
        if container.covers(g):
            candidates.append(g)
    return candidates


def _place_free(name, area, aspect, angle, usable, placed, norms, desired=None):
    """Place a component only where it does not intersect and has the required gap."""
    candidates = _candidate_rects(usable, area, angle, aspect=aspect, desired=desired)
    for g in candidates:
        ok = True
        for other_name, other in placed.items():
            if g.intersects(other):
                ok = False
                break
            gap = _required_gap(norms, name, other_name)
            if g.distance(other) + 1e-6 < gap:
                ok = False
                break
        if ok:
            return g
    return None


def place_layout(site, usable, params, entries, norms):
    # Main building at the useful-zone centroid. All remaining objects are
    # subsequently placed by a collision-aware search inside the useful zone.
    minx, miny, maxx, maxy = usable.bounds
    angle = 0 if (maxx - minx) >= (maxy - miny) else 90

    dc_poly = oriented_rect(usable.centroid, params["s_dc"], angle, aspect=1.5)
    dc_poly = fit_inside(dc_poly, usable)
    placed = {"dc_building": dc_poly}

    dc = dc_poly
    uds = entries["entry_point_uds"]
    pA = entries["entry_point_power_A"]
    # Desired zones are only preferences; the hard rule is non-overlap + safety gap.
    desired_sub = Point(dc.bounds[0] - 25, dc.centroid.y) if pA.x < dc.centroid.x else Point(dc.bounds[2] + 25, dc.centroid.y)
    desired_fuel = Point(dc.bounds[2] + 35, dc.centroid.y) if pA.x < dc.centroid.x else Point(dc.bounds[0] - 35, dc.centroid.y)
    desired_kpp = usable.centroid.buffer(1).centroid
    # Bias KPP toward the UDS entry.
    desired_kpp = Point(uds.x + (usable.centroid.x - uds.x) * 0.28,
                        uds.y + (usable.centroid.y - uds.y) * 0.28)

    sub_poly = _place_free(
        "power_substation_gpp", params["s_sub"], 1.5, angle, usable, placed, norms, desired_sub
    )
    fuel_poly = _place_free(
        "fuel_complex", params["s_fuel"], 1.5, angle, usable, {**placed, **({"power_substation_gpp": sub_poly} if sub_poly else {})}, norms, desired_fuel
    )
    if sub_poly is not None:
        placed["power_substation_gpp"] = sub_poly
    if fuel_poly is not None:
        placed["fuel_complex"] = fuel_poly

    kpp_poly = _place_free(
        "kpp", params["s_kpp"], 1.0, angle, usable, placed, norms, desired_kpp
    )
    if kpp_poly is not None:
        placed["kpp"] = kpp_poly

    desired_parking = kpp_poly.centroid if kpp_poly is not None else usable.centroid
    parking_poly = _place_free(
        "parking", params["s_parking"], 1.8, angle, usable, placed, norms, desired_parking
    )

    # Parking is visually important on the demo plan. If the first deterministic
    # grid cannot find a slot, retry with a denser angular/radial search before
    # declaring it unavailable. Every accepted candidate still obeys the same
    # non-overlap and safety-gap rules.
    if parking_poly is None:
        candidates = []
        center = desired_parking if desired_parking is not None else usable.centroid
        max_r = max(usable.bounds[2]-usable.bounds[0], usable.bounds[3]-usable.bounds[1])
        for r in np.linspace(0, max_r, 36):
            for a in np.linspace(0, 2*np.pi, 72, endpoint=False):
                c = Point(center.x + r*np.cos(a), center.y + r*np.sin(a))
                g = oriented_rect(c, params["s_parking"], angle, aspect=1.8)
                if not usable.covers(g):
                    continue
                ok = True
                for other_name, other in placed.items():
                    if g.intersects(other):
                        ok = False
                        break
                    if g.distance(other) + 1e-6 < _required_gap(norms, "parking", other_name):
                        ok = False
                        break
                if ok:
                    candidates.append(g)
        if candidates:
            parking_poly = min(candidates, key=lambda g: g.distance(desired_parking))

    objects = {
        "dc_building": dc_poly,
        "power_substation_gpp": sub_poly,
        "fuel_complex": fuel_poly,
        "kpp": kpp_poly,
        "parking": parking_poly,
    }

    invalid = []
    for name, g in objects.items():
        if g is None:
            invalid.append(f"{name}: не удалось разместить без пересечения с другими объектами")
            continue
        if not usable.covers(g):
            invalid.append(f"{name}: выходит за пределы полезного пятна")
        if not site.covers(g):
            invalid.append(f"{name}: выходит за пределы исходного участка")

    # Hard validation: zero overlap + table-driven safety gap.
    valid_objects = {k: v for k, v in objects.items() if v is not None}
    for i, (a, ga) in enumerate(valid_objects.items()):
        for b, gb in list(valid_objects.items())[i+1:]:
            if ga.intersects(gb):
                invalid.append(f"{a} ↔ {b}: геометрии пересекаются")
            required = _required_gap(norms, a, b)
            if ga.distance(gb) + 1e-6 < required:
                invalid.append(f"{a} ↔ {b}: межобъектный разрыв < {required:.1f} м")

    return objects, invalid


# -----------------------------
# Routing
# -----------------------------
def rounded_route(points, turn_radius=15.0):
    # MVP: preserve the method's radius parameter and create a readable orthogonal-ish route.
    # Actual road geometry remains a LINESTRING and is buffered separately for area accounting.
    if len(points) < 2:
        return LineString(points)
    out = [points[0]]
    for i in range(1, len(points)-1):
        a, b, c = points[i-1], points[i], points[i+1]
        # Keep point if segments are long enough; the radius is encoded in properties.
        out.append(b)
    out.append(points[-1])
    return LineString([(p.x, p.y) for p in out])


def build_routes(objects, entries, norms):
    dc = objects["dc_building"]
    fuel = objects["fuel_complex"]
    kpp = objects["kpp"]
    parking = objects["parking"]

    road_w = float(norm_row(norms, "internal_roads")["area_coef"])
    road_r = float(norm_row(norms, "internal_roads")["turn_radius_m"])
    walk_w = float(norm_row(norms, "pedestrian_walkways")["area_coef"])

    uds = entries["entry_point_uds"]
    # Road goes through KPP -> DC loading edge -> fuel complex.
    if kpp is None:
        kpp = dc.centroid.buffer(4)
    dc_access = nearest_points(dc.boundary, kpp)[0]
    fuel_access = nearest_points(fuel.boundary, kpp)[0] if fuel is not None else dc_access
    road = rounded_route([uds, kpp.centroid, dc_access, fuel_access], road_r)

    if parking is not None and not parking.is_empty:
        walk_1 = LineString([
            (kpp.centroid.x, kpp.centroid.y),
            (parking.centroid.x, parking.centroid.y),
        ])
    else:
        # Keep the transport layer valid even when parking could not be placed.
        walk_1 = LineString([
            (kpp.centroid.x, kpp.centroid.y),
            (dc_access.x, dc_access.y),
        ])
    walk_2 = LineString([
        (kpp.centroid.x, kpp.centroid.y),
        (dc_access.x, dc_access.y),
    ])

    return {
        "internal_roads": road,
        "pedestrian_walkways": [walk_1, walk_2],
        "road_width_m": road_w,
        "walk_width_m": walk_w,
        "turn_radius_m": road_r,
    }


def build_infrastructure(objects, entries):
    dc = objects["dc_building"]
    sub = objects["power_substation_gpp"]
    # Keep the demo drawable even if a restrictive input layer leaves no valid
    # GPP slot. The validation panel still reports the placement issue.
    if sub is None:
        sub = Point(dc.bounds[0] - 15, dc.centroid.y)

    dc_left = Point(dc.bounds[0], dc.centroid.y)
    dc_right = Point(dc.bounds[2], dc.centroid.y)

    return [
        ("entry_point_power_A", entries["entry_point_power_A"],
         LineString([(entries["entry_point_power_A"].x, entries["entry_point_power_A"].y),
                     (sub.centroid.x, sub.centroid.y)]), "power"),
        ("entry_point_power_B", entries["entry_point_power_B"],
         LineString([(entries["entry_point_power_B"].x, entries["entry_point_power_B"].y),
                     (sub.centroid.x, sub.centroid.y)]), "power"),
        ("entry_point_telecom_A", entries["entry_point_telecom_A"],
         LineString([(entries["entry_point_telecom_A"].x, entries["entry_point_telecom_A"].y),
                     (dc_left.x, dc_left.y)]), "telecom"),
        ("entry_point_telecom_B", entries["entry_point_telecom_B"],
         LineString([(entries["entry_point_telecom_B"].x, entries["entry_point_telecom_B"].y),
                     (dc_right.x, dc_right.y)]), "telecom"),
        ("entry_point_water", entries["entry_point_water"],
         LineString([(entries["entry_point_water"].x, entries["entry_point_water"].y),
                     (dc.centroid.x, dc.centroid.y)]), "water"),
    ]


# -----------------------------
# GeoJSON export
# -----------------------------
def feature(geom, props):
    return {
        "type": "Feature",
        "geometry": mapping(geom),
        "properties": props,
    }


def fc(features):
    return {"type": "FeatureCollection", "features": features}


def make_layers(site, usable, objects, routes, infra, restrictions, norms, entries):
    # Layer 01
    f1 = []
    for name, geom in objects.items():
        # Some constrained layouts can legitimately fail to place an object.
        # Do not serialize None into GeoJSON and do not call geom.area on None.
        if geom is None or geom.is_empty:
            continue
        r = norm_row(norms, name)
        f1.append(feature(geom, {
            "element_name": name,
            "element_name_ru": r["element_name_ru"],
            "qgis_layer_name": LAYER_BUILDINGS,
            "qgis_vector_color": PALETTE.get(name, "#64748B"),
            "height_attr": float(r["height_attr"]),
            "floors_max": int(r["floors_max"]),
            "safety_offset_m": float(r["buffer_m"]),
            "area_m2": round(geom.area, 2),
            "calc_method": r["calc_method"],
        }))

    # Layer 02
    f2 = [
        feature(routes["internal_roads"], {
            "element_name": "internal_roads",
            "element_name_ru": "Внутренние дороги",
            "qgis_layer_name": LAYER_TRANSPORT,
            "qgis_vector_color": PALETTE["internal_roads"],
            "width_m": routes["road_width_m"],
            "turn_radius_m": routes["turn_radius_m"],
            "height_attr": 0.0,
        })
    ]
    for i, g in enumerate(routes["pedestrian_walkways"], 1):
        f2.append(feature(g, {
            "element_name": "pedestrian_walkways",
            "element_name_ru": f"Тротуар {i}",
            "qgis_layer_name": LAYER_TRANSPORT,
            "qgis_vector_color": PALETTE["pedestrian_walkways"],
            "width_m": routes["walk_width_m"],
            "height_attr": 0.0,
        }))
    f2.append(feature(entries["entry_point_uds"], {
        "element_name": "entry_point_uds",
        "element_name_ru": "Точка подключения к УДС",
        "qgis_layer_name": LAYER_TRANSPORT,
        "qgis_vector_color": PALETTE["internal_roads"],
        "height_attr": 0.0,
    }))

    # Layer 03
    f3 = []
    for name, p, line, typ in infra:
        f3.append(feature(p, {
            "element_name": name,
            "qgis_layer_name": LAYER_INFRA,
            "qgis_vector_color": PALETTE[typ],
            "height_attr": 0.0,
            "utility_type": typ,
        }))
        f3.append(feature(line, {
            "element_name": f"{name}_route",
            "qgis_layer_name": LAYER_INFRA,
            "qgis_vector_color": PALETTE[typ],
            "height_attr": 0.0,
            "utility_type": typ,
        }))

    # Layer 04
    f4 = [
        feature(site.boundary, {
            "element_name": "site_fence",
            "element_name_ru": "Граница участка",
            "qgis_layer_name": LAYER_RESTRICTIONS,
            "qgis_vector_color": PALETTE["site"],
            "height_attr": 2.5,
            "restriction_type": "site_boundary",
        }),
        feature(usable, {
            "element_name": "usable_zone_polygon",
            "element_name_ru": "Полезное пятно застройки",
            "qgis_layer_name": LAYER_RESTRICTIONS,
            "qgis_vector_color": PALETTE["usable"],
            "height_attr": 0.0,
            "restriction_type": "usable_zone",
        }),
    ]
    for i, (g, meta) in enumerate(restrictions, 1):
        f4.append(feature(g, {
            "element_name": f"keepout_{i}",
            "qgis_layer_name": LAYER_RESTRICTIONS,
            "qgis_vector_color": PALETTE["restriction"],
            "height_attr": 0.0,
            **meta,
        }))

    return {
        LAYER_BUILDINGS: fc(f1),
        LAYER_TRANSPORT: fc(f2),
        LAYER_INFRA: fc(f3),
        LAYER_RESTRICTIONS: fc(f4),
    }


def geojson_bytes(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")


def zip_layers(layers):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for layer_name, obj in layers.items():
            z.writestr(f"{layer_name}.geojson", geojson_bytes(obj))
    return buf.getvalue()


def tep_dataframe(site, usable, params, routes):
    paved = routes["internal_roads"].buffer(routes["road_width_m"]/2.0).union(
        unary_union([g.buffer(routes["walk_width_m"]/2.0) for g in routes["pedestrian_walkways"]])
    )
    parking_geom = objects_global.get("parking")
    if parking_geom is not None and not parking_geom.is_empty:
        paved = paved.union(parking_geom)
    tep = [
        ("Общая площадь (Sуч)", "м²", site.area),
        ("Полезная площадь (Sполезн)", "м²", usable.area),
        ("Площадь застройки (Sзаст)", "м²", params["s_build"]),
        ("Количество этажей (Nэт)", "шт", 1),
        ("Площадь твердых покрытий и проездов (Sпокрыт)", "м²", paved.area),
        ("Коэффициент застройки (Кзаст)", "%", params["k_build"]),
        ("Общая мощность электроснабжения (Pобщ)", "МВт", params["p_total_mw"]),
        ("Общая доступная мощность электроснабжения (PIT)", "МВт", params["power_mw"]),
        ("Коэффициент энергоэффективности (PUE)", "число", params["pue"]),
        ("Общее количество стоек (Nстойки)", "шт", params["n_racks"]),
    ]
    return pd.DataFrame(tep, columns=["Показатель", "Ед. изм.", "Значение"])


# -----------------------------
# 3D
# -----------------------------
def prism_mesh(poly, height, z0=0):
    coords = list(poly.exterior.coords)[:-1]
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    n = len(coords)
    x = xs + xs
    y = ys + ys
    z = [z0]*n + [z0+height]*n
    i, j, k = [], [], []
    for a in range(1, n-1):
        i.append(0); j.append(a); k.append(a+1)
        i.append(n); j.append(n+a+1); k.append(n+a)
    for a in range(n):
        b = (a+1) % n
        i += [a, a, n+a, n+a]
        j += [b, n+a, n+b, b]
        k += [n+b, b, b, n+b]
    return x, y, z, i, j, k


def _rgb(hex_color):
    h = hex_color.lstrip("#")
    return [int(h[i:i+2], 16) for i in (0, 2, 4)] + [220]


def _geojson_feature(geom, props=None):
    return {"type": "Feature", "geometry": mapping(geom), "properties": props or {}}


def make_3d(objects, site, routes, infra, local_crs, norms):
    """Presentation 3D: real geographic coordinates + key site layers.

    Uses MapLibre/CARTO's public style URL rather than a token-bound Mapbox/CARTO
    provider, so the scene has a real basemap without an API-key watermark.
    """
    import pydeck as pdk

    site_w = wgs_geom(site, local_crs)
    center = site_w.centroid

    # Building features are geographic polygons with semantic height.
    building_features = []
    for name, geom in objects.items():
        if geom is None:
            continue
        r = norm_row(norms, name)
        building_features.append(_geojson_feature(wgs_geom(geom, local_crs), {
            "name": r["element_name_ru"],
            "height": float(r["height_attr"]),
            "color": _rgb(PALETTE.get(name, "#64748B")),
            "area": round(geom.area, 1),
        }))

    site_feature = _geojson_feature(site_w, {"name": "Участок"})
    usable_feature = _geojson_feature(wgs_geom(site.buffer(-1.0), local_crs), {"name": "Участок"}) if site.area > 100 else None

    # Routes + engineering networks as visible 3D paths.
    road_w = wgs_geom(routes["internal_roads"], local_crs)
    path_rows = [{
        "path": [[x, y] for x, y in road_w.coords],
        "color": _rgb(PALETTE["internal_roads"]), "width": 8, "name": "Внутренние дороги"
    }]
    for g in routes["pedestrian_walkways"]:
        gw = wgs_geom(g, local_crs)
        path_rows.append({"path": [[x, y] for x, y in gw.coords], "color": _rgb(PALETTE["pedestrian_walkways"]), "width": 4, "name": "Тротуар"})
    for name, p, line, typ in infra:
        lw = wgs_geom(line, local_crs)
        path_rows.append({"path": [[x, y] for x, y in lw.coords], "color": _rgb(PALETTE[typ]), "width": 5, "name": name})

    # A subtle site footprint keeps the composition readable even when the map is busy.
    site_layer = pdk.Layer(
        "GeoJsonLayer", data={"type": "FeatureCollection", "features": [site_feature]},
        id="site-footprint", stroked=True, filled=True,
        get_fill_color=[34, 197, 94, 25], get_line_color=[15, 23, 42, 190],
        line_width_min_pixels=2, pickable=False,
    )
    building_layer = pdk.Layer(
        "GeoJsonLayer", data={"type": "FeatureCollection", "features": building_features},
        id="buildings-3d", stroked=True, filled=True, extruded=True, wireframe=True,
        get_elevation="properties.height", elevation_scale=1.0,
        get_fill_color="properties.color", get_line_color=[255, 255, 255, 210],
        line_width_min_pixels=1, pickable=True,
    )
    road_layer = pdk.Layer(
        "PathLayer", data=path_rows, id="site-routes", get_path="path",
        get_color="color", get_width="width", width_min_pixels=2, width_max_pixels=8,
        pickable=True, rounded=True,
    )

    # Engineering connection nodes.
    points = []
    for name, p, line, typ in infra:
        pw = wgs_geom(p, local_crs)
        points.append({"position": [pw.x, pw.y], "color": _rgb(PALETTE[typ]), "name": name})
    point_layer = pdk.Layer(
        "ScatterplotLayer", data=points, get_position="position", get_fill_color="color",
        get_radius=8, radius_min_pixels=5, radius_max_pixels=10, pickable=True,
    )

    view = pdk.ViewState(
        latitude=center.y, longitude=center.x, zoom=16.2,
        pitch=52, bearing=-18,
    )
    return pdk.Deck(
        layers=[site_layer, road_layer, building_layer, point_layer],
        initial_view_state=view,
        map_provider="maplibre",
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        tooltip={"html": "<b>{name}</b><br/>Высота: {height} м", "style": {"backgroundColor": "#111827", "color": "white"}},
    )


# -----------------------------
# Map visualization
# -----------------------------
def wgs_geom(local_geom, local_crs):
    return inverse_project_geom(local_geom, local_crs, CRS.from_epsg(4326))


def add_geojson_map(m, geom, style, tooltip=None):
    gj = folium.GeoJson(
        data=mapping(geom),
        style_function=lambda _: style,
        tooltip=tooltip,
    )
    gj.add_to(m)


def build_map(site, usable, objects, routes, infra, restrictions, local_crs):
    site_w = wgs_geom(site, local_crs)
    usable_w = wgs_geom(usable, local_crs)
    center = [site_w.centroid.y, site_w.centroid.x]

    m = folium.Map(
        location=center, zoom_start=16,
        tiles="OpenStreetMap",
        control_scale=True,
        prefer_canvas=True,
    )

    add_geojson_map(m, site_w, {
        "color": PALETTE["site"], "weight": 3, "fill": False
    }, "Граница участка")

    add_geojson_map(m, usable_w, {
        "color": PALETTE["usable"], "weight": 2,
        "fillColor": PALETTE["usable"], "fillOpacity": 0.08
    }, "Полезное пятно")

    for name, g in objects.items():
        if g is None or g.is_empty:
            continue
        gw = wgs_geom(g, local_crs)
        r = norm_row(norms_global, name)
        folium.GeoJson(
            mapping(gw),
            name=r["element_name_ru"],
            style_function=lambda _, c=PALETTE.get(name, "#64748B"): {
                "color": c, "weight": 2, "fillColor": c, "fillOpacity": .48
            },
            # No GeoJsonTooltip here: the object is supplied as a raw geometry
            # (not a Feature with a properties dict), and Folium expects
            # properties to exist when a GeoJsonTooltip is rendered.
            # The object name is shown by the persistent map label below.
        ).add_to(m)

        c = gw.centroid
        folium.Marker(
            [c.y, c.x],
            icon=folium.DivIcon(
                html=f'<div style="font-size:10px;font-weight:700;color:#111827;white-space:nowrap;'
                     f'text-shadow:0 1px 2px white">{r["element_name_ru"]}</div>'
            )
        ).add_to(m)

    rw = wgs_geom(routes["internal_roads"], local_crs)
    folium.GeoJson(mapping(rw), style_function=lambda _: {
        "color": PALETTE["internal_roads"], "weight": 6, "opacity": .9
    }, name="Внутренние дороги").add_to(m)

    for g in routes["pedestrian_walkways"]:
        gw = wgs_geom(g, local_crs)
        folium.GeoJson(mapping(gw), style_function=lambda _: {
            "color": PALETTE["pedestrian_walkways"], "weight": 3, "opacity": .95
        }).add_to(m)

    for name, p, line, typ in infra:
        pw = wgs_geom(p, local_crs)
        lw = wgs_geom(line, local_crs)
        folium.CircleMarker(
            [pw.y, pw.x], radius=5, color=PALETTE[typ],
            fill=True, fill_opacity=.95,
            tooltip=name
        ).add_to(m)
        folium.GeoJson(mapping(lw), style_function=lambda _, c=PALETTE[typ]: {
            "color": c, "weight": 3, "dashArray": "6 4"
        }).add_to(m)

    for g, meta in restrictions:
        gw = wgs_geom(g, local_crs)
        folium.GeoJson(mapping(gw), style_function=lambda _: {
            "color": PALETTE["restriction"], "weight": 1,
            "fillColor": PALETTE["restriction"], "fillOpacity": .14
        }, tooltip=meta.get("restriction_type", "Ограничение")).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[site_w.bounds[1], site_w.bounds[0]],
                  [site_w.bounds[3], site_w.bounds[2]]])
    return m


# -----------------------------
# App state / UI
# -----------------------------
try:
    norms_global = load_normatives()
except Exception as e:
    st.error(str(e))
    st.stop()

st.markdown("""
<div class="hero">
  <h1>Автоматизированное формирование концепции территории ЦОД</h1>
  <p>НИР-2 · MVP на Streamlit · параметрический пространственный синтез · экспорт для QGIS</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("1 · Исходные данные")
    st.caption("Метод ограничен Greenfield, капитальным коммерческим ЦОД уровня Tier III, площадью до 3 га.")

    mode = st.radio(
        "Источник геометрии",
        ["Нарисовать на карте", "Загрузить GeoJSON"],
        index=0
    )

    center_lat = st.number_input("Центр карты · широта", value=59.934280, format="%.6f")
    center_lon = st.number_input("Центр карты · долгота", value=30.335099, format="%.6f")

    site_upload = None
    if mode == "Загрузить GeoJSON":
        site_upload = st.file_uploader(
            "selected_site.geojson",
            type=["geojson", "json"],
            help="Polygon / MultiPolygon в WGS84."
        )

    st.divider()
    power_choice = st.radio(
        "Сценарий мощности",
        ["Заданная мощность", "Максимальный потенциал"],
        index=0
    )
    user_power = None
    if power_choice == "Заданная мощность":
        user_power = st.number_input(
            "PIT — ИТ-мощность, МВт",
            min_value=0.25, max_value=20.0, value=5.0, step=0.25,
            help="По методу: не менее 0,25 МВт."
        )

    st.divider()
    st.header("2 · ЗОУИТ")
    zouit_upload = st.file_uploader(
        "user_zouit_layer.geojson",
        type=["geojson", "json"],
        help="Необязательный FeatureCollection с Polygon/MultiPolygon."
    )
    use_osm = st.checkbox(
        "Если ЗОУИТ не загружены — запросить OSM/Overpass",
        value=False,
        help="Может потребовать доступа в интернет. Для защиты можно выключить и показать детерминированный результат."
    )

    st.divider()
    st.header("3 · Запуск")
    demo = st.button("Создать демонстрационный участок", use_container_width=True)
    run = st.button("▶ Сформировать концепцию", type="primary", use_container_width=True)

# Drawing map
site_wgs = None

if "demo_site" not in st.session_state:
    st.session_state.demo_site = None

if demo:
    # ~1.5 ha rectangular test site around the chosen center.
    lat = center_lat
    lon = center_lon
    dlat = 0.00055
    dlon = 0.0009 / max(math.cos(math.radians(lat)), 0.2)
    st.session_state.demo_site = Polygon([
        (lon-dlon, lat-dlat),
        (lon+dlon, lat-dlat),
        (lon+dlon, lat+dlat),
        (lon-dlon, lat+dlat),
        (lon-dlon, lat-dlat),
    ])

if mode == "Загрузить GeoJSON" and site_upload:
    try:
        site_wgs = parse_geojson_polygon(site_upload.getvalue())
    except Exception as e:
        st.error(f"Ошибка selected_site: {e}")
        st.stop()
elif st.session_state.demo_site is not None:
    site_wgs = st.session_state.demo_site
else:
    draw_map = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=16,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    folium.plugins.Draw(
        export=False,
        draw_options={
            "polyline": False, "rectangle": True, "circle": False,
            "marker": False, "circlemarker": False, "polygon": True
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(draw_map)
    draw_state = st_folium(
        draw_map, width=None, height=510, key="site_draw"
    )
    if draw_state and draw_state.get("last_active_drawing"):
        try:
            site_wgs = shape(draw_state["last_active_drawing"]["geometry"])
        except Exception:
            pass

st.markdown("### Ввод / проверка участка")

if site_wgs is None:
    st.info("Нарисуйте Polygon на карте или загрузите selected_site.geojson. Для быстрой демонстрации нажмите «Создать демонстрационный участок».")
    st.stop()

if site_wgs.geom_type == "MultiPolygon":
    site_wgs = max(site_wgs.geoms, key=lambda x: x.area)

if not site_wgs.is_valid:
    st.warning(f"Геометрия невалидна: {explain_validity(site_wgs)}. Попробуйте перерисовать полигон.")

local_crs = utm_crs_for_geom(site_wgs)
site = project_geom(site_wgs, CRS.from_epsg(4326), local_crs)
area = site.area

c1, c2, c3, c4 = st.columns(4)
c1.metric("Sуч", f"{area:,.0f} м²")
c2.metric("Sуч", f"{area/10000:.2f} га")
c3.metric("Ограничение метода", "≤ 3.00 га")
c4.metric("CRS расчёта", local_crs.to_string().replace("EPSG:", "EPSG "))

if area > 30000:
    st.error("Расчёт заблокирован: площадь участка превышает жёсткое ограничение метода 3 га.")
    st.stop()
if area <= 0:
    st.error("Площадь участка должна быть положительной.")
    st.stop()

# Restrictions
restrictions = []
if zouit_upload:
    try:
        for g, props in parse_geojson_any(zouit_upload.getvalue()):
            if g.geom_type in ("Polygon", "MultiPolygon"):
                gl = project_geom(g, CRS.from_epsg(4326), local_crs)
                restrictions.append((gl, {
                    "restriction_type": props.get("restriction_type", "user_ZOUIT"),
                    "source": "user_zouit_layer",
                    "buffer_m": 0.0,
                }))
    except Exception as e:
        st.error(f"Ошибка ЗОУИТ: {e}")
        st.stop()
elif use_osm:
    with st.spinner("OSM/Overpass: поиск ЛЭП и промышленного окружения..."):
        restrictions, osm_error = predictive_restrictions(site_wgs, local_crs)
    if osm_error:
        st.warning(osm_error)

# Useful footprint = site minus restrictions.
if restrictions:
    mask = unary_union([g for g, _ in restrictions])
    usable = site.difference(mask)
else:
    usable = site

if usable.is_empty:
    st.error("Полезное пятно застройки равно нулю: зоны ограничений полностью перекрыли участок.")
    st.stop()

if usable.geom_type == "MultiPolygon":
    usable = max(usable.geoms, key=lambda x: x.area)

entries = auto_entry_points(site)

st.markdown("### Параметрический этап")
if power_choice == "Заданная мощность" and user_power is not None and user_power < 0.25:
    st.error("PIT должна быть не менее 0,25 МВт.")
    st.stop()

params = calculate_parameters(
    site, usable,
    user_power if power_choice == "Заданная мощность" else None,
    norms_global
)

p1, p2, p3, p4, p5 = st.columns(5)
p1.metric("Сценарий", params["scenario"])
p2.metric("PIT", f'{params["power_mw"]:.2f} МВт')
p3.metric("Pобщ", f'{params["p_total_mw"]:.2f} МВт')
p4.metric("PUE", f'{params["pue"]:.2f}')
p5.metric("Стойки", f'{params["n_racks"]:,}')

# Generate only on explicit click, but keep a demo fallback for smooth interaction.
if run or "result" not in st.session_state:
    objects, validation_errors = place_layout(
        site, usable, params, entries, norms_global
    )
    routes = build_routes(objects, entries, norms_global)
    infra = build_infrastructure(objects, entries)

    # Global result stored for download tabs.
    st.session_state.result = {
        "site": site, "usable": usable,
        "objects": objects, "routes": routes,
        "infra": infra, "restrictions": restrictions,
        "entries": entries, "params": params,
        "validation_errors": validation_errors,
        "local_crs": local_crs,
    }

if "result" not in st.session_state:
    st.stop()

res = st.session_state.result
objects_global = res["objects"]

# Recalculate after possible session reuse.
layers = make_layers(
    res["site"], res["usable"], res["objects"], res["routes"],
    res["infra"], res["restrictions"], norms_global, res["entries"]
)
tep = tep_dataframe(
    res["site"], res["usable"], res["params"], res["routes"]
)

# Validation summary
if res["validation_errors"]:
    st.warning("Компоновка сформирована, но глобальная пространственная валидация обнаружила коллизии:")
    for err in res["validation_errors"]:
        st.write("• " + err)
else:
    st.success("Глобальная пространственная валидация: коллизий не обнаружено.")

missing_objects = [
    name for name, geom in res["objects"].items()
    if geom is None or geom.is_empty
]
if missing_objects:
    st.info(
        "Некоторые объекты не удалось разместить в пределах полезного пятна "
        "с заданными разрывами: " + ", ".join(missing_objects) +
        ". Остальные слои и визуализация сформированы корректно."
    )

st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺 Планировочная схема",
    "🏢 3D-модель",
    "📊 ТЭП",
    "📦 Экспорт для QGIS",
])

with tab1:
    map_obj = build_map(
        res["site"], res["usable"], res["objects"],
        res["routes"], res["infra"], res["restrictions"], res["local_crs"]
    )
    st_folium(map_obj, width=None, height=680, key="result_map")

with tab2:
    st.caption("3D-визуализация: географическая подложка, выдавливание зданий по height_attr, внутренние дороги и инженерные связи.")
    deck = make_3d(res["objects"], res["site"], res["routes"], res["infra"], res["local_crs"], norms_global)
    st.pydeck_chart(deck, use_container_width=True, height=700)

with tab3:
    st.dataframe(
        tep.style.format({"Значение": lambda x: f"{x:,.2f}".replace(",", " ")}),
        use_container_width=True,
        hide_index=True,
    )
    # Compact cards for screenshot.
    a, b, c = st.columns(3)
    a.metric("Sполезн", f'{res["usable"].area:,.0f} м²')
    b.metric("Sзаст", f'{res["params"]["s_build"]:,.0f} м²')
    c.metric("Кзаст", f'{res["params"]["k_build"]:.1f} %')

with tab4:
    st.markdown("#### 4 отдельных GeoJSON-слоя")
    st.caption("Каждый файл можно добавить в QGIS как отдельный векторный слой. В свойствах объектов сохранены qgis_layer_name, qgis_vector_color и height_attr.")
    cols = st.columns(4)
    for i, (layer_name, obj) in enumerate(layers.items()):
        with cols[i]:
            st.download_button(
                f"Скачать {layer_name}",
                data=geojson_bytes(obj),
                file_name=f"{layer_name}.geojson",
                mime="application/geo+json",
                use_container_width=True,
                key=f"dl_{layer_name}",
            )

    st.download_button(
        "⬇ Скачать ZIP: 4 GeoJSON-слоя",
        data=zip_layers(layers),
        file_name="Layers_Campus_4_layers.zip",
        mime="application/zip",
        type="primary",
        use_container_width=True,
    )

    xlsx_buf = io.BytesIO()
    with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
        tep.to_excel(writer, sheet_name="TEP_Data", index=False)
        pd.DataFrame([
            ["Сценарий", res["params"]["scenario"]],
            ["PIT, МВт", res["params"]["power_mw"]],
            ["Pобщ, МВт", res["params"]["p_total_mw"]],
            ["PUE", res["params"]["pue"]],
            ["Nстойки", res["params"]["n_racks"]],
            ["Sуч, м²", res["site"].area],
            ["Sполезн, м²", res["usable"].area],
            ["Sзаст, м²", res["params"]["s_build"]],
            ["Кзаст, %", res["params"]["k_build"]],
        ], columns=["Параметр", "Значение"]).to_excel(
            writer, sheet_name="Scenario", index=False
        )
        norms_global.to_excel(writer, sheet_name="Normatives_used", index=False)

        # Presentation-ready formatting.
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter
        thin = Side(style="thin", color="D9E2EC")
        header_fill = PatternFill("solid", fgColor="111827")
        accent_fill = PatternFill("solid", fgColor="E8EEF7")
        white_font = Font(color="FFFFFF", bold=True)
        bold_font = Font(bold=True)

        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.fill = header_fill
                cell.font = white_font
                cell.alignment = Alignment(horizontal="center", vertical="center")
                cell.border = Border(bottom=thin)
            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    cell.border = Border(bottom=thin)
                    cell.alignment = Alignment(vertical="center", wrap_text=True)
            for col_idx in range(1, ws.max_column + 1):
                max_len = 0
                for row_idx in range(1, min(ws.max_row, 80) + 1):
                    val = ws.cell(row_idx, col_idx).value
                    max_len = max(max_len, len(str(val)) if val is not None else 0)
                ws.column_dimensions[get_column_letter(col_idx)].width = min(max(max_len + 2, 12), 42)
            ws.row_dimensions[1].height = 28

        ws = writer.book["TEP_Data"]
        ws.column_dimensions["A"].width = 58
        ws.column_dimensions["B"].width = 14
        ws.column_dimensions["C"].width = 20
        for cell in ws["C"][1:]:
            cell.number_format = '#,##0.00'
        ws["A1"].font = white_font

        ws = writer.book["Scenario"]
        ws.column_dimensions["A"].width = 30
        ws.column_dimensions["B"].width = 22
        for cell in ws["B"][1:]:
            if isinstance(cell.value, (int, float)):
                cell.number_format = '#,##0.00'

        ws = writer.book["Normatives_used"]
        ws.freeze_panes = "A2"

    st.download_button(
        "⬇ Скачать TEP_Data.xlsx",
        data=xlsx_buf.getvalue(),
        file_name="TEP_Data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        use_container_width=True,
    )

    st.download_button(
        "⬇ Скачать единый Layers_Campus.geojson",
        data=geojson_bytes({
            "type": "FeatureCollection",
            "features": sum(
                [v["features"] for v in layers.values()], []
            )
        }),
        file_name="Layers_Campus.geojson",
        mime="application/geo+json",
        use_container_width=True,
    )

st.markdown("---")
with st.expander("Машиночитаемая библиотека · normatives.xlsx"):
    st.dataframe(norms_global, use_container_width=True, hide_index=True)

with st.expander("Контрольные параметры реализации"):
    st.write({
        "CRS": res["local_crs"].to_string(),
        "S_total_m2": round(res["site"].area, 2),
        "S_usable_m2": round(res["usable"].area, 2),
        "power_mw": round(res["params"]["power_mw"], 4),
        "P_total_mw": round(res["params"]["p_total_mw"], 4),
        "PUE": round(res["params"]["pue"], 4),
        "n_racks": res["params"]["n_racks"],
        "objects": {k: (round(v.area, 2) if v is not None and not v.is_empty else None) for k, v in res["objects"].items()},
        "layers": list(layers.keys()),
    })

st.caption(
    "MVP воспроизводит описанный в НИР-2 конвейер: валидация → полезное пятно → "
    "сценарий мощности → расчёт площадей по normatives.xlsx → компоновка → "
    "трассировка → ТЭП → 4 GeoJSON-слоя + XLSX + 3D."
)
