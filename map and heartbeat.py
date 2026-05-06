import streamlit as st
import streamlit_folium as st_folium
import folium
from folium import TileLayer
import pandas as pd
import time
import datetime
import json
import os
import math
from shapely.geometry import Polygon, LineString, Point, MultiPoint
from shapely.ops import unary_union

# ==================== 页面配置 ====================
st.set_page_config(page_title="南京科技职业学院 - 无人机导航系统", layout="wide")

st.markdown("""
<style>
.left-panel {background:#f8f9fa; padding:20px; border-radius:10px; height:95vh;}
</style>
""", unsafe_allow_html=True)

# ==================== 持久化 ====================
STATE_FILE = "ground_station_state.json"

def save_state():
    state = {
        "obstacles": st.session_state.obstacles,
        "draw_points": st.session_state.draw_points,
        "home_point": st.session_state.home_point,
        "waypoints": st.session_state.waypoints,
        "click_mode": st.session_state.click_mode,
        "latA": st.session_state.latA, "lngA": st.session_state.lngA,
        "latB": st.session_state.latB, "lngB": st.session_state.lngB,
        "heartbeat_data": st.session_state.heartbeat_data[-200:],
        "heartbeat_seq": st.session_state.heartbeat_seq,
        "heartbeat_running": st.session_state.heartbeat_running,
        "flight_status": st.session_state.flight_status,
        "flight_start_time": st.session_state.flight_start_time.isoformat() if st.session_state.flight_start_time else None,
        "flight_paused_duration": st.session_state.flight_paused_duration,
        "flight_speed": st.session_state.flight_speed,
        "safety_radius": st.session_state.safety_radius,
    }
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            data = json.load(f)
        if data.get("flight_start_time"):
            data["flight_start_time"] = datetime.datetime.fromisoformat(data["flight_start_time"])
        return data
    return {}

def ensure_session_state():
    defaults = {
        "obstacles": [], "draw_points": [], "home_point": [32.2335, 118.7475], "waypoints": [],
        "last_click": None, "click_mode": "障碍物圈选",
        "latA": 32.233500, "lngA": 118.747500, "latB": 32.233800, "lngB": 118.747900,
        "heartbeat_data": [], "heartbeat_seq": 0, "heartbeat_running": False,
        "seq": 0, "running": False,
        "flight_status": "idle", "flight_start_time": None, "flight_paused_duration": 0.0,
        "flight_speed": 8.5, "safety_radius": 5.0,
    }
    loaded = load_state()
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = loaded.get(k, v)

ensure_session_state()

# ==================== 坐标工具 ====================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def lonlat_to_xy(lon, lat, center_lon, center_lat):
    meter_per_deg_lat = 111320.0
    meter_per_deg_lon = 111320.0 * math.cos(math.radians(center_lat))
    x = (lon - center_lon) * meter_per_deg_lon
    y = (lat - center_lat) * meter_per_deg_lat
    return x, y

def xy_to_lonlat(x, y, center_lon, center_lat):
    meter_per_deg_lat = 111320.0
    meter_per_deg_lon = 111320.0 * math.cos(math.radians(center_lat))
    lon = center_lon + x / meter_per_deg_lon
    lat = center_lat + y / meter_per_deg_lat
    return lon, lat

# ==================== 贝塞尔弧线 ====================
def create_bezier_arc_path(path_pts, control_scale=0.5):
    if len(path_pts) < 3:
        return path_pts
    arc = [path_pts[0]]
    for i in range(1, len(path_pts)-1):
        p0, p1, p2 = path_pts[i-1], path_pts[i], path_pts[i+1]
        dx1, dy1 = p1[0]-p0[0], p1[1]-p0[1]
        dx2, dy2 = p2[0]-p1[0], p2[1]-p1[1]
        control_x = p1[0] - (dx1 + dx2) * control_scale
        control_y = p1[1] - (dy1 + dy2) * control_scale
        steps = 10
        for t in range(1, steps+1):
            t = t/10
            x = (1-t)**2 * p0[0] + 2*(1-t)*t * control_x + t**2 * p2[0]
            y = (1-t)**2 * p0[1] + 2*(1-t)*t * control_y + t**2 * p2[1]
            arc.append((x, y))
    arc.append(path_pts[-1])
    return arc

def quadratic_bezier(p0, p1, p2, steps=30):
    curve = []
    for i in range(steps+1):
        t = i / steps
        x = (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0]
        y = (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]
        curve.append((x, y))
    return curve

# ==================== 核心绕飞算法 ====================
def compute_avoid_path(latA, lngA, latB, lngB, fly_height, obstacles, safety_radius_m=5.0):
    if not obstacles:
        return {"left": [], "right": [], "shortest": []}

    center_lon, center_lat = lngA, latA
    start_xy = lonlat_to_xy(lngA, latA, center_lon, center_lat)
    end_xy = lonlat_to_xy(lngB, latB, center_lon, center_lat)

    # 构建安全缓冲区
    buffers = []
    for ob in obstacles:
        if fly_height >= ob.get("height", 0):
            continue
        pts = ob["points"]
        if len(pts) < 3:
            continue
        xy_pts = [lonlat_to_xy(p[0], p[1], center_lon, center_lat) for p in pts]
        if xy_pts[0] != xy_pts[-1]:
            xy_pts.append(xy_pts[0])
        poly = Polygon(xy_pts)
        buffers.append(poly.buffer(safety_radius_m))

    if not buffers:
        straight = [(latA, lngA), (latB, lngB)]
        return {"left": straight, "right": straight, "shortest": straight}

    merged = unary_union(buffers)
    direct_line = LineString([start_xy, end_xy])

    # 无冲突
    if not direct_line.intersects(merged):
        straight = [(latA, lngA), (latB, lngB)]
        return {"left": straight, "right": straight, "shortest": straight}

    # 获取精确缓冲区外边界
    boundary = merged.exterior
    coords = list(boundary.coords)

    # 交点
    intersection = boundary.intersection(direct_line)
    pts = []
    if isinstance(intersection, Point):
        pts = [intersection]
    elif isinstance(intersection, MultiPoint):
        pts = list(intersection.geoms)
    if len(pts) < 2:
        dx = end_xy[0] - start_xy[0]
        dy = end_xy[1] - start_xy[1]
        ext_line = LineString([(start_xy[0]-dx*0.01, start_xy[1]-dy*0.01),
                               (end_xy[0]+dx*0.01, end_xy[1]+dy*0.01)])
        intersection = boundary.intersection(ext_line)
        if isinstance(intersection, Point):
            pts = [intersection]
        elif isinstance(intersection, MultiPoint):
            pts = list(intersection.geoms)
        if len(pts) < 2:
            return {"left": [], "right": [], "shortest": []}

    pts.sort(key=lambda p: direct_line.project(p))
    entry_pt = pts[0]
    exit_pt = pts[-1]

    def nearest_idx(pt, coords):
        best, idx = float('inf'), 0
        for i, (x, y) in enumerate(coords):
            d = math.hypot(x - pt.x, y - pt.y)
            if d < best:
                best, idx = d, i
        return idx

    i_entry = nearest_idx(entry_pt, coords)
    i_exit = nearest_idx(exit_pt, coords)

    # 边界两段
    if i_entry <= i_exit:
        seg1 = coords[i_entry:i_exit+1]
        seg2 = coords[i_exit:] + coords[:i_entry+1]
    else:
        seg1 = coords[i_entry:] + coords[:i_exit+1]
        seg2 = coords[i_exit:i_entry+1]

    entry_xy = (entry_pt.x, entry_pt.y)
    exit_xy = (exit_pt.x, exit_pt.y)

    # 构建两个候选路径
    cand1 = [start_xy, entry_xy] + seg1[1:-1] + [exit_xy, end_xy]
    cand2 = [start_xy, entry_xy] + seg2[1:-1] + [exit_xy, end_xy]

    # 判定左右：取路径上远离端点的点（30%处）计算叉积
    dir_vec = (end_xy[0]-start_xy[0], end_xy[1]-start_xy[1])
    def side_of_point(pt):
        vx, vy = pt[0]-start_xy[0], pt[1]-start_xy[1]
        return dir_vec[0]*vy - dir_vec[1]*vx

    def path_side(path, fraction=0.3):
        # 取路径上大概 fraction 位置的点
        idx = int(len(path) * fraction)
        idx = min(idx, len(path)-1)
        return side_of_point(path[idx])

    s1 = path_side(cand1)
    s2 = path_side(cand2)
    if s1 > s2:
        left_raw, right_raw = cand1, cand2
    else:
        left_raw, right_raw = cand2, cand1

    # 贝塞尔平滑
    left_bezier = create_bezier_arc_path(left_raw)
    right_bezier = create_bezier_arc_path(right_raw)

    def is_safe(path, merged_obj):
        for x, y in path:
            pt = Point(x, y)
            if pt.within(merged_obj) or pt.distance(merged_obj) < 0.5:
                return False
        return True

    final_left = left_bezier if is_safe(left_bezier, merged) else left_raw
    final_right = right_bezier if is_safe(right_bezier, merged) else right_raw

    # ---------- 上绕弧线 ----------
    # 找到缓冲区边界上离直线有向距离 (叉积) 绝对值最大的点
    max_dist = 0
    best_pt = None
    for x, y in coords:
        d = abs(side_of_point((x, y)))
        if d > max_dist:
            max_dist = d
            best_pt = (x, y)

    # 方向：从该点向外继续延伸
    if best_pt is not None:
        # 该点的有向距离符号决定偏移方向
        s = side_of_point(best_pt)
        perp = (-dir_vec[1], dir_vec[0]) if s > 0 else (dir_vec[1], -dir_vec[0])
        perp_len = math.hypot(perp[0], perp[1])
        if perp_len > 0:
            perp = (perp[0]/perp_len, perp[1]/perp_len)

        # 控制点：在最远点的基础上继续向外偏移 20 米
        control_x = best_pt[0] + perp[0] * 20.0
        control_y = best_pt[1] + perp[1] * 20.0
        control_pt = (control_x, control_y)

        over_curve = quadratic_bezier(start_xy, control_pt, end_xy, steps=30)

        # 安全检查并迭代增加偏移
        attempt = 0
        while not is_safe(over_curve, merged) and attempt < 20:
            control_x += perp[0] * 5.0
            control_y += perp[1] * 5.0
            control_pt = (control_x, control_y)
            over_curve = quadratic_bezier(start_xy, control_pt, end_xy, steps=30)
            attempt += 1
    else:
        # 后备：直接用左/右中较短的一条
        if len(left_raw) < 2 or len(right_raw) < 2:
            over_curve = []
        else:
            def len_xy(p):
                return sum(math.hypot(p[i+1][0]-p[i][0], p[i+1][1]-p[i][1]) for i in range(len(p)-1))
            over_curve = left_raw if len_xy(left_raw) < len_xy(right_raw) else right_raw

    # 转经纬度
    def to_latlon(xy_list):
        return [(xy_to_lonlat(x, y, center_lon, center_lat)[1],
                 xy_to_lonlat(x, y, center_lon, center_lat)[0]) for x, y in xy_list]

    return {
        "left": to_latlon(final_left),
        "right": to_latlon(final_right),
        "shortest": to_latlon(over_curve)
    }

# ==================== 以下UI部分不变（与之前版本相同） ====================
# 为了简洁，这里省略之前相同的UI代码，请将上面函数替换到原有完整代码中即可。
