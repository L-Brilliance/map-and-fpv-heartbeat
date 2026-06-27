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
import numpy as np
from shapely.geometry import Polygon, LineString, Point, MultiPoint
from shapely.ops import unary_union
# 新增拓扑图与MAV解析库
from streamlit_echarts import st_echarts
from pymavlink import mavutil

# ==================== 页面配置 ====================
st.set_page_config(page_title="南京科技职业学院 - 无人机导航系统", layout="wide")

st.markdown("""
<style>
.left-panel {background:#f8f9fa; padding:20px; border-radius:10px; height:95vh;}
.log-box {background-color:#1E1E1E; color:#FFFFFF; padding:12px; border-radius:8px; height:320px; overflow-y:auto; font-family:monospace; font-size:14px; line-height:1.5;}
.log-yellow {color:#FFC107;}
.log-blue {color:#03A9F4;}
.log-green {color:#4CAF50;}
.log-gray {color:#90A4AE;}
.log-time {color:#999;}
/* 新增拓扑、MAV面板样式 */
.mav-box {background:#0a101f; color:#00ff99; padding:15px; border-radius:8px; max-height:400px; overflow:auto; font-family:Consolas; font-size:13px;}
.topology-card {background:#f0f7ff; padding:16px; border-radius:10px; margin-bottom:20px;}
</style>
""", unsafe_allow_html=True)

STATE_FILE = "ground_station_state.json"

def save_state():
    state = {
        "obstacles": st.session_state.obstacles,
        "draw_points": st.session_state.draw_points,
        "home_point": st.session_state.home_point,
        "waypoints": st.session_state.waypoints,
        "click_mode": st.session_state.click_mode,
        "latA": st.session_state.latA,
        "lngA": st.session_state.lngA,
        "latB": st.session_state.latB,
        "lngB": st.session_state.lngB,
        "heartbeat_data": st.session_state.heartbeat_data[-100:],
        "seq": st.session_state.seq,
        "running": st.session_state.running,
        "flight_status": st.session_state.flight_status,
        "flight_start_time": st.session_state.flight_start_time,
        "safety_radius": st.session_state.safety_radius,
        "tx_logs": st.session_state.get("tx_logs", []),
        "rx_logs": st.session_state.get("rx_logs", []),
        "operate_logs": st.session_state.get("operate_logs", []),
        "last_wp": -1,
        # 新增MAV报文持久化
        "mavlink_packets": st.session_state.get("mavlink_packets", [])[-50:]
    }
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    return {}

def ensure_session_state():
    defaults = {
        "obstacles": [],
        "draw_points": [],
        "home_point": [32.2335, 118.7475],
        "waypoints": [],
        "last_click": None,
        "click_mode": "障碍物圈选",
        "latA": 32.233500,
        "lngA": 118.747500,
        "latB": 32.233800,
        "lngB": 118.747900,
        "heartbeat_data": [],
        "seq": 0,
        "running": False,
        "flight_status": "idle",
        "flight_start_time": None,
        "flight_speed": 8.5,
        "safety_radius": 5.0,
        "elapsed_flight": 0.0,
        "tx_logs": [],
        "rx_logs": [],
        "operate_logs": [],
        "last_wp": -1,
        # 新增会话变量
        "mavlink_packets": [],
    }
    loaded = load_state()
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = loaded.get(key, default_value)
    st.session_state.flight_speed = 8.5
    if "init" not in st.session_state:
        st.session_state.init = True

ensure_session_state()

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# ==================== 日志工具 ====================
def add_operate_log(msg):
    t = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    log = f"[{t}] <span class='log-gray'>{msg}</span>"
    st.session_state.operate_logs.append(log)
    st.session_state.operate_logs = st.session_state.operate_logs[-40:]

def add_tx_log(msg):
    t = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    log = f"[{t}] <span class='log-blue'>{msg}</span>"
    st.session_state.tx_logs.append(log)
    st.session_state.tx_logs = st.session_state.tx_logs[-30:]

def add_rx_log(msg):
    t = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    log = f"[{t}] <span class='log-yellow'>{msg}</span>"
    st.session_state.rx_logs.append(log)
    st.session_state.rx_logs = st.session_state.rx_logs[-30:]

# ==================== 【新增1】MAVLink报文管理函数 ====================
def add_mav_packet(packet_type, direction, raw_msg):
    """存储MAV原始报文，tx=上行GCS→FCU rx=下行FCU→GCS"""
    t = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    pkt_info = {
        "time": t,
        "dir": direction,
        "type": packet_type,
        "content": str(raw_msg)
    }
    st.session_state.mavlink_packets.append(pkt_info)
    st.session_state.mavlink_packets = st.session_state.mavlink_packets[-50:]
    save_state()

def render_mavlink_view():
    """渲染MAV实时报文面板"""
    st.subheader("📡 MAVLink 原始数据流报文窗口")
    mav_html = "<div class='mav-box'>"
    # 倒序展示最新报文
    for pkt in reversed(st.session_state.mavlink_packets):
        dir_text = "↑上行 GCS→FCU" if pkt["dir"] == "tx" else "↓下行 FCU→GCS"
        mav_html += f"[{pkt['time']}] {dir_text} | MSG_ID: {pkt['type']}<br>Payload: {pkt['content']}<hr style='border:#333'>"
    if len(st.session_state.mavlink_packets) == 0:
        mav_html += "<span style='color:#888'>暂无MAV报文，下发航线/启动任务自动生成报文记录</span>"
    mav_html += "</div>"
    st.markdown(mav_html, unsafe_allow_html=True)
    # 模拟报文调试按钮
    b1, b2 = st.columns(2)
    with b1:
        if st.button("模拟下发航点MAV包", use_container_width=True):
            mock = mavutil.mavlink.MAVLink_mission_item_message(
                0,0,0,mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,0,1,5,0,0,0,32.2335,118.7475,50
            )
            add_mav_packet("MISSION_ITEM", "tx", mock)
            add_tx_log("GCS→OBC→FCU: MAV_MSG MISSION_ITEM 航点下发")
            st.rerun()
    with b2:
        if st.button("模拟回传位置遥测包", use_container_width=True):
            mock = mavutil.mavlink.MAVLink_global_position_int_message(
                int(118.7475*1e7), int(32.2335*1e7), 50000, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0
            )
            add_mav_packet("GLOBAL_POSITION_INT", "rx", mock)
            add_rx_log("FCU→OBC→GCS: MAV_MSG GLOBAL_POSITION_INT 全局位置上报")
            st.rerun()

# ==================== 【新增2】GCS-OBC-FCU三层通信拓扑绘图函数（已修复字符串语法错误） ====================
def draw_comm_topology():
    st.markdown("<div class='topology-card'>", unsafe_allow_html=True)
    st.subheader("🔗 GCS-OBC-FCU 无人机三层通信拓扑结构图")
    option = {
        "tooltip": {"trigger": "item"},
        "series": [
            {
                "type": "graph",
                "layout": "force",
                "animation": False,
                "label": {"show": True, "fontSize": 14},
                "draggable": True,
                "data": [
                    {"name": "GCS地面站", "symbolSize": 65, "itemStyle": {"color": "#03A9F4"}},
                    {"name": "OBC机载计算机", "symbolSize": 65, "itemStyle": {"color": "#FFC107"}},
                    {"name": "FCU Pixhawk6X飞控", "symbolSize": 65, "itemStyle": {"color": "#4CAF50"}},
                    {"name": "任务载荷(云台/相机)", "symbolSize": 45, "itemStyle": {"color": "#9C27B0"}},
                    {"name": "IMU/GPS/避障传感器", "symbolSize": 45, "itemStyle": {"color": "#FF5722"}}
                ],
                "links": [
                    {"source": "GCS地面站", "target": "OBC机载计算机", "value": "无线数传 MAVLink双向"},
                    {"source": "OBC机载计算机", "target": "FCU Pixhawk6X飞控", "value": "UART/CAN MAVLink指令/遥测"},
                    {"source": "FCU Pixhawk6X飞控", "target": "IMU/GPS/避障传感器", "value": "I2C/CAN 原始传感数据"},
                    {"source": "OBC机载计算机", "target": "任务载荷(云台/相机)", "value": "USB/以太网 图像与云台控制"}
                ],
                "force": {"repulsion": 900}
            }
        ]
    }
    st_echarts(options=option, height="420px", key="topology_graph")
    st.markdown("</div>", unsafe_allow_html=True)

# ==================== 绕飞算法（完全原样保留，无修改） ====================
def compute_avoid_path(latA, lngA, latB, lngB, fly_height, obstacles, safety_radius_m=5.0):
    start = (lngA, latA)
    end = (lngB, latB)
    line = LineString([start, end])
    avg_lat = (latA + latB) / 2.0
    meter_per_deg_lat = 111320.0
    meter_per_deg_lon = 111320.0 * math.cos(math.radians(avg_lat))
    buffer_deg = safety_radius_m / ((meter_per_deg_lat + meter_per_deg_lon) / 2.0)

    blocking = []
    for ob in obstacles:
        if fly_height >= ob.get("height", 0):
            continue
        pts = ob["points"]
        if len(pts) < 3:
            continue
        coords = pts.copy()
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        poly = Polygon(coords)
        poly_buff = poly.buffer(buffer_deg)
        if line.intersects(poly_buff):
            blocking.append(poly_buff)

    if not blocking:
        straight = [start, end]
        return {"left": straight, "right": straight, "shortest": straight, "over": straight}

    merged = unary_union(blocking)
    if merged.geom_type == 'Polygon':
        boundary = merged.exterior
    elif merged.geom_type == 'MultiPolygon':
        boundary = merged.convex_hull.exterior
    else:
        boundary = merged.boundary

    coords = list(boundary.coords)
    if len(coords) < 4:
        straight = [start, end]
        return {"left": straight, "right": straight, "shortest": straight, "over": straight}

    intersection = boundary.intersection(line)
    pts = []
    if isinstance(intersection, Point):
        pts = [intersection]
    elif isinstance(intersection, MultiPoint):
        pts = list(intersection.geoms)
    if len(pts) < 2:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        ext_line = LineString([
            (start[0] - dx*0.01, start[1] - dy*0.01),
            (end[0] + dx*0.01, end[1] + dy*0.01)
        ])
        intersection = boundary.intersection(ext_line)
        if isinstance(intersection, Point):
            pts = [intersection]
        elif isinstance(intersection, MultiPoint):
            pts = list(intersection.geoms)
        if len(pts) < 2:
            straight = [start, end]
            return {"left": straight, "right": straight, "shortest": straight, "over": straight}

    pts.sort(key=lambda p: line.project(p))
    entry = pts[0]
    exit_ = pts[-1]

    def nearest_idx(point, coords):
        best, idx = float('inf'), 0
        for i, (x, y) in enumerate(coords):
            d = math.hypot(x - point.x, y - point.y)
            if d < best:
                best, idx = d, i
        return idx

    i_entry = nearest_idx(entry, coords)
    i_exit = nearest_idx(exit_, coords)

    if i_entry <= i_exit:
        arc1 = coords[i_entry:i_exit+1]
        arc2 = coords[i_exit:] + coords[:i_entry+1]
    else:
        arc1 = coords[i_entry:] + coords[:i_exit+1]
        arc2 = coords[i_exit:i_entry+1]

    dir_vec = (end[0] - start[0], end[1] - start[1])
    def cross_z(pt):
        vx, vy = pt[0] - start[0], pt[1] - start[1]
        return dir_vec[0]*vy - dir_vec[1]*vx

    avg_cross1 = sum(cross_z(p) for p in arc1) / len(arc1)
    avg_cross2 = sum(cross_z(p) for p in arc2) / len(arc2)
    if avg_cross1 > avg_cross2:
        left_boundary = arc1
        right_boundary = arc2
    else:
        left_boundary = arc2
        right_boundary = arc1

    entry_pt = (entry.x, entry.y)
    exit_pt = (exit_.x, exit_.y)

    def ensure_direction(boundary_seg, entry_pt, exit_pt):
        best_entry_idx = min(range(len(boundary_seg)),
                             key=lambda i: math.hypot(boundary_seg[i][0]-entry_pt[0],
                                                      boundary_seg[i][1]-entry_pt[1]))
        best_exit_idx = min(range(len(boundary_seg)),
                            key=lambda i: math.hypot(boundary_seg[i][0]-exit_pt[0],
                                                     boundary_seg[i][1]-exit_pt[1]))
        if best_entry_idx > best_exit_idx:
            return boundary_seg[::-1]
        return boundary_seg

    left_boundary = ensure_direction(left_boundary, entry_pt, exit_pt)
    right_boundary = ensure_direction(right_boundary, entry_pt, exit_pt)

    def sample_boundary(pts, min_pts=3):
        if len(pts) <= min_pts + 2:
            return pts[1:-1]
        n = len(pts)
        indices = [int(i) for i in np.linspace(0, n-1, min_pts+2)]
        sampled = [pts[i] for i in indices]
        return sampled[1:-1]

    left_mid = sample_boundary(left_boundary, 3)
    right_mid = sample_boundary(right_boundary, 3)

    left_path = [entry_pt] + left_mid + [exit_pt]
    right_path = [entry_pt] + right_mid + [exit_pt]

    def to_lat_lng(points):
        return [(p[1], p[0]) for p in points]

    left_latlon = to_lat_lng(left_path)
    right_latlon = to_lat_lng(right_path)

    def path_len(pts):
        if not pts:
            return float('inf')
        return sum(math.hypot(pts[i+1][0]-pts[i][0], pts[i+1][1]-pts[i][1]) for i in range(len(pts)-1))

    shortest_latlon = left_latlon if path_len(left_latlon) < path_len(right_latlon) else right_latlon

    max_dist = 0
    best_pt = None
    for x, y in coords:
        d = abs(cross_z((x, y)))
        if d > max_dist:
            max_dist = d
            best_pt = (x, y)

    over_curve = shortest_latlon
    if best_pt is not None:
        s = cross_z(best_pt)
        perp = (-dir_vec[1], dir_vec[0]) if s > 0 else (dir_vec[1], -dir_vec[0])
        perp_len = math.hypot(perp[0], perp[1])
        if perp_len > 0:
            perp = (perp[0]/perp_len, perp[1]/perp_len)
        offset_deg = 20.0 / ((meter_per_deg_lat + meter_per_deg_lon) / 2.0)
        control_x = best_pt[0] + perp[0] * offset_deg
        control_y = best_pt[1] + perp[1] * offset_deg
        control_pt = (control_x, control_y)

        steps = 30
        over_path = []
        for i in range(steps+1):
            t = i / steps
            x = (1-t)**2 * start[0] + 2*(1-t)*t * control_pt[0] + t**2 * end[0]
            y = (1-t)**2 * start[1] + 2*(1-t)*t * control_pt[1] + t**2 * end[1]
            over_path.append((x, y))

        if all(Point(x, y).distance(merged) > buffer_deg * 0.5 for x, y in over_path):
            over_curve = to_lat_lng(over_path)

    return {
        "left": left_latlon,
        "right": right_latlon,
        "shortest": shortest_latlon,
        "over": over_curve
    }

# ==================== 左侧面板 ====================
col_left, col_right = st.columns([1, 3])

with col_left:
    st.markdown('<div class="left-panel">', unsafe_allow_html=True)
    st.subheader("🧭 导航")
    # 新增第三个页面：通信拓扑与MAV调试
    page = st.radio("", ["航线规划", "飞行监控", "通信拓扑/MAV调试"], label_visibility="collapsed")
    st.divider()

    if page == "航线规划":
        click_mode = st.radio(
            "点击地图时",
            ["障碍物圈选", "选择终点"],
            horizontal=True,
            index=0 if st.session_state.click_mode == "障碍物圈选" else 1
        )
        if click_mode != st.session_state.click_mode:
            st.session_state.click_mode = click_mode
            add_operate_log(f"切换点击模式为：{click_mode}")
            save_state()

        st.divider()
        st.markdown("### 🛡️ 安全设置")
        safety_radius = st.slider("安全距离 (米)", 1.0, 30.0, st.session_state.safety_radius, 0.5)
        if safety_radius != st.session_state.safety_radius:
            st.session_state.safety_radius = safety_radius
            add_operate_log(f"修改安全避让距离为：{safety_radius}米")
            save_state()

        st.divider()
        st.markdown("### 🚧 障碍物圈选")
        name = st.text_input("障碍物名称", "教学楼")
        height = st.number_input("高度(m)", 1, 500, 25)
        st.info(f"当前已打点：{len(st.session_state.draw_points)} 个")
        if st.button("🧹 清空当前打点", use_container_width=True):
            st.session_state.draw_points = []
            add_operate_log("清空障碍物临时打点坐标")
            save_state()
            st.rerun()
        if st.button("✅ 保存障碍物", type="primary", use_container_width=True):
            if len(st.session_state.draw_points) >= 3:
                st.session_state.obstacles.append({
                    "name": name, "height": height,
                    "points": st.session_state.draw_points.copy()
                })
                st.session_state.draw_points = []
                add_operate_log(f"成功保存障碍物：{name}，高度{height}m")
                save_state()
                st.success("✅ 保存成功！")
                st.rerun()
            else:
                st.warning("⚠️ 至少需要3个点")
        st.divider()
        st.markdown("### 📋 已保存障碍物")
        for i, ob in enumerate(st.session_state.obstacles):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.write(f"📍 {ob['name']} ({ob['height']}m)")
            with c2:
                if st.button("🗑️ 删除", key=f"del_{i}", use_container_width=True):
                    del st.session_state.obstacles[i]
                    add_operate_log(f"删除障碍物：{ob['name']}")
                    save_state()
                    st.rerun()
    elif page == "通信拓扑/MAV调试":
        st.info("右侧展示三层通信拓扑图与MAVLink原始报文")
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== 右侧布局 ====================
with col_right:
    st.markdown("# 🎓 南京科技职业学院")
    st.markdown("## 无人机航线导航与监控系统")

    if page == "航线规划":
        st.markdown("### 🎯 AB点航线")
        c1, c2 = st.columns(2)
        with c1:
            latA = st.number_input("起点A纬度", value=st.session_state.latA, format="%.6f")
            lngA = st.number_input("起点A经度", value=st.session_state.lngA, format="%.6f")
        with c2:
            latB = st.number_input("终点B纬度", value=st.session_state.latB, format="%.6f")
            lngB = st.number_input("终点B经度", value=st.session_state.lngB, format="%.6f")
        st.session_state.latA, st.session_state.lngA = latA, lngA
        st.session_state.latB, st.session_state.lngB = latB, lngB

        fly_h = st.number_input("飞行高度(m)", 1, 500, 50)
        map_type = st.radio("🗺️ 地图模式", ["高德普通地图", "卫星影像地图"], horizontal=True)

        paths = compute_avoid_path(latA, lngA, latB, lngB, fly_h, st.session_state.obstacles, st.session_state.safety_radius)
        left_pts = paths["left"]
        right_pts = paths["right"]
        shortest_pts = paths["shortest"]
        over_pts = paths["over"]

        center_lat = (latA + latB) / 2
        center_lng = (lngA + lngB) / 2
        m = folium.Map(location=[center_lat, center_lng], zoom_start=17, control_scale=True)
        if map_type == "卫星影像地图":
            TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri", name="卫星影像", max_zoom=20).add_to(m)
        else:
            TileLayer(
                tiles="https://webrd01.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}",
                attr="© 高德", name="高德地图", max_zoom=20).add_to(m)

        folium.PolyLine([[latA, lngA], [latB, lngB]], color="red", weight=2, opacity=0.6, dash_array="5,5").add_to(m)

        if left_pts:
            folium.PolyLine([[latA, lngA]] + left_pts + [[latB, lngB]], color="blue", weight=3, popup="左绕").add_to(m)
        if right_pts:
            folium.PolyLine([[latA, lngA]] + right_pts + [[latB, lngB]], color="green", weight=3, popup="右绕").add_to(m)
        if shortest_pts:
            folium.PolyLine([[latA, lngA]] + shortest_pts + [[latB, lngB]], color="orange", weight=4, popup="最短绕飞").add_to(m)
        if over_pts:
            folium.PolyLine([[latA, lngA]] + over_pts + [[latB, lngB]], color="purple", weight=4, dash_array="10,5", popup="上绕弧线").add_to(m)

        folium.Marker([latA, lngA], icon=folium.Icon(color="green", icon="info-sign")).add_to(m)
        folium.Marker([latB, lngB], icon=folium.Icon(color="red", icon="info-sign")).add_to(m)

        for ob in st.session_state.obstacles:
            ps = [[lat, lng] for (lng, lat) in ob["points"]]
            folium.Polygon(ps, color="red", fill=True, fill_opacity=0.5, popup=f"{ob['name']} ({ob['height']}m)").add_to(m)

        if len(st.session_state.draw_points) >= 2:
            ps = [[lat, lng] for (lng, lat) in st.session_state.draw_points]
            folium.Polygon(ps, color="blue", fill=True, fill_opacity=0.2).add_to(m)

        o = st_folium.st_folium(m, width=1400, height=700, returned_objects=["last_clicked"])
        if o and o.get("last_clicked"):
            click_lat = o["last_clicked"]["lat"]
            click_lng = o["last_clicked"]["lng"]
            pt = (round(click_lng, 6), round(click_lat, 6))
            if st.session_state.click_mode == "障碍物圈选":
                if pt != st.session_state.last_click:
                    st.session_state.last_click = pt
                    st.session_state.draw_points.append(pt)
                    add_operate_log(f"地图点击添加障碍物打点：经纬度{click_lat},{click_lng}")
                    save_state()
                    st.rerun()
            else:
                st.session_state.latB = round(click_lat, 6)
                st.session_state.lngB = round(click_lng, 6)
                st.session_state.last_click = pt
                add_operate_log(f"地图点击设置飞行终点：{click_lat},{click_lng}")
                save_state()
                st.rerun()

        st.markdown("---")
        if over_pts:
            col_path1, col_path2, col_path3 = st.columns(3)
            with col_path1:
                if st.button("🛩️ 上绕越障", use_container_width=True):
                    st.session_state.waypoints = [[latA, lngA]] + over_pts + [[latB, lngB]]
                    st.session_state.flight_status = "idle"
                    add_operate_log("用户选定航线：上绕越障航线")
                    add_tx_log("GCS→OBC→FCU: 上传上绕越障航线任务")
                    # 同步生成MAV航线报文
                    mock_mission_cnt = mavutil.mavlink.MAVLink_mission_count_message(0,0,len(st.session_state.waypoints))
                    add_mav_packet("MISSION_COUNT", "tx", mock_mission_cnt)
                    save_state()
                    st.success("上绕航线已设置！")
            with col_path2:
                if st.button("🛩️ 左绕航线", use_container_width=True):
                    st.session_state.waypoints = [[latA, lngA]] + left_pts + [[latB, lngB]]
                    st.session_state.flight_status = "idle"
                    add_operate_log("用户选定航线：左侧绕行航线")
                    add_tx_log("GCS→OBC→FCU: 上传左绕航线任务")
                    mock_mission_cnt = mavutil.mavlink.MAVLink_mission_count_message(0,0,len(st.session_state.waypoints))
                    add_mav_packet("MISSION_COUNT", "tx", mock_mission_cnt)
                    save_state()
                    st.success("左绕航线已设置！")
            with col_path3:
                if st.button("🛩️ 右绕航线", use_container_width=True):
                    st.session_state.waypoints = [[latA, lngA]] + right_pts + [[latB, lngB]]
                    st.session_state.flight_status = "idle"
                    add_operate_log("用户选定航线：右侧绕行航线")
                    add_tx_log("GCS→OBC→FCU: 上传右绕航线任务")
                    mock_mission_cnt = mavutil.mavlink.MAVLink_mission_count_message(0,0,len(st.session_state.waypoints))
                    add_mav_packet("MISSION_COUNT", "tx", mock_mission_cnt)
                    save_state()
                    st.success("右绕航线已设置！")
        else:
            if st.button("🛩️ 使用直飞航线", use_container_width=True):
                st.session_state.waypoints = [[latA, lngA], [latB, lngB]]
                st.session_state.flight_status = "idle"
                add_operate_log("用户选定航线：两点直飞航线")
                add_tx_log("GCS→OBC→FCU: 上传直飞航线任务")
                mock_mission_cnt = mavutil.mavlink.MAVLink_mission_count_message(0,0,2)
                add_mav_packet("MISSION_COUNT", "tx", mock_mission_cnt)
                save_state()
                st.success("直飞航线已设置。")

    elif page == "飞行监控":
        st.markdown("## ✈️ 飞行实时画面 - 任务执行监控")
        st.markdown("### 📡 通信链路拓扑与数据流")
        
        link_cols = st.columns(4)
        link_cols[0].success("🟢 GCS 在线")
        link_cols[1].success("🟢 OBC 在线")
        link_cols[2].success("🟢 FCU 在线")
        link_cols[3].info("📶 链路正常")

        st.divider()
        log_col0, log_col1, log_col2 = st.columns(3)

        with log_col0:
            st.markdown("#### 📝 业务流程")
            log_html = "<div class='log-box'>"
            for line in st.session_state.get("operate_logs", []):
                log_html += f"{line}<br>"
            log_html += "</div>"
            st.markdown(log_html, unsafe_allow_html=True)

        with log_col1:
            st.markdown("#### 📤 GCS → OBC → FCU")
            log_html = "<div class='log-box'>"
            for line in st.session_state.get("tx_logs", []):
                log_html += f"{line}<br>"
            log_html += "</div>"
            st.markdown(log_html, unsafe_allow_html=True)

        with log_col2:
            st.markdown("#### 📥 FCU → OBC → GCS")
            log_html = "<div class='log-box'>"
            for line in st.session_state.get("rx_logs", []):
                log_html += f"{line}<br>"
            log_html += "</div>"
            st.markdown(log_html, unsafe_allow_html=True)

        st.divider()
        waypoints = st.session_state.waypoints
        if not waypoints:
            st.warning("⚠️ 尚未设置航线，请先在航线规划页面计算并应用航线。")
        else:
            if st.session_state.flight_status == "idle":
                btn_col1, btn_col2 = st.columns(2)
                if btn_col1.button("▶️ 开始任务", use_container_width=True):
                    st.session_state.flight_status = "running"
                    st.session_state.flight_start_time = time.time()
                    st.session_state.elapsed_flight = 0.0
                    st.session_state.last_wp = -1
                    add_operate_log("用户手动点击开始执行飞行任务")
                    add_tx_log("GCS→OBC→FCU: 启动任务 AUTO")
                    add_rx_log("FCU→OBC→GCS: ACK | Mode: AUTO")
                    # 生成启动MAV报文
                    start_mav = mavutil.mavlink.MAVLink_set_mode_message(0, mavutil.mavlink.MAV_MODE_FLAG_AUTO_ENABLED, mavutil.mavlink.MAV_MODE_AUTO_MISSION)
                    add_mav_packet("SET_MODE", "tx", start_mav)
                    ack_mav = mavutil.mavlink.MAVLink_command_ack_message(mavutil.mavlink.MAV_CMD_MISSION_START, 0,0,0,0,0,0,0)
                    add_mav_packet("COMMAND_ACK", "rx", ack_mav)
                    save_state()
                    st.rerun()
                btn_col2.button("⏸️ 暂停任务", disabled=True)

            elif st.session_state.flight_status == "running":
                btn_col1, btn_col2 = st.columns(2)
                btn_col1.button("▶️ 开始任务", disabled=True)
                if btn_col2.button("⏸️ 暂停任务", use_container_width=True):
                    st.session_state.elapsed_flight += time.time() - st.session_state.flight_start_time
                    st.session_state.flight_status = "paused"
                    add_operate_log("用户手动暂停当前飞行任务")
                    pause_mav = mavutil.mavlink.MAVLink_set_mode_message(0, mavutil.mavlink.MAV_MODE_FLAG_STABILIZE_ENABLED, mavutil.mavlink.MAV_MODE_STABILIZE_HOLD)
                    add_mav_packet("SET_MODE", "tx", pause_mav)
                    save_state()
                    st.rerun()

            elif st.session_state.flight_status == "paused":
                btn_col1, btn_col2, btn_col3 = st.columns(3)
                if btn_col1.button("▶️ 继续任务", use_container_width=True):
                    st.session_state.flight_start_time = time.time()
                    st.session_state.flight_status = "running"
                    add_operate_log("用户手动恢复继续飞行任务")
                    resume_mav = mavutil.mavlink.MAVLink_set_mode_message(0, mavutil.mavlink.MAV_MODE_FLAG_AUTO_ENABLED, mavutil.mavlink.MAV_MODE_AUTO_MISSION)
                    add_mav_packet("SET_MODE", "tx", resume_mav)
                    save_state()
                    st.rerun()
                if btn_col2.button("⏹️ 重置任务", use_container_width=True):
                    st.session_state.flight_status = "idle"
                    st.session_state.elapsed_flight = 0.0
                    st.session_state.flight_start_time = None
                    st.session_state.last_wp = -1
                    add_operate_log("用户手动重置本次飞行任务")
                    clear_mav = mavutil.mavlink.MAVLink_mission_clear_all_message(0,0)
                    add_mav_packet("MISSION_CLEAR_ALL", "tx", clear_mav)
                    save_state()
                    st.rerun()
                btn_col3.button("⏸️ 暂停任务", disabled=True)

            if st.session_state.flight_status in ("running", "paused"):
                waypoint_list = waypoints
                total_distance = 0.0
                segments = []
                for i in range(len(waypoint_list)-1):
                    d = haversine(waypoint_list[i][0], waypoint_list[i][1], waypoint_list[i+1][0], waypoint_list[i+1][1])
                    segments.append(d)
                    total_distance += d

                if st.session_state.flight_status == "running":
                    current_elapsed = st.session_state.elapsed_flight + (time.time() - st.session_state.flight_start_time)
                else:
                    current_elapsed = st.session_state.elapsed_flight

                SPEED = 8.5
                flown = min(current_elapsed * SPEED, total_distance)
                remain = total_distance - flown
                eta_seconds = remain / SPEED if SPEED > 0 else 0

                cur_lat, cur_lon = waypoint_list[0]
                seg_idx = 0
                if total_distance > 0:
                    cum = 0.0
                    for i, d in enumerate(segments):
                        if cum + d >= flown:
                            seg_idx = i
                            seg_progress = (flown - cum) / d
                            lat1, lon1 = waypoint_list[i]
                            lat2, lon2 = waypoint_list[i+1]
                            cur_lat = lat1 + (lat2 - lat1) * seg_progress
                            cur_lon = lon1 + (lon2 - lon1) * seg_progress
                            break
                        cum += d
                    else:
                        seg_idx = len(segments) - 1
                        cur_lat, cur_lon = waypoint_list[-1]

                total_wp = len(waypoint_list) - 1
                current_wp = min(seg_idx + 1, total_wp)
                wp_disp = f"{current_wp}/{total_wp}"

                if current_wp > st.session_state.get("last_wp", -1) and current_wp <= total_wp:
                    st.session_state.last_wp = current_wp
                    add_rx_log(f"FCU→OBC→GCS: WP_REACHED #{current_wp}")
                    wp_mav = mavutil.mavlink.MAVLink_mission_item_reached_message(current_wp)
                    add_mav_packet("MISSION_ITEM_REACHED", "rx", wp_mav)
                    if current_wp == total_wp:
                        add_rx_log("FCU→OBC→GCS: MISSION_COMPLETE")
                        finish_mav = mavutil.mavlink.MAVLink_mission_ack_message(0, mavutil.mavlink.MAV_MISSION_ACCEPTED)
                        add_mav_packet("MISSION_ACK", "rx", finish_mav)
                        add_operate_log("飞行任务全部完成，已抵达终点")

                batt = max(0.0, 100 - (flown / total_distance * 100)) if total_distance > 0 else 100.0
                elapsed_str = str(datetime.timedelta(seconds=int(current_elapsed)))

                cols = st.columns(5)
                cols[0].metric("当前航点", wp_disp)
                cols[1].metric("飞行速度", f"{SPEED:.1f} m/s")
                cols[2].metric("已用时间", elapsed_str)
                cols[3].metric("剩余距离", f"{remain:.1f} m")
                cols[4].metric("电量模拟", f"{batt:.1f}%")

                st.write(f"DEBUG | 剩余距离: {remain:.2f}m | 速度: {SPEED} m/s | 预计秒数: {eta_seconds:.2f}s")
                progress = flown / total_distance if total_distance > 0 else 1.0
                st.progress(min(progress, 1.0))

                m2 = folium.Map(location=[cur_lat, cur_lon], zoom_start=17)
                folium.PolyLine(waypoint_list, color="orange", weight=4).add_to(m2)
                folium.Marker(waypoint_list[0], icon=folium.Icon(color="green"), popup="起点").add_to(m2)
                folium.Marker(waypoint_list[-1], icon=folium.Icon(color="red"), popup="终点").add_to(m2)
                folium.Marker([cur_lat, cur_lon], icon=folium.Icon(color="blue", icon="plane", prefix="fa"), popup="无人机").add_to(m2)
                st_folium.st_folium(m2, width=1400, height=400)

        st.markdown("---")
        st.markdown("### 💓 地面站心跳监测")
        if not st.session_state.running:
            if st.button("▶️ 开始心跳监测"):
                st.session_state.running = True
                add_operate_log("开启心跳监测")
                save_state()
                st.rerun()
        else:
            if st.button("⏸️ 停止心跳监测"):
                st.session_state.running = False
                add_operate_log("停止心跳监测")
                save_state()
                st.rerun()

        if st.session_state.running:
            st.session_state.seq += 1
            t = datetime.datetime.now().strftime("%H:%M:%S")
            st.session_state.heartbeat_data.append({"序号": st.session_state.seq, "时间": t, "状态": "正常"})
            # 心跳同步生成MAV HEARTBEAT下行报文
            hb_mav = mavutil.mavlink.MAVLink_heartbeat_message(
                mavutil.mavlink.MAV_TYPE_QUADROTOR,
                mavutil.mavlink.MAV_AUTOPILOT_PX4,
                0,0,0,0
            )
            add_mav_packet("HEARTBEAT", "rx", hb_mav)
            save_state()
            df = pd.DataFrame(st.session_state.heartbeat_data[-20:])
            st.line_chart(df.set_index("时间")["序号"])
            st.dataframe(df, use_container_width=True)

    elif page == "通信拓扑/MAV调试":
        # 1. 绘制三层GCS-OBC-FCU拓扑图
        draw_comm_topology()
        st.divider()
        # 2. MAVLink报文实时窗口
        render_mavlink_view()

# 原有底部自动刷新逻辑完整保留
if page == "飞行监控" and (st.session_state.flight_status == "running" or st.session_state.running):
    time.sleep(1)
    st.rerun()

save_state()
