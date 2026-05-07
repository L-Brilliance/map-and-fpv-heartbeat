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
from shapely.geometry import Polygon, LineString
from shapely.ops import unary_union

# ==================== 页面配置 ====================
st.set_page_config(page_title="南京科技职业学院 - 无人机导航系统", layout="wide")

# ==================== 样式 ====================
st.markdown("""
<style>
.left-panel {background:#f8f9fa; padding:20px; border-radius:10px; height:95vh;}
</style>
""", unsafe_allow_html=True)

# ==================== 持久化文件 ====================
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
        "flight_start_time": st.session_state.flight_start_time.isoformat() if st.session_state.flight_start_time else None,
        "flight_paused_duration": st.session_state.flight_paused_duration,
        "flight_speed": st.session_state.flight_speed,
        "safety_radius": st.session_state.safety_radius,
    }
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 恢复 datetime 对象
        if data.get("flight_start_time"):
            data["flight_start_time"] = datetime.datetime.fromisoformat(data["flight_start_time"])
        return data
    return {}

# ==================== 强健的 session_state 初始化 ====================
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
        "flight_status": "idle",       # idle, running, paused
        "flight_start_time": None,
        "flight_paused_duration": 0.0,
        "flight_speed": 8.5,           # m/s
        "safety_radius": 5.0,          # 米
    }
    loaded = load_state()
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = loaded.get(key, default_value)
    if "init" not in st.session_state:
        st.session_state.init = True

ensure_session_state()

# ==================== Haversine 距离计算 ====================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ==================== 绕飞路径计算（支持安全距离） ====================
def compute_avoid_path(latA, lngA, latB, lngB, fly_height, obstacles, safety_radius_m=5.0):
    start = (lngA, latA)
    end = (lngB, latB)
    line = LineString([start, end])

    # 近似经纬度偏移：1度 ≈ 111320米，考虑平均纬度
    avg_lat = (latA + latB) / 2.0
    meter_per_deg_lat = 111320.0
    meter_per_deg_lon = 111320.0 * math.cos(math.radians(avg_lat))
    # 使用平均比例统一做 buffer（近似圆形）
    buffer_deg = safety_radius_m / ((meter_per_deg_lat + meter_per_deg_lon) / 2.0)

    blocking_buffered = []
    for ob in obstacles:
        ob_height = ob.get("height", 0)
        if fly_height >= ob_height:
            continue
        pts = ob["points"]
        if len(pts) < 3:
            continue
        coords = pts.copy()
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        poly = Polygon(coords)
        # 扩展安全缓冲区
        poly_buff = poly.buffer(buffer_deg)
        if line.intersects(poly_buff):
            blocking_buffered.append(poly_buff)

    if not blocking_buffered:
        return {"left": [], "right": [], "shortest": []}

    merged = unary_union(blocking_buffered)
    polys = list(merged.geoms) if merged.geom_type == "MultiPolygon" else [merged]

    def side_points(poly):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        left_pts, right_pts = [], []
        for (x, y) in poly.exterior.coords:
            vx, vy = x - start[0], y - start[1]
            cross = dx * vy - dy * vx
            if cross > 1e-9:
                left_pts.append((cross, (x, y)))
            elif cross < -1e-9:
                right_pts.append((-cross, (x, y)))
        best_left = max(left_pts, key=lambda t: t[0])[1] if left_pts else None
        best_right = max(right_pts, key=lambda t: t[0])[1] if right_pts else None
        return best_left, best_right

    left_c, right_c = [], []
    for p in polys:
        l, r = side_points(p)
        if l: left_c.append(l)
        if r: right_c.append(r)

    def dist(p):
        return math.hypot(p[0]-start[0], p[1]-start[1])

    left_c.sort(key=dist)
    right_c.sort(key=dist)

    left_path = None
    if left_c:
        left_path = [start] + left_c + [end]
    right_path = None
    if right_c:
        right_path = [start] + right_c + [end]

    def create_bezier_arc_path(path_pts, control_scale=0.5):
        if not path_pts or len(path_pts) < 3:
            return []
        arc_path = []
        arc_path.append(path_pts[0])
        for i in range(1, len(path_pts)-1):
            p0 = path_pts[i-1]
            p1 = path_pts[i]
            p2 = path_pts[i+1]
            dx1 = p1[0] - p0[0]
            dy1 = p1[1] - p0[1]
            dx2 = p2[0] - p1[0]
            dy2 = p2[1] - p1[1]
            control_x = p1[0] - (dx1 + dx2) * control_scale
            control_y = p1[1] - (dy1 + dy2) * control_scale
            steps = 10
            for step in range(1, steps+1):
                t = step / steps
                x = (1-t)**2 * p0[0] + 2*(1-t)*t * control_x + t**2 * p2[0]
                y = (1-t)**2 * p0[1] + 2*(1-t)*t * control_y + t**2 * p2[1]
                arc_path.append((x, y))
        arc_path.append(path_pts[-1])
        return arc_path

    left_arc = create_bezier_arc_path(left_path) if left_path else []
    right_arc = create_bezier_arc_path(right_path) if right_path else []

    def path_len(p):
        if not p:
            return float('inf')
        return sum(math.hypot(p[i+1][0]-p[i][0], p[i+1][1]-p[i][1]) for i in range(len(p)-1))

    len_l = path_len(left_arc)
    len_r = path_len(right_arc)
    shortest_path = left_arc if len_l < len_r else right_arc

    def to_lat_lng(points):
        return [(p[1], p[0]) for p in points] if points else []

    return {
        "left": to_lat_lng(left_path) if left_path else [],
        "right": to_lat_lng(right_path) if right_path else [],
        "shortest": to_lat_lng(shortest_path) if shortest_path else []
    }

# ==================== 左侧布局 ====================
col_left, col_right = st.columns([1, 3])

with col_left:
    st.markdown('<div class="left-panel">', unsafe_allow_html=True)
    st.subheader("🧭 导航")
    page = st.radio("", ["航线规划", "飞行监控"], label_visibility="collapsed")
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
            save_state()

        st.divider()

        # 安全距离设置
        st.markdown("### 🛡️ 安全设置")
        safety_radius = st.slider("安全距离 (米)", min_value=1.0, max_value=30.0, value=st.session_state.safety_radius, step=0.5)
        if safety_radius != st.session_state.safety_radius:
            st.session_state.safety_radius = safety_radius
            save_state()

        st.divider()

        st.markdown("### 🚧 障碍物圈选")
        name = st.text_input("障碍物名称", "教学楼")
        height = st.number_input("高度(m)", min_value=1, max_value=500, value=25, step=1)

        st.info(f"当前已打点：{len(st.session_state.draw_points)} 个")

        if st.button("🧹 清空当前打点", use_container_width=True):
            st.session_state.draw_points = []
            save_state()
            st.rerun()

        if st.button("✅ 保存障碍物", type="primary", use_container_width=True):
            if len(st.session_state.draw_points) >= 3:
                st.session_state.obstacles.append({
                    "name": name,
                    "height": height,
                    "points": st.session_state.draw_points.copy()
                })
                st.session_state.draw_points = []
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
                    save_state()
                    st.rerun()

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

        st.session_state.latA = latA
        st.session_state.lngA = lngA
        st.session_state.latB = latB
        st.session_state.lngB = lngB

        fly_h = st.number_input("飞行高度(m)", min_value=1, max_value=500, value=50, step=1)
        map_type = st.radio("🗺️ 地图模式", ["高德普通地图", "卫星影像地图"], horizontal=True)

        paths = compute_avoid_path(
            latA, lngA, latB, lngB, fly_h,
            st.session_state.obstacles,
            safety_radius_m=st.session_state.safety_radius
        )
        left_pts = paths["left"]
        right_pts = paths["right"]
        shortest_pts = paths["shortest"]

        # 构建地图
        center_lat = (latA + latB) / 2
        center_lng = (lngA + lngB) / 2
        m = folium.Map(location=[center_lat, center_lng], zoom_start=17, control_scale=True)

        if map_type == "卫星影像地图":
            TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri", name="卫星影像", max_zoom=20
            ).add_to(m)
        else:
            TileLayer(
                tiles="https://webrd01.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}",
                attr="© 高德", name="高德地图", max_zoom=20
            ).add_to(m)

        # 原始航线（红色虚线）
        folium.PolyLine(
            locations=[[latA, lngA], [latB, lngB]],
            color="red", weight=2, opacity=0.6, dash_array="5,5", popup="原始直飞航线"
        ).add_to(m)

        if left_pts:
            path = [[latA, lngA]] + [[lat, lng] for (lat, lng) in left_pts] + [[latB, lngB]]
            folium.PolyLine(locations=path, color="blue", weight=3, opacity=0.8, popup="向左绕飞路径").add_to(m)

        if right_pts:
            path = [[latA, lngA]] + [[lat, lng] for (lat, lng) in right_pts] + [[latB, lngB]]
            folium.PolyLine(locations=path, color="green", weight=3, opacity=0.8, popup="向右绕飞路径").add_to(m)

        if shortest_pts:
            path = [[lat, lng] for (lat, lng) in shortest_pts]
            folium.PolyLine(locations=path, color="orange", weight=5, opacity=0.9, popup="最短弧线路径（推荐）").add_to(m)

        folium.Marker([latA, lngA], popup="起点A", icon=folium.Icon(color="green", icon="info-sign")).add_to(m)
        folium.Marker([latB, lngB], popup="终点B", icon=folium.Icon(color="red", icon="info-sign")).add_to(m)

        for ob in st.session_state.obstacles:
            ps = [[lat, lng] for (lng, lat) in ob["points"]]
            folium.Polygon(locations=ps, color="red", fill=True, fill_opacity=0.5,
                           popup=f"{ob['name']} ({ob['height']}m)").add_to(m)

        if len(st.session_state.draw_points) >= 2:
            ps = [[lat, lng] for (lng, lat) in st.session_state.draw_points]
            folium.Polygon(locations=ps, color="blue", fill=True, fill_opacity=0.2).add_to(m)

        o = st_folium.st_folium(m, width=1400, height=700, returned_objects=["last_clicked"])

        if o and o.get("last_clicked"):
            click_lat = o["last_clicked"]["lat"]
            click_lng = o["last_clicked"]["lng"]
            pt = (round(click_lng, 6), round(click_lat, 6))

            if st.session_state.click_mode == "障碍物圈选":
                if pt != st.session_state.last_click:
                    st.session_state.last_click = pt
                    st.session_state.draw_points.append(pt)
                    save_state()
                    st.rerun()
            else:
                st.session_state.latB = round(click_lat, 6)
                st.session_state.lngB = round(click_lng, 6)
                st.session_state.last_click = pt
                save_state()
                st.rerun()

        # 应用航线按钮
        st.markdown("---")
        if shortest_pts:
            col_path1, col_path2 = st.columns(2)
            with col_path1:
                if st.button("🛩️ 使用推荐航线飞行", use_container_width=True):
                    st.session_state.waypoints = [[latA, lngA]] + [[lat, lng] for (lat, lng) in shortest_pts] + [[latB, lngB]]
                    st.session_state.flight_status = "idle"
                    save_state()
                    st.success("航线已设置，请切换到“飞行监控”页面执行任务。")
            with col_path2:
                # 也可以手动设置
                pass
        else:
            st.info("当前航线与障碍物无冲突，可直飞。点击按钮设置直飞航线。")
            if st.button("🛩️ 使用直飞航线", use_container_width=True):
                st.session_state.waypoints = [[latA, lngA], [latB, lngB]]
                st.session_state.flight_status = "idle"
                save_state()
                st.success("直飞航线已设置。")

    else:
        # ==================== 飞行监控页面 ====================
        st.markdown("## ✈️ 飞行实时画面 - 任务执行监控")

        # 通信链路状态
        st.markdown("### 📡 通信链路拓扑与数据流")
        link_cols = st.columns(4)
        with link_cols[0]:
            st.success("🟢 GCS在线")
        with link_cols[1]:
            st.success("🟢 OBC在线")
        with link_cols[2]:
            st.success("🟢 FCU在线")
        with link_cols[3]:
            # 电量单独显示，后面还有详细
            pass

        # 控制按钮
        waypoints = st.session_state.waypoints
        if not waypoints:
            st.warning("⚠️ 尚未设置航线，请先在“航线规划”页面计算并应用航线。")
        else:
            # 控制逻辑
            if st.session_state.flight_status == "idle":
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button("▶️ 开始任务", use_container_width=True):
                        st.session_state.flight_status = "running"
                        st.session_state.flight_start_time = datetime.datetime.now()
                        st.session_state.flight_paused_duration = 0.0
                        save_state()
                        st.rerun()
                with btn_col2:
                    st.button("⏸️ 暂停任务", disabled=True, use_container_width=True)
            elif st.session_state.flight_status == "running":
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    st.button("▶️ 开始任务", disabled=True, use_container_width=True)
                with btn_col2:
                    if st.button("⏸️ 暂停任务", use_container_width=True):
                        # 记录暂停时刻
                        now = datetime.datetime.now()
                        elapsed = (now - st.session_state.flight_start_time).total_seconds() - st.session_state.flight_paused_duration
                        st.session_state.flight_paused_duration += elapsed
                        st.session_state.flight_status = "paused"
                        save_state()
                        st.rerun()
            elif st.session_state.flight_status == "paused":
                btn_col1, btn_col2, btn_col3 = st.columns(3)
                with btn_col1:
                    if st.button("▶️ 继续任务", use_container_width=True):
                        # 继续时不重置开始时间，只清除暂停累计时长？实际上继续需要继续累计时间
                        # 简单实现：继续等于恢复 running，并保持累计暂停时长
                        st.session_state.flight_status = "running"
                        # 不修改 start_time 和 paused_duration，下一次循环会计算从开始到现在总用时减去暂停累计
                        save_state()
                        st.rerun()
                with btn_col2:
                    if st.button("⏹️ 重置任务", use_container_width=True):
                        st.session_state.flight_status = "idle"
                        st.session_state.flight_start_time = None
                        st.session_state.flight_paused_duration = 0.0
                        save_state()
                        st.rerun()
                with btn_col3:
                    st.button("⏸️ 暂停任务", disabled=True, use_container_width=True)

            # 如果任务在运行或暂停，显示动态数据
            if st.session_state.flight_status in ("running", "paused"):
                # 计算飞行进度
                waypoint_list = waypoints
                total_distance = 0.0
                segments = []
                for i in range(len(waypoint_list)-1):
                    d = haversine(waypoint_list[i][0], waypoint_list[i][1],
                                  waypoint_list[i+1][0], waypoint_list[i+1][1])
                    segments.append(d)
                    total_distance += d

                # 计算已飞距离
                if st.session_state.flight_status == "running":
                    now = datetime.datetime.now()
                    elapsed = (now - st.session_state.flight_start_time).total_seconds() - st.session_state.flight_paused_duration
                else:  # paused
                    elapsed = st.session_state.flight_paused_duration

                flown_distance = min(elapsed * st.session_state.flight_speed, total_distance)
                remaining_distance = total_distance - flown_distance

                # 定位当前航点/位置
                if total_distance == 0:
                    current_lat, current_lon = waypoint_list[0]
                    current_seg = 0
                else:
                    cum = 0.0
                    seg_idx = 0
                    for i, d in enumerate(segments):
                        if cum + d >= flown_distance:
                            seg_idx = i
                            seg_progress = (flown_distance - cum) / d
                            # 线性插值
                            lat1, lon1 = waypoint_list[i]
                            lat2, lon2 = waypoint_list[i+1]
                            current_lat = lat1 + (lat2 - lat1) * seg_progress
                            current_lon = lon1 + (lon2 - lon1) * seg_progress
                            break
                        cum += d
                    else:
                        # 已经到达终点
                        seg_idx = len(segments) - 1
                        current_lat, current_lon = waypoint_list[-1]

                # 当前航点显示
                total_waypoints = len(waypoint_list) - 1  # 段数作为航点数
                current_wp_display = f"{min(seg_idx+1, total_waypoints)}/{total_waypoints}"

                # 剩余时间 & 预计到达
                if st.session_state.flight_speed > 0:
                    remaining_seconds = remaining_distance / st.session_state.flight_speed
                else:
                    remaining_seconds = 0
                eta = datetime.datetime.now() + datetime.timedelta(seconds=remaining_seconds)
                eta_str = eta.strftime("%H:%M:%S")

                # 已用时间
                elapsed_td = datetime.timedelta(seconds=int(elapsed))
                elapsed_str = str(elapsed_td)

                # 电量模拟：按飞行距离比例消耗，剩余电量 = max(0, 100 - 100 * flown_distance / total_distance)
                battery = max(0.0, 100.0 - 100.0 * flown_distance / total_distance) if total_distance > 0 else 100.0

                # 显示指标面板
                st.markdown("---")
                metric_cols = st.columns(5)
                metric_cols[0].metric("📍 当前航点", current_wp_display)
                metric_cols[1].metric("⚡ 飞行速度", f"{st.session_state.flight_speed:.1f} m/s")
                metric_cols[2].metric("⏱️ 已用时间", elapsed_str)
                metric_cols[3].metric("🛣️ 剩余距离", f"{remaining_distance:.1f} m")
                metric_cols[4].metric("⏰ 预计到达", eta_str)

                st.markdown("---")
                # 电量模拟单独行（紫色框）
                st.markdown(f"""
                <div style="background-color:#f3e5f5; border-radius:10px; padding:10px; margin-bottom:10px;">
                    <strong>🔋 电量模拟</strong>&nbsp;&nbsp;
                    <span style="font-size:1.2em; color:{'red' if battery < 20 else 'green'}">{battery:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)

                # 任务进度条
                progress = flown_distance / total_distance if total_distance > 0 else 1.0
                st.progress(min(progress, 1.0))
                st.caption(f"任务进度：{progress*100:.1f}%")

                # 显示实时地图
                m2 = folium.Map(location=[current_lat, current_lon], zoom_start=17, control_scale=True)
                # 添加航线
                if len(waypoint_list) > 1:
                    folium.PolyLine(locations=waypoint_list, color="orange", weight=3, popup="航线").add_to(m2)
                # 起点、终点标记
                folium.Marker(waypoint_list[0], popup="起点", icon=folium.Icon(color="green")).add_to(m2)
                folium.Marker(waypoint_list[-1], popup="终点", icon=folium.Icon(color="red")).add_to(m2)
                # 无人机当前位置
                folium.Marker(
                    [current_lat, current_lon],
                    popup="无人机",
                    icon=folium.Icon(color="blue", icon="plane", prefix="fa")
                ).add_to(m2)
                st_folium.st_folium(m2, width=1400, height=400)

            else:
                st.info("任务未开始，点击“开始任务”执行飞行模拟。")

save_state()
