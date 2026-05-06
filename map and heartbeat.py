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
from shapely.geometry import Polygon, LineString, Point, MultiLineString, MultiPoint
from shapely.ops import unary_union, split, nearest_points
from shapely import affinity

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
        "heartbeat_data": st.session_state.heartbeat_data[-200:],
        "heartbeat_seq": st.session_state.heartbeat_seq,
        "heartbeat_running": st.session_state.heartbeat_running,
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
        if data.get("flight_start_time"):
            data["flight_start_time"] = datetime.datetime.fromisoformat(data["flight_start_time"])
        return data
    return {}

# ==================== session_state 初始化 ====================
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
        "heartbeat_seq": 0,
        "heartbeat_running": False,
        "seq": 0,
        "running": False,
        "flight_status": "idle",
        "flight_start_time": None,
        "flight_paused_duration": 0.0,
        "flight_speed": 8.5,
        "safety_radius": 5.0,
    }
    loaded = load_state()
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = loaded.get(key, default_value)

ensure_session_state()

# ==================== 辅助函数 ====================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

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

# ==================== 安全绕飞路径算法（左右侧多条拐弯） ====================
def compute_avoid_path(latA, lngA, latB, lngB, fly_height, obstacles, safety_radius_m=5.0):
    if not obstacles:
        return {"left": [], "right": [], "shortest": []}

    center_lon, center_lat = lngA, latA
    start_xy = lonlat_to_xy(lngA, latA, center_lon, center_lat)
    end_xy = lonlat_to_xy(lngB, latB, center_lon, center_lat)

    # 构建所有障碍物缓冲区（米制）
    all_buffers = []
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
        all_buffers.append(poly.buffer(safety_radius_m))

    if not all_buffers:
        # 无高度冲突，直接返回直线
        return {
            "left": [],
            "right": [],
            "shortest": [(latA, lngA), (latB, lngB)]
        }

    merged = unary_union(all_buffers)
    direct_line = LineString([start_xy, end_xy])

    # 若无相交，最短路径就是直线，左右路径也提供一条（可选直线）
    if not direct_line.intersects(merged):
        straight = [(latA, lngA), (latB, lngB)]
        return {"left": straight, "right": straight, "shortest": straight}

    # 获取合并缓冲区的外边界（正方向逆时针）和外边界（孔洞不计）
    if merged.geom_type == 'Polygon':
        boundaries = [merged.exterior]
    elif merged.geom_type == 'MultiPolygon':
        boundaries = [poly.exterior for poly in merged.geoms]
    else:
        boundaries = [merged.boundary] if merged.boundary else []

    # 沿着边界生成绕行路径：对于每个障碍物边界，找出直线与边界的交点，将边界分成两段，提供左右绕行
    left_paths = []
    right_paths = []

    for boundary in boundaries:
        # 计算直线与边界的交点
        intersection = boundary.intersection(direct_line)
        if intersection.is_empty:
            continue
        # 将交点转换为点列表
        if isinstance(intersection, Point):
            pts_list = [intersection]
        elif isinstance(intersection, MultiPoint):
            pts_list = list(intersection.geoms)
        elif isinstance(intersection, (LineString, MultiLineString)):
            # 如果重合，略过
            continue
        else:
            continue

        if len(pts_list) < 2:
            continue

        # 沿直线方向排序交点
        pts_list.sort(key=lambda p: direct_line.project(p))
        # 取第一个和最后一个交点作为切点
        p_entry = pts_list[0]
        p_exit = pts_list[-1]

        coords = list(boundary.coords)

        # 找到 p_entry 和 p_exit 在边界上的位置
        def nearest_index(point, coords):
            min_dist = float('inf')
            idx = 0
            for i, (x, y) in enumerate(coords):
                d = math.hypot(x - point.x, y - point.y)
                if d < min_dist:
                    min_dist = d
                    idx = i
            return idx

        idx_entry = nearest_index(p_entry, coords)
        idx_exit = nearest_index(p_exit, coords)

        # 生成两条路径：顺时针方向和逆时针方向沿着边界
        # 确保索引顺序
        if idx_entry <= idx_exit:
            segment1 = coords[idx_entry:idx_exit+1]  # 从 entry 到 exit 顺时针/逆时针？取决于多边形方向
            segment2 = coords[idx_exit:] + coords[:idx_entry+1]  # 另一边
        else:
            segment1 = coords[idx_entry:] + coords[:idx_exit+1]
            segment2 = coords[idx_exit:idx_entry+1]

        # 判断哪一段是左绕：根据 cross product 矢量叉积，若从起点到终点向量为 dir，则左绕点应在 dir 的左侧
        dir_vec = (end_xy[0]-start_xy[0], end_xy[1]-start_xy[1])
        def side_of_point(pt):
            v = (pt[0]-start_xy[0], pt[1]-start_xy[1])
            return dir_vec[0]*v[1] - dir_vec[1]*v[0]

        # 计算两段中间点的侧向（取非端点的点）
        mid_idx1 = len(segment1)//2
        if len(segment1) < 3:
            side1 = side_of_point(segment1[0])  # 近似
        else:
            side1 = side_of_point(segment1[mid_idx1])

        if side1 > 0:
            left_seg = segment1
            right_seg = segment2
        else:
            left_seg = segment2
            right_seg = segment1

        # 将边界段简化（去除太密集的点，保留拐弯特征）
        def simplify_path(pts, tolerance=1.0):
            if len(pts) <= 2:
                return pts
            # 简单道格拉斯-普克，此处用每隔一定距离采样替代
            result = [pts[0]]
            for i in range(1, len(pts)-1):
                if math.hypot(pts[i][0]-result[-1][0], pts[i][1]-result[-1][1]) >= tolerance:
                    result.append(pts[i])
            result.append(pts[-1])
            return result

        left_seg = simplify_path(left_seg, tolerance=2.0)   # 2米一个拐点
        right_seg = simplify_path(right_seg, tolerance=2.0)

        # 构建完整路径：起点 -> 左侧入口 -> 左侧边界段 -> 出口 -> 终点
        left_path = [start_xy] + [p_entry] + left_seg[1:-1] + [p_exit] + [end_xy]
        right_path = [start_xy] + [p_entry] + right_seg[1:-1] + [p_exit] + [end_xy]

        # 转为经纬度
        def xy_to_latlon_list(xy_list):
            res = []
            for x, y in xy_list:
                lon, lat = xy_to_lonlat(x, y, center_lon, center_lat)
                res.append((lat, lon))
            return res

        left_paths.append(xy_to_latlon_list(left_path))
        right_paths.append(xy_to_latlon_list(right_path))

    # 如果有多个障碍物，路径需要合并（简单串联）
    # 这里简化处理：只取第一个障碍物生成的左右路径
    final_left = left_paths[0] if left_paths else []
    final_right = right_paths[0] if right_paths else []

    # 最短路径：比较左右长度，选择短的
    def path_len(waypoints):
        if not waypoints:
            return float('inf')
        length = 0
        for i in range(len(waypoints)-1):
            length += haversine(waypoints[i][0], waypoints[i][1],
                                waypoints[i+1][0], waypoints[i+1][1])
        return length

    shortest = final_left if path_len(final_left) < path_len(final_right) else final_right

    return {
        "left": final_left,
        "right": final_right,
        "shortest": shortest
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
        st.markdown("### 🛡️ 安全设置")
        safety_radius = st.slider("安全距离 (米)", min_value=1.0, max_value=30.0,
                                  value=st.session_state.safety_radius, step=0.5)
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

        paths = compute_avoid_path(latA, lngA, latB, lngB, fly_h,
                                   st.session_state.obstacles,
                                   safety_radius_m=st.session_state.safety_radius)
        left_pts = paths["left"]
        right_pts = paths["right"]
        shortest_pts = paths["shortest"]

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

        # 向左绕飞（蓝色）
        if left_pts:
            folium.PolyLine(locations=left_pts, color="blue", weight=3, opacity=0.8,
                            popup="向左绕飞路径（安全）").add_to(m)

        # 向右绕飞（绿色）
        if right_pts:
            folium.PolyLine(locations=right_pts, color="green", weight=3, opacity=0.8,
                            popup="向右绕飞路径（安全）").add_to(m)

        # 最短路径（橙色）
        if shortest_pts:
            folium.PolyLine(locations=shortest_pts, color="orange", weight=5, opacity=0.9,
                            popup="最短绕飞路径（推荐）").add_to(m)

        folium.Marker([latA, lngA], popup="起点A", icon=folium.Icon(color="green", icon="info-sign")).add_to(m)
        folium.Marker([latB, lngB], popup="终点B", icon=folium.Icon(color="red", icon="info-sign")).add_to(m)

        # 障碍物
        for ob in st.session_state.obstacles:
            ps = [[lat, lng] for (lng, lat) in ob["points"]]
            folium.Polygon(locations=ps, color="red", fill=True, fill_opacity=0.5,
                           popup=f"{ob['name']} ({ob['height']}m)").add_to(m)

        # 临时打点
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

        st.markdown("---")
        if shortest_pts:
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("🛩️ 使用推荐航线飞行", use_container_width=True):
                    st.session_state.waypoints = shortest_pts
                    st.session_state.flight_status = "idle"
                    save_state()
                    st.success("航线已设置，请切换到“飞行监控”。")
            with col_btn2:
                # 也可以让用户选择左绕或右绕
                pass
        else:
            if st.button("🛩️ 使用直飞航线", use_container_width=True):
                st.session_state.waypoints = [[latA, lngA], [latB, lngB]]
                st.session_state.flight_status = "idle"
                save_state()
                st.success("直飞航线已设置。")

    else:
        # ==================== 飞行监控页面（完整） ====================
        st.markdown("## ✈️ 飞行实时画面 - 任务执行监控")

        # 通信链路
        st.markdown("### 📡 通信链路拓扑与数据流")
        link_cols = st.columns(4)
        with link_cols[0]:
            st.success("🟢 GCS在线")
        with link_cols[1]:
            st.success("🟢 OBC在线")
        with link_cols[2]:
            st.success("🟢 FCU在线")
        with link_cols[3]:
            # 电量单独放后面
            pass

        waypoints = st.session_state.waypoints
        if not waypoints:
            st.warning("⚠️ 尚未设置航线，请先在“航线规划”页面计算并应用航线。")
        else:
            # 控制按钮
            if st.session_state.flight_status == "idle":
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("▶️ 开始任务", use_container_width=True):
                        st.session_state.flight_status = "running"
                        st.session_state.flight_start_time = datetime.datetime.now()
                        st.session_state.flight_paused_duration = 0.0
                        save_state()
                        st.rerun()
                with col2:
                    st.button("⏸️ 暂停任务", disabled=True, use_container_width=True)
            elif st.session_state.flight_status == "running":
                col1, col2 = st.columns(2)
                with col1:
                    st.button("▶️ 开始任务", disabled=True, use_container_width=True)
                with col2:
                    if st.button("⏸️ 暂停任务", use_container_width=True):
                        now = datetime.datetime.now()
                        elapsed = (now - st.session_state.flight_start_time).total_seconds() - st.session_state.flight_paused_duration
                        st.session_state.flight_paused_duration += elapsed
                        st.session_state.flight_status = "paused"
                        save_state()
                        st.rerun()
            elif st.session_state.flight_status == "paused":
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("▶️ 继续任务", use_container_width=True):
                        st.session_state.flight_status = "running"
                        save_state()
                        st.rerun()
                with col2:
                    if st.button("⏹️ 重置任务", use_container_width=True):
                        st.session_state.flight_status = "idle"
                        st.session_state.flight_start_time = None
                        st.session_state.flight_paused_duration = 0.0
                        save_state()
                        st.rerun()
                with col3:
                    st.button("⏸️ 暂停任务", disabled=True, use_container_width=True)

            # 飞行动态模拟
            if st.session_state.flight_status in ("running", "paused"):
                wp_list = waypoints
                total_dist = 0.0
                seg_dists = []
                for i in range(len(wp_list)-1):
                    d = haversine(wp_list[i][0], wp_list[i][1],
                                  wp_list[i+1][0], wp_list[i+1][1])
                    seg_dists.append(d)
                    total_dist += d

                if st.session_state.flight_status == "running":
                    now = datetime.datetime.now()
                    elapsed = (now - st.session_state.flight_start_time).total_seconds() - st.session_state.flight_paused_duration
                else:
                    elapsed = st.session_state.flight_paused_duration

                flown_dist = min(elapsed * st.session_state.flight_speed, total_dist)
                remaining_dist = total_dist - flown_dist

                # 当前位置
                if total_dist == 0:
                    cur_lat, cur_lon = wp_list[0]
                    seg_idx = 0
                else:
                    cum = 0.0
                    seg_idx = 0
                    for i, d in enumerate(seg_dists):
                        if cum + d >= flown_dist:
                            seg_idx = i
                            progress = (flown_dist - cum) / d if d > 0 else 0
                            lat1, lon1 = wp_list[i]
                            lat2, lon2 = wp_list[i+1]
                            cur_lat = lat1 + (lat2 - lat1) * progress
                            cur_lon = lon1 + (lon2 - lon1) * progress
                            break
                        cum += d
                    else:
                        seg_idx = len(seg_dists) - 1
                        cur_lat, cur_lon = wp_list[-1]

                wp_total = len(wp_list) - 1
                wp_display = f"{min(seg_idx+1, wp_total)}/{wp_total}"

                if st.session_state.flight_speed > 0:
                    rem_seconds = remaining_dist / st.session_state.flight_speed
                else:
                    rem_seconds = 0
                eta = datetime.datetime.now() + datetime.timedelta(seconds=rem_seconds)
                eta_str = eta.strftime("%H:%M:%S")

                elapsed_td = datetime.timedelta(seconds=int(elapsed))
                elapsed_str = str(elapsed_td)

                battery = max(0.0, 100.0 - 100.0 * flown_dist / total_dist) if total_dist > 0 else 100.0

                # 指标面板
                with st.container():
                    metrics = st.columns(5)
                    metrics[0].metric("📍 当前航点", wp_display)
                    metrics[1].metric("⚡ 飞行速度", f"{st.session_state.flight_speed:.1f} m/s")
                    metrics[2].metric("⏱️ 已用时间", elapsed_str)
                    metrics[3].metric("🛣️ 剩余距离", f"{remaining_dist:.1f} m")
                    metrics[4].metric("⏰ 预计到达", eta_str)

                st.markdown(f"""
                <div style="background-color:#f3e5f5; border-radius:10px; padding:10px; margin-bottom:10px;">
                    <strong>🔋 电量模拟</strong>&nbsp;&nbsp;
                    <span style="font-size:1.2em; color:{'red' if battery < 20 else 'green'}">{battery:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)

                progress = flown_dist / total_dist if total_dist > 0 else 1.0
                st.progress(min(progress, 1.0))
                st.caption(f"任务进度：{progress*100:.1f}%")

                # 实时地图
                m2 = folium.Map(location=[cur_lat, cur_lon], zoom_start=17, control_scale=True)
                if len(wp_list) > 1:
                    folium.PolyLine(locations=wp_list, color="orange", weight=3, popup="航线").add_to(m2)
                folium.Marker(wp_list[0], popup="起点", icon=folium.Icon(color="green")).add_to(m2)
                folium.Marker(wp_list[-1], popup="终点", icon=folium.Icon(color="red")).add_to(m2)
                folium.Marker(
                    [cur_lat, cur_lon],
                    popup="无人机",
                    icon=folium.Icon(color="blue", icon="plane", prefix="fa")
                ).add_to(m2)
                st_folium.st_folium(m2, width=1400, height=400)

        # ==================== 心跳监控区域 ====================
        st.markdown("---")
        st.markdown("### 💓 地面站心跳监控")

        col_hb1, col_hb2 = st.columns(2)
        with col_hb1:
            if not st.session_state.heartbeat_running:
                if st.button("▶️ 开始心跳监测", use_container_width=True):
                    st.session_state.heartbeat_running = True
                    save_state()
                    st.rerun()
            else:
                if st.button("⏸️ 停止心跳监测", use_container_width=True):
                    st.session_state.heartbeat_running = False
                    save_state()
                    st.rerun()

        if st.session_state.heartbeat_running:
            st.session_state.heartbeat_seq += 1
            now_str = datetime.datetime.now().strftime("%H:%M:%S")
            st.session_state.heartbeat_data.append({
                "序号": st.session_state.heartbeat_seq,
                "时间": now_str,
                "状态": "正常"
            })
            save_state()

        if st.session_state.heartbeat_data:
            df = pd.DataFrame(st.session_state.heartbeat_data[-50:])
            st.line_chart(df.set_index("时间")["序号"])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("暂无心跳数据，点击“开始心跳监测”记录。")

# ==================== 实时刷新 ====================
if page == "飞行监控" and (st.session_state.flight_status == "running" or st.session_state.heartbeat_running):
    time.sleep(1)
    st.rerun()
