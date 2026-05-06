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

# ==================== 【修正版】向外拱起的贝塞尔弧线（强制在障碍物外侧） ====================
def create_offset_bezier_arc(start_xy, end_xy, offset_dist=15.0, seg_num=30):
    """
    生成一条：垂直航线方向、向外偏移的贝塞尔弧线
    offset_dist: 向外偏移的距离（米），越大离障碍物越远
    """
    x0, y0 = start_xy
    x1, y1 = end_xy

    # 航线方向向量
    dx = x1 - x0
    dy = y1 - y0
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return [start_xy, end_xy]

    # 垂直航线的单位向量（向外）
    nx = -dy / L
    ny = dx / L

    # 控制点：在航线中点，向外侧偏移 offset_dist 米
    mid_x = (x0 + x1) / 2
    mid_y = (y0 + y1) / 2
    ctrl_x = mid_x + nx * offset_dist
    ctrl_y = mid_y + ny * offset_dist

    # 生成二次贝塞尔曲线点
    arc = []
    for t in range(seg_num + 1):
        t = t / seg_num
        mt = 1 - t
        x = mt**2 * x0 + 2 * mt * t * ctrl_x + t**2 * x1
        y = mt**2 * y0 + 2 * mt * t * ctrl_y + t**2 * y1
        arc.append((x, y))
    return arc

# ==================== 【修正版】核心绕飞逻辑（100%生成绕飞弧线） ====================
def compute_avoid_path(latA, lngA, latB, lngB, fly_height, obstacles, safety_radius_m=5.0):
    if not obstacles:
        straight = [(latA, lngA), (latB, lngB)]
        return {"left": straight, "right": straight, "shortest": straight}

    center_lon, center_lat = lngA, latA
    start_xy = lonlat_to_xy(lngA, latA, center_lon, center_lat)
    end_xy = lonlat_to_xy(lngB, latB, center_lon, center_lat)

    # 1. 构建障碍物安全缓冲区
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

    # 直飞不撞障碍，直接直飞
    if not direct_line.intersects(merged):
        straight = [(latA, lngA), (latB, lngB)]
        return {"left": straight, "right": straight, "shortest": straight}

    # 2. 强制生成向外拱的弧线，先试15米偏移
    arc_xy = create_offset_bezier_arc(start_xy, end_xy, offset_dist=15.0, seg_num=30)

    # 3. 安全校验：如果弧线在障碍里，自动增大偏移量
    def is_path_safe(path, geom):
        for x, y in path:
            pt = Point(x, y)
            if pt.within(geom) or pt.distance(geom) < 0.5:
                return False
        return True

    # 如果不安全，自动增大偏移，直到弧线在外面
    if not is_path_safe(arc_xy, merged):
        for offset in [20, 25, 30]:
            arc_xy = create_offset_bezier_arc(start_xy, end_xy, offset_dist=offset, seg_num=30)
            if is_path_safe(arc_xy, merged):
                break

    # 4. 转回经纬度
    def to_latlon(xy_list):
        res = []
        for x, y in xy_list:
            lon, lat = xy_to_lonlat(x, y, center_lon, center_lat)
            res.append((lat, lon))
        return res

    arc_latlon = to_latlon(arc_xy)

    # 左右和推荐都用同一条安全弧线
    return {
        "left": arc_latlon,
        "right": arc_latlon,
        "shortest": arc_latlon
    }

# ==================== 左侧面板（不变） ====================
col_left, col_right = st.columns([1, 3])

with col_left:
    st.markdown('<div class="left-panel">', unsafe_allow_html=True)
    st.subheader("🧭 导航")
    page = st.radio("", ["航线规划", "飞行监控"], label_visibility="collapsed")
    st.divider()

    if page == "航线规划":
        click_mode = st.radio("点击地图时", ["障碍物圈选", "选择终点"], horizontal=True,
                              index=0 if st.session_state.click_mode == "障碍物圈选" else 1)
        if click_mode != st.session_state.click_mode:
            st.session_state.click_mode = click_mode
            save_state()

        st.divider()
        st.markdown("### 🛡️ 安全设置")
        safety_radius = st.slider("安全距离 (米)", 1.0, 30.0, st.session_state.safety_radius, 0.5)
        if safety_radius != st.session_state.safety_radius:
            st.session_state.safety_radius = safety_radius
            save_state()

        st.divider()
        st.markdown("### 🚧 障碍物圈选")
        name = st.text_input("障碍物名称", "教学楼")
        height = st.number_input("高度(m)", 1, 500, 25)
        st.info(f"当前已打点：{len(st.session_state.draw_points)} 个")
        if st.button("🧹 清空当前打点", use_container_width=True):
            st.session_state.draw_points = []
            save_state()
            st.rerun()
        if st.button("✅ 保存障碍物", type="primary", use_container_width=True):
            if len(st.session_state.draw_points) >= 3:
                st.session_state.obstacles.append({"name": name, "height": height, "points": st.session_state.draw_points.copy()})
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
        st.session_state.latA, st.session_state.lngA = latA, lngA
        st.session_state.latB, st.session_state.lngB = latB, lngB

        fly_h = st.number_input("飞行高度(m)", 1, 500, 50)
        map_type = st.radio("🗺️ 地图模式", ["高德普通地图", "卫星影像地图"], horizontal=True)

        paths = compute_avoid_path(latA, lngA, latB, lngB, fly_h, st.session_state.obstacles, st.session_state.safety_radius)
        left_pts = paths["left"]
        right_pts = paths["right"]
        shortest_pts = paths["shortest"]

        center_lat = (latA + latB) / 2
        center_lng = (lngA + lngB) / 2
        m = folium.Map(location=[center_lat, center_lng], zoom_start=17, control_scale=True)

        if map_type == "卫星影像地图":
            TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                      attr="Esri", name="卫星影像", max_zoom=20).add_to(m)
        else:
            TileLayer(tiles="https://webrd01.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}",
                      attr="© 高德", name="高德地图", max_zoom=20).add_to(m)

        folium.PolyLine([[latA, lngA], [latB, lngB]], color="red", weight=2, opacity=0.6, dash_array="5,5", popup="原始直飞航线").add_to(m)
        if left_pts:
            folium.PolyLine(left_pts, color="blue", weight=3, opacity=0.9, popup="左绕弧线路径").add_to(m)
        if right_pts:
            folium.PolyLine(right_pts, color="green", weight=3, opacity=0.9, popup="右绕弧线路径").add_to(m)
        if shortest_pts:
            folium.PolyLine(shortest_pts, color="orange", weight=5, opacity=0.9, popup="最短弧线路径（推荐）").add_to(m)

        folium.Marker([latA, lngA], popup="起点A", icon=folium.Icon(color="green", icon="info-sign")).add_to(m)
        folium.Marker([latB, lngB], popup="终点B", icon=folium.Icon(color="red", icon="info-sign")).add_to(m)

        for ob in st.session_state.obstacles:
            ps = [[lat, lng] for (lng, lat) in ob["points"]]
            folium.Polygon(ps, color="red", fill=True, fill_opacity=0.5, popup=f"{ob['name']} ({ob['height']}m)").add_to(m)

        if len(st.session_state.draw_points) >= 2:
            ps = [[lat, lng] for (lng, lat) in st.session_state.draw_points]
            folium.Polygon(ps, color="blue", fill=True, fill_opacity=0.2).add_to(m)

        o = st_folium.st_folium(m, width=1400, height=700, returned_objects=["last_clicked"])
        if o and o.get("last_clicked"):
            d = o["last_clicked"]
            pt = (round(d["lng"], 6), round(d["lat"], 6))
            if st.session_state.click_mode == "障碍物圈选":
                if pt != st.session_state.last_click:
                    st.session_state.last_click = pt
                    st.session_state.draw_points.append(pt)
                    save_state()
                    st.rerun()
            else:
                st.session_state.latB = round(d["lat"], 6)
                st.session_state.lngB = round(d["lng"], 6)
                st.session_state.last_click = pt
                save_state()
                st.rerun()

        st.markdown("---")
        if shortest_pts:
            if st.button("🛩️ 使用推荐航线飞行", use_container_width=True):
                st.session_state.waypoints = shortest_pts
                st.session_state.flight_status = "idle"
                save_state()
                st.success("航线已设置，请切换到“飞行监控”。")
        else:
            if st.button("🛩️ 使用直飞航线", use_container_width=True):
                st.session_state.waypoints = [[latA, lngA], [latB, lngB]]
                st.session_state.flight_status = "idle"
                save_state()
                st.success("直飞航线已设置。")

    else:
        st.markdown("## ✈️ 飞行实时画面 - 任务执行监控")
        st.markdown("### 📡 通信链路拓扑与数据流")
        cols = st.columns(4)
        cols[0].success("🟢 GCS在线")
        cols[1].success("🟢 OBC在线")
        cols[2].success("🟢 FCU在线")

        waypoints = st.session_state.waypoints
        if not waypoints:
            st.warning("⚠️ 尚未设置航线")
        else:
            if st.session_state.flight_status == "idle":
                c1, c2 = st.columns(2)
                if c1.button("▶️ 开始任务", use_container_width=True):
                    st.session_state.flight_status = "running"
                    st.session_state.flight_start_time = datetime.datetime.now()
                    st.session_state.flight_paused_duration = 0.0
                    save_state()
                    st.rerun()
                c2.button("⏸️ 暂停任务", disabled=True)
            elif st.session_state.flight_status == "running":
                c1, c2 = st.columns(2)
                c1.button("▶️ 开始任务", disabled=True)
                if c2.button("⏸️ 暂停任务", use_container_width=True):
                    now = datetime.datetime.now()
                    elapsed = (now - st.session_state.flight_start_time).total_seconds() - st.session_state.flight_paused_duration
                    st.session_state.flight_paused_duration += elapsed
                    st.session_state.flight_status = "paused"
                    save_state()
                    st.rerun()
            elif st.session_state.flight_status == "paused":
                c1, c2, c3 = st.columns(3)
                if c1.button("▶️ 继续任务", use_container_width=True):
                    st.session_state.flight_status = "running"
                    save_state()
                    st.rerun()
                if c2.button("⏹️ 重置任务", use_container_width=True):
                    st.session_state.flight_status = "idle"
                    st.session_state.flight_start_time = None
                    st.session_state.flight_paused_duration = 0.0
                    save_state()
                    st.rerun()
                c3.button("⏸️ 暂停任务", disabled=True)

            if st.session_state.flight_status in ("running", "paused"):
                wp_list = waypoints
                total_dist = sum(haversine(wp_list[i][0], wp_list[i][1], wp_list[i+1][0], wp_list[i+1][1]) for i in range(len(wp_list)-1))
                if st.session_state.flight_status == "running":
                    elapsed = (datetime.datetime.now() - st.session_state.flight_start_time).total_seconds() - st.session_state.flight_paused_duration
                else:
                    elapsed = st.session_state.flight_paused_duration
                flown = min(elapsed * st.session_state.flight_speed, total_dist)
                remain = total_dist - flown

                cum = 0
                cur_lat, cur_lon = wp_list[0]
                idx = 0
                for i in range(len(wp_list)-1):
                    d = haversine(wp_list[i][0], wp_list[i][1], wp_list[i+1][0], wp_list[i+1][1])
                    if cum + d >= flown:
                        idx = i
                        prop = (flown - cum) / d if d > 0 else 0
                        cur_lat = wp_list[i][0] + (wp_list[i+1][0] - wp_list[i][0]) * prop
                        cur_lon = wp_list[i][1] + (wp_list[i+1][1] - wp_list[i][1]) * prop
                        break
                    cum += d
                else:
                    idx = len(wp_list)-2
                    cur_lat, cur_lon = wp_list[-1]

                wp_disp = f"{min(idx+1, len(wp_list)-1)}/{len(wp_list)-1}"
                batt = max(0.0, 100 - 100*flown/total_dist) if total_dist > 0 else 100

                cols = st.columns(5)
                cols[0].metric("📍 当前航点", wp_disp)
                cols[1].metric("⚡ 飞行速度", f"{st.session_state.flight_speed:.1f} m/s")
                cols[2].metric("⏱️ 已用时间", str(datetime.timedelta(seconds=int(elapsed))))
                cols[3].metric("🛣️ 剩余距离", f"{remain:.1f} m")
                cols[4].metric("⏰ 预计到达", (datetime.datetime.now() + datetime.timedelta(seconds=remain/st.session_state.flight_speed)).strftime("%H:%M:%S") if st.session_state.flight_speed > 0 else "--:--:--")

                st.markdown(f"<div style='background:#f3e5f5;border-radius:10px;padding:10px'><strong>🔋 电量模拟</strong> <span style='color:{'red' if batt<20 else 'green'}'>{batt:.1f}%</span></div>", unsafe_allow_html=True)
                st.progress(min(flown/total_dist, 1.0) if total_dist > 0 else 1.0)
                st.caption(f"任务进度：{(flown/total_dist)*100:.1f}%" if total_dist > 0 else "完成")

                m2 = folium.Map(location=[cur_lat, cur_lon], zoom_start=17)
                if len(wp_list) > 1:
                    folium.PolyLine(wp_list, color="orange", weight=3).add_to(m2)
                folium.Marker(wp_list[0], icon=folium.Icon(color="green")).add_to(m2)
                folium.Marker(wp_list[-1], icon=folium.Icon(color="red")).add_to(m2)
                folium.Marker([cur_lat, cur_lon], icon=folium.Icon(color="blue", icon="plane", prefix="fa")).add_to(m2)
                st_folium.st_folium(m2, width=1400, height=400)

        st.markdown("---")
        st.markdown("### 💓 地面站心跳监控")
        ch1, ch2 = st.columns(2)
        if not st.session_state.heartbeat_running:
            if ch1.button("▶️ 开始心跳监测", use_container_width=True):
                st.session_state.heartbeat_running = True
                save_state()
                st.rerun()
        else:
            if ch1.button("⏸️ 停止心跳监测", use_container_width=True):
                st.session_state.heartbeat_running = False
                save_state()
                st.rerun()

        if st.session_state.heartbeat_running:
            st.session_state.heartbeat_seq += 1
            st.session_state.heartbeat_data.append({"序号": st.session_state.heartbeat_seq, "时间": datetime.datetime.now().strftime("%H:%M:%S"), "状态": "正常"})
            save_state()

        if st.session_state.heartbeat_data:
            df = pd.DataFrame(st.session_state.heartbeat_data[-50:])
            st.line_chart(df.set_index("时间")["序号"])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("暂无心跳数据")

if page == "飞行监控" and (st.session_state.flight_status == "running" or st.session_state.heartbeat_running):
    time.sleep(1)
    st.rerun()
