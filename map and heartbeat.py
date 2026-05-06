streamlit as st
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
        "running": st.session_state.running
    }
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
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
    }
    loaded = load_state()
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = loaded.get(key, default_value)
    if "init" not in st.session_state:
        st.session_state.init = True

ensure_session_state()

# ==================== 绕飞路径计算（支持左绕、右绕、真正的弧线最短路径） ====================
def compute_avoid_path(latA, lngA, latB, lngB, fly_height, obstacles, turn_radius=0.0002):
    start = (lngA, latA)
    end = (lngB, latB)
    line = LineString([start, end])

    blocking = []
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
        if line.intersects(poly):
            blocking.append((poly, ob_height))

    if not blocking:
        return {"left": [], "right": [], "shortest": []}

    merged = unary_union([b[0] for b in blocking])
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

    # 生成平滑弧线路径（基于二次贝塞尔曲线，真正的弧线）
    def create_bezier_arc_path(path_pts, control_scale=0.5):
        if not path_pts or len(path_pts) < 3:
            return []
        arc_path = []
        # 起点
        arc_path.append(path_pts[0])
        # 对每一段路径生成弧线
        for i in range(1, len(path_pts)-1):
            p0 = path_pts[i-1]
            p1 = path_pts[i]
            p2 = path_pts[i+1]
            
            # 计算控制点
            dx1 = p1[0] - p0[0]
            dy1 = p1[1] - p0[1]
            dx2 = p2[0] - p1[0]
            dy2 = p2[1] - p1[1]
            
            # 控制点取在角平分线上，生成平滑过渡
            control_x = p1[0] - (dx1 + dx2) * control_scale
            control_y = p1[1] - (dy1 + dy2) * control_scale
            
            # 生成贝塞尔曲线点
            steps = 10
            for t in range(1, steps+1):
                t = t / steps
                # 二次贝塞尔公式
                x = (1-t)**2 * p0[0] + 2*(1-t)*t * control_x + t**2 * p2[0]
                y = (1-t)**2 * p0[1] + 2*(1-t)*t * control_y + t**2 * p2[1]
                arc_path.append((x, y))
        # 终点
        arc_path.append(path_pts[-1])
        return arc_path

    left_arc = create_bezier_arc_path(left_path) if left_path else []
    right_arc = create_bezier_arc_path(right_path) if right_path else []

    # 选择最短路径
    def path_len(p):
        if not p:
            return float('inf')
        return sum(math.hypot(p[i+1][0]-p[i][0], p[i+1][1]-p[i][1]) for i in range(len(p)-1))

    len_l = path_len(left_arc)
    len_r = path_len(right_arc)
    shortest_path = left_arc if len_l < len_r else right_arc

    # 转为 (lat, lng) 格式
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
        # 地图点击模式选择
        st.markdown("### 🖱️ 地图点击用途")
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

        # 障碍物圈选
        st.markdown("### 🚧 障碍物圈选（永久记忆）")
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
                st.success("✅ 保存成功！永久显示！")
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
        # AB点坐标（已移除固定的key，保证与点击终点同步）
        st.markdown("### 🎯 AB点航线")
        c1, c2 = st.columns(2)
        with c1:
            latA = st.number_input("起点A纬度", value=st.session_state.latA, format="%.6f")
            lngA = st.number_input("起点A经度", value=st.session_state.lngA, format="%.6f")
        with c2:
            latB = st.number_input("终点B纬度", value=st.session_state.latB, format="%.6f")
            lngB = st.number_input("终点B经度", value=st.session_state.lngB, format="%.6f")

        # 双向同步
        st.session_state.latA = latA
        st.session_state.lngA = lngA
        st.session_state.latB = latB
        st.session_state.lngB = lngB

        fly_h = st.number_input("飞行高度(m)", min_value=1, max_value=500, value=50, step=1)
        map_type = st.radio("🗺️ 地图模式", ["高德普通地图", "卫星影像地图"], horizontal=True)

        # 计算三种绕飞路径
        paths = compute_avoid_path(latA, lngA, latB, lngB, fly_h, st.session_state.obstacles)
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

        # 向左绕飞路径（蓝色）
        if left_pts:
            path = [[latA, lngA]] + [[lat, lng] for (lat, lng) in left_pts] + [[latB, lngB]]
            folium.PolyLine(locations=path, color="blue", weight=3, opacity=0.8, popup="向左绕飞路径").add_to(m)

        # 向右绕飞路径（绿色）
        if right_pts:
            path = [[latA, lngA]] + [[lat, lng] for (lat, lng) in right_pts] + [[latB, lngB]]
            folium.PolyLine(locations=path, color="green", weight=3, opacity=0.8, popup="向右绕飞路径").add_to(m)

        # 最短弧线路径（橙色，真正的弧线）
        if shortest_pts:
            path = [[lat, lng] for (lat, lng) in shortest_pts]
            folium.PolyLine(locations=path, color="orange", weight=5, opacity=0.9, popup="最短弧线路径（推荐）").add_to(m)

        folium.Marker([latA, lngA], popup="起点A", icon=folium.Icon(color="green", icon="info-sign")).add_to(m)
        folium.Marker([latB, lngB], popup="终点B", icon=folium.Icon(color="red", icon="info-sign")).add_to(m)

        # 障碍物
        for ob in st.session_state.obstacles:
            ps = [[lat, lng] for (lng, lat) in ob["points"]]
            folium.Polygon(locations=ps, color="red", fill=True, fill_opacity=0.5,
                           popup=f"{ob['name']} ({ob['height']}m)").add_to(m)

        # 当前圈选临时多边形
        if len(st.session_state.draw_points) >= 2:
            ps = [[lat, lng] for (lng, lat) in st.session_state.draw_points]
            folium.Polygon(locations=ps, color="blue", fill=True, fill_opacity=0.2).add_to(m)

        # 地图渲染
        o = st_folium.st_folium(m, width=1400, height=700, returned_objects=["last_clicked"])

        # 处理点击
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
            else:  # 选择终点模式
                st.session_state.latB = round(click_lat, 6)
                st.session_state.lngB = round(click_lng, 6)
                st.session_state.last_click = pt
                save_state()
                st.rerun()

    else:
        # 心跳监控（带记忆）
        st.title("📡 无人机心跳监控")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("▶️ 开始监测", use_container_width=True):
                st.session_state.running = True
                save_state()
        with c2:
            if st.button("⏸️ 暂停监测", use_container_width=True):
                st.session_state.running = False
                save_state()

        placeholder = st.empty()
        while True:
            if st.session_state.running:
                st.session_state.seq += 1
                t = datetime.datetime.now().strftime("%H:%M:%S")
                st.session_state.heartbeat_data.append({
                    "序号": st.session_state.seq,
                    "时间": t,
                    "状态": "正常"
                })
                save_state()
                df = pd.DataFrame(st.session_state.heartbeat_data)
                with placeholder.container():
                    st.line_chart(df, x="时间", y="序号")
                    st.dataframe(df, use_container_width=True)
                time.sleep(1)
            else:
                if st.session_state.heartbeat_data:
                    df = pd.DataFrame(st.session_state.heartbeat_data)
                    with placeholder.container():
                        st.line_chart(df, x="时间", y="序号")
                        st.dataframe(df, use_container_width=True)
                else:
                    with placeholder.container():
                        st.info("暂无监控数据")
                time.sleep(0.5)
