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
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union

# ==================== 页面配置 ====================
st.set_page_config(page_title="南京科技职业学院 - 无人机导航系统", layout="wide")

# ==================== 样式 ====================
st.markdown("""
<style>
.left-panel {background:#f8f9fa; padding:20px; border-radius:10px; height:95vh;}
</style>
""", unsafe_allow_html=True)

# ==================== 持久化（增强版，含心跳数据） ====================
STATE_FILE = "ground_station_state.json"

def save_state():
    state = {
        "obstacles": st.session_state.obstacles,
        "draw_points": st.session_state.draw_points,
        "home_point": st.session_state.home_point,
        "waypoints": st.session_state.waypoints,
        # 心跳记忆字段
        "heartbeat_data": st.session_state.heartbeat_data[-100:],  # 保留最近100条
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

# ==================== 初始化（新增心跳恢复） ====================
if "init" not in st.session_state:
    loaded = load_state()
    st.session_state.obstacles = loaded.get("obstacles", [])
    st.session_state.draw_points = loaded.get("draw_points", [])
    st.session_state.home_point = loaded.get("home_point", [32.2335, 118.7475])
    st.session_state.waypoints = loaded.get("waypoints", [])
    st.session_state.last_click = None
    # 心跳状态恢复
    st.session_state.heartbeat_data = loaded.get("heartbeat_data", [])
    st.session_state.seq = loaded.get("seq", 0)
    st.session_state.running = loaded.get("running", False)
    st.session_state.init = True

# ==================== 绕飞路径计算 ====================
def compute_avoid_path(latA, lngA, latB, lngB, fly_height, obstacles):
    """
    返回绕飞航路点列表 [(lat, lng), ...] （不含首尾）
    若无需绕飞则返回空列表
    """
    start = (lngA, latA)
    end = (lngB, latB)
    line = LineString([start, end])

    # 筛选出需要回避的障碍物（飞行高度 < 障碍物高度，且直线与多边形相交）
    blocking = []
    for ob in obstacles:
        ob_height = ob.get("height", 0)
        if fly_height >= ob_height:
            continue
        # 障碍物点存储为 (lng, lat)，转换为 (lng, lat) 列表
        pts = ob["points"]  # [(lng, lat), ...]
        if len(pts) < 3:
            continue
        coords = pts.copy()
        if coords[0] != coords[-1]:
            coords.append(coords[0])  # 闭合
        poly = Polygon(coords)
        if line.intersects(poly):
            blocking.append((poly, ob_height))

    if not blocking:
        return []

    # 合并所有障碍物
    merged = unary_union([b[0] for b in blocking])
    if merged.geom_type == "MultiPolygon":
        polys = list(merged.geoms)
    else:
        polys = [merged]

    # 左右侧最远点
    def side_points(poly):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        left_pts, right_pts = [], []
        for (x, y) in poly.exterior.coords:
            vx = x - start[0]
            vy = y - start[1]
            cross = dx * vy - dy * vx
            if cross > 1e-9:
                left_pts.append((cross, (x, y)))
            elif cross < -1e-9:
                right_pts.append((-cross, (x, y)))
        best_left = max(left_pts, key=lambda t: t[0])[1] if left_pts else None
        best_right = max(right_pts, key=lambda t: t[0])[1] if right_pts else None
        return best_left, best_right

    left_candidates, right_candidates = [], []
    for poly in polys:
        l, r = side_points(poly)
        if l:
            left_candidates.append(l)
        if r:
            right_candidates.append(r)

    # 按距离起点排序
    def dist(p):
        return math.hypot(p[0]-start[0], p[1]-start[1])
    left_candidates.sort(key=dist)
    right_candidates.sort(key=dist)

    left_path = [start] + left_candidates + [end] if left_candidates else None
    right_path = [start] + right_candidates + [end] if right_candidates else None

    if left_path and right_path:
        len_l = sum(dist(left_path[i+1]) for i in range(len(left_path)-1))
        len_r = sum(dist(right_path[i+1]) for i in range(len(right_path)-1))
        chosen = left_path if len_l < len_r else right_path
    elif left_path:
        chosen = left_path
    else:
        chosen = right_path

    if not chosen:
        return []

    # 去掉首尾，返回 (lat, lng) 格式
    waypoints = [(p[1], p[0]) for p in chosen[1:-1]]
    return waypoints

# ==================== 左侧布局 ====================
col_left, col_right = st.columns([1, 3])

with col_left:
    st.markdown('<div class="left-panel">', unsafe_allow_html=True)
    st.subheader("🧭 导航")
    page = st.radio("", ["航线规划", "飞行监控"], label_visibility="collapsed")
    st.divider()

    if page == "航线规划":
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
        # ==================== AB点 ====================
        st.markdown("### 🎯 AB点航线")
        c1, c2 = st.columns(2)
        with c1:
            latA = st.number_input("起点A纬度", value=32.233500, format="%.6f")
            lngA = st.number_input("起点A经度", value=118.747500, format="%.6f")
        with c2:
            latB = st.number_input("终点B纬度", value=32.233800, format="%.6f")
            lngB = st.number_input("终点B经度", value=118.747900, format="%.6f")

        fly_h = st.number_input("飞行高度(m)", min_value=1, max_value=500, value=50, step=1)

        # ==================== 地图模式切换 ====================
        map_type = st.radio("🗺️ 地图模式", ["高德普通地图", "卫星影像地图"], horizontal=True)

        # ==================== 计算绕飞路径 ====================
        avoid_pts = compute_avoid_path(latA, lngA, latB, lngB, fly_h, st.session_state.obstacles)

        # ==================== 初始化地图 ====================
        center_lat = (latA + latB) / 2
        center_lng = (lngA + lngB) / 2
        m = folium.Map(location=[center_lat, center_lng], zoom_start=17, control_scale=True)

        # ==================== 图层 ====================
        if map_type == "卫星影像地图":
            TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri",
                name="卫星影像",
                max_zoom=20
            ).add_to(m)
        else:
            TileLayer(
                tiles="https://webrd01.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}",
                attr="© 高德",
                name="高德地图",
                max_zoom=20
            ).add_to(m)

        # ==================== 原始航线（红色虚线） ====================
        folium.PolyLine(
            locations=[[latA, lngA], [latB, lngB]],
            color="red",
            weight=2,
            opacity=0.6,
            dash_array="5,5"
        ).add_to(m)

        # ==================== 绕飞路径（黄色实线） ====================
        if avoid_pts:
            path = [[latA, lngA]] + [[lat, lng] for (lat, lng) in avoid_pts] + [[latB, lngB]]
            folium.PolyLine(
                locations=path,
                color="orange",
                weight=5,
                opacity=0.9
            ).add_to(m)

        # ==================== AB点标记 ====================
        folium.Marker(
            [latA, lngA],
            popup="起点A",
            icon=folium.Icon(color="green", icon="info-sign")
        ).add_to(m)

        folium.Marker(
            [latB, lngB],
            popup="终点B",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)

        # ==================== 障碍物（红色半透明） ====================
        for ob in st.session_state.obstacles:
            ps = [[lat, lng] for (lng, lat) in ob["points"]]
            folium.Polygon(
                locations=ps,
                color="red",
                fill=True,
                fill_opacity=0.5,
                popup=f"{ob['name']} ({ob['height']}m)"
            ).add_to(m)

        # ==================== 当前圈选临时多边形 ====================
        if len(st.session_state.draw_points) >= 2:
            ps = [[lat, lng] for (lng, lat) in st.session_state.draw_points]
            folium.Polygon(
                locations=ps,
                color="blue",
                fill=True,
                fill_opacity=0.2
            ).add_to(m)

        # ==================== 地图渲染 ====================
        o = st_folium.st_folium(
            m,
            width=1400,
            height=700,
            returned_objects=["last_clicked"]
        )

        # ==================== 点击打点 ====================
        if o and o.get("last_clicked"):
            lat = o["last_clicked"]["lat"]
            lng = o["last_clicked"]["lng"]
            pt = (round(lng, 6), round(lat, 6))
            if pt != st.session_state.last_click:
                st.session_state.last_click = pt
                st.session_state.draw_points.append(pt)
                save_state()
                st.rerun()

    else:
        # ==================== 心跳监控（带记忆） ====================
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

        # 循环监测（同时保存状态）
        while True:
            if st.session_state.running:
                st.session_state.seq += 1
                t = datetime.datetime.now().strftime("%H:%M:%S")
                st.session_state.heartbeat_data.append({
                    "序号": st.session_state.seq,
                    "时间": t,
                    "状态": "正常"
                })
                # 每次更新保存状态（限制频率以避免过多写入，但这里每条都保存问题不大）
                save_state()
                df = pd.DataFrame(st.session_state.heartbeat_data)
                with placeholder.container():
                    st.line_chart(df, x="时间", y="序号")
                    st.dataframe(df, use_container_width=True)
                time.sleep(1)
            else:
                # 暂停时显示现有数据
                if st.session_state.heartbeat_data:
                    df = pd.DataFrame(st.session_state.heartbeat_data)
                    with placeholder.container():
                        st.line_chart(df, x="时间", y="序号")
                        st.dataframe(df, use_container_width=True)
                else:
                    with placeholder.container():
                        st.info("暂无监控数据")
                time.sleep(0.5)
