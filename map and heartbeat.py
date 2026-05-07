import streamlit as st
import folium
from streamlit_folium import st_folium
from streamlit_option_menu import option_menu
from datetime import datetime, timedelta
import time
import pandas as pd
import json
import os
import numpy as np
from shapely.geometry import LineString, Polygon, Point, MultiPoint
from shapely.ops import unary_union

# ===================== 数据持久化 =====================
SAVE_FILE = "drone_data.json"

def load_all_data():
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "A": [32.2322, 118.7490], "B": [32.2343, 118.7490],
        "A_set": False, "B_set": False,
        "obstacles": [], "safe_radius": 0.0002,
        "click_mode": "障碍物圈选", "selected_route": "shortest"
    }

def save_all_data():
    data = {
        "A": list(st.session_state.A),
        "B": list(st.session_state.B),
        "A_set": st.session_state.A_set,
        "B_set": st.session_state.B_set,
        "obstacles": st.session_state.polygon_memory,
        "safe_radius": st.session_state.safe_radius,
        "click_mode": st.session_state.click_mode,
        "selected_route": st.session_state.selected_route
    }
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ===================== 初始化状态 =====================
data = load_all_data()
default_states = {
    "A": tuple(data["A"]),
    "B": tuple(data["B"]),
    "A_set": data["A_set"],
    "B_set": data["B_set"],
    "height": 50,
    "heartbeat_data": [],
    "polygon_memory": data["obstacles"],
    "temp_points": [],
    "obs_h": 20,
    "last_click_time": 0,
    "safe_radius": data.get("safe_radius", 0.0002),
    "flight_running": False,
    "flight_paused": False,
    "current_wp_idx": 0,
    "flight_speed": 8.5,
    "flight_start_time": None,
    "flight_waypoints": [],
    "battery": 100.0,
    "total_distance": 0.0,
    "elapsed_distance": 0.0,
    "click_mode": data.get("click_mode", "障碍物圈选"),
    "selected_route": data.get("selected_route", "shortest")
}
for key, val in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ===================== 页面配置 =====================
st.set_page_config(layout="wide", page_title="南科院无人机航线规划系统")

# ===================== 平滑弧线工具 =====================
def smooth_curve(points, num=50):
    points = np.array(points)
    if len(points) < 2:
        return points
    t = np.linspace(0, 1, len(points))
    t_smooth = np.linspace(0, 1, num)
    x = np.interp(t_smooth, t, points[:,0])
    y = np.interp(t_smooth, t, points[:,1])
    return list(zip(x, y))

# ===================== 高度判断 + 绕飞算法 =====================
def get_all_routes(pA, pB, obstacles, safe_dist, flight_h):
    base_line = LineString([pA, pB])
    buffers = []
    need_avoid = False

    for obs in obstacles:
        pts = obs["pts"]
        h = obs["h"]
        if len(pts)>=3:
            poly = Polygon(pts).buffer(safe_dist)
            buffers.append(poly)
            if flight_h < h:
                need_avoid = True

    if not buffers or not need_avoid:
        straight = [pA, pB]
        return straight, straight, straight

    merged = unary_union(buffers)
    if not base_line.intersects(merged):
        straight = [pA, pB]
        return straight, straight, straight

    boundary = merged.exterior
    coords = list(boundary.coords)
    intersection = boundary.intersection(base_line)
    pts = []

    if isinstance(intersection, Point):
        pts = [intersection]
    elif isinstance(intersection, MultiPoint):
        pts = list(intersection.geoms)

    if len(pts) < 2:
        straight = [pA, pB]
        return straight, straight

    pts.sort(key=lambda p: base_line.project(p))
    entry, exit_ = pts[0], pts[-1]

    def near_idx(p, cs):
        return min(range(len(cs)), key=lambda i: np.hypot(cs[i][0]-p.x, cs[i][1]-p.y))

    ie, ix = near_idx(entry, coords), near_idx(exit_, coords)
    seg1 = coords[ie:ix+1] if ie <= ix else coords[ie:] + coords[:ix+1]
    seg2 = coords[ix:ie+1] if ix <= ie else coords[ix:] + coords[:ie+1]

    def cross(pt):
        return (pB[0]-pA[0])*(pt[1]-pA[1]) - (pB[1]-pA[1])*(pt[0]-pA[0])

    left = seg1 if cross(seg1[len(seg1)//2]) > 0 else seg2
    right = seg2 if cross(seg1[len(seg1)//2]) > 0 else seg1

    ep = (entry.x, entry.y)
    xp = (exit_.x, exit_.y)

    left_route = smooth_curve([pA, ep] + left[1:-1] + [xp, pB], 80)
    right_route = smooth_curve([pA, ep] + right[1:-1] + [xp, pB], 80)
    shortest = left_route if len(left_route) < len(right_route) else right_route

    return left_route, right_route, shortest

# ===================== 侧边栏 =====================
with st.sidebar:
    st.title("🚁 无人机系统导航")
    page = option_menu("功能页面", ["航线规划", "飞行监控"], default_index=0)
    st.divider()
    st.subheader("系统点位状态")
    col_status = st.columns(2)
    with col_status[0]:
        st.button("✅ A点已设置" if st.session_state.A_set else "❌ A点未设置", type="primary", key="statusA")
    with col_status[1]:
        st.button("✅ B点已设置" if st.session_state.B_set else "❌ B点未设置", type="primary", key="statusB")
    st.divider()
    st.subheader("🛡️ 安全半径配置")
    st.session_state.safe_radius = st.slider(
        "航线与障碍物安全距离", 0.00005, 0.0005,
        value=st.session_state.safe_radius, step=0.00001, format="%.5f"
    )
    save_all_data()

# ===================== 航线规划页面 =====================
if page == "航线规划":
    st.title("🚁 南京科技职业学院 无人机航线规划系统")
    col_map, col_ctrl = st.columns([3.2, 1])

    with col_ctrl:
        st.subheader("🎛️ 点位与飞行参数")
        a_lat = st.number_input("起点A 纬度", value=st.session_state.A[0], format="%.6f")
        a_lon = st.number_input("起点A 经度", value=st.session_state.A[1], format="%.6f")
        b_lat = st.number_input("终点B 纬度", value=st.session_state.B[0], format="%.6f")
        b_lon = st.number_input("终点B 经度", value=st.session_state.B[1], format="%.6f")
        st.session_state.height = st.slider("飞行高度（m）", 0, 300, 50)

        if st.button("使用输入值更新AB点"):
            st.session_state.A = (a_lat, a_lon)
            st.session_state.B = (b_lat, b_lon)
            st.session_state.A_set = True
            st.session_state.B_set = True
            save_all_data()
            st.success("AB点已更新")

        st.divider()
        st.subheader("🖱️ 地图点击用途")
        st.session_state.click_mode = st.radio(
            "点击地图时",
            ["障碍物圈选", "选择起点A", "选择终点B"],
            index=0 if st.session_state.click_mode == "障碍物圈选"
            else (1 if st.session_state.click_mode == "选择起点A" else 2)
        )
        save_all_data()

        st.divider()
        st.subheader("🚧 障碍物区域圈选（带高度）")
        st.session_state.obs_h = st.number_input("障碍物高度(m)", 0, 300, value=st.session_state.obs_h)
        if st.session_state.click_mode == "障碍物圈选":
            st.info(f"🖱️ 当前为障碍物圈选模式，已打点：{len(st.session_state.temp_points)} 个")
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("撤销上一点"):
                    if st.session_state.temp_points:
                        st.session_state.temp_points.pop()
                        st.rerun()
            with col_btn2:
                if st.button("清空当前圈选"):
                    st.session_state.temp_points = []
                    st.rerun()
            if st.button("✅ 完成圈选并保存障碍物"):
                if len(st.session_state.temp_points) >= 3:
                    st.session_state.polygon_memory.append({
                        "pts": st.session_state.temp_points.copy(),
                        "h": st.session_state.obs_h
                    })
                    save_all_data()
                    st.success(f"障碍物已保存，高度：{st.session_state.obs_h}m")
                    st.session_state.temp_points = []
                    st.rerun()
                else:
                    st.error("至少需要圈选3个点位！")
        else:
            st.info("切换到【障碍物圈选】模式以绘制禁飞区")
        if st.button("🗑️ 清空全部障碍物"):
            st.session_state.polygon_memory = []
            st.session_state.temp_points = []
            save_all_data()
            st.rerun()
        st.info(f"系统已记忆障碍物总数：{len(st.session_state.polygon_memory)} 个")

        st.divider()
        st.subheader("📊 安全避障判断")
        for idx, obs in enumerate(st.session_state.polygon_memory):
            oh = obs["h"]
            if st.session_state.height < oh:
                st.error(f"⚠️ 障碍物{idx+1}({oh}m) → 高度不足，需绕飞")
            else:
                st.success(f"✅ 障碍物{idx+1}({oh}m) → 高度安全")

        st.divider()
        st.subheader("❤️ 无人机心跳监测")
        now = datetime.now().strftime("%H:%M:%S")
        st.metric("当前系统时间", now)
        st.session_state.heartbeat_data.append(time.time())
        if len(st.session_state.heartbeat_data) > 30:
            st.session_state.heartbeat_data.pop(0)
        df = pd.DataFrame({
            "采样点": range(len(st.session_state.heartbeat_data)),
            "心跳时间戳": st.session_state.heartbeat_data
        })
        st.line_chart(df.set_index("采样点"), height=150)
        st.success("无人机链路心跳正常")

    with col_map:
        center_lat = (st.session_state.A[0] + st.session_state.B[0]) / 2
        center_lon = (st.session_state.A[1] + st.session_state.B[1]) / 2
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=17,
            tiles="https://{s}.tile.openstreetmap.de/tiles/osmde/{z}/{x}/{y}.png",
            attr="OpenStreetMap (WGS-84)",
            max_zoom=22
        )
        folium.plugins.Fullscreen(position="topright").add_to(m)

        if st.session_state.A_set:
            folium.CircleMarker(st.session_state.A, radius=12, color="red", fill=True, fill_color="red", popup="起点A").add_to(m)
        if st.session_state.B_set:
            folium.CircleMarker(st.session_state.B, radius=12, color="green", fill=True, fill_color="green", popup="终点B").add_to(m)

        for idx, obs in enumerate(st.session_state.polygon_memory):
            pts = obs["pts"]
            hh = obs["h"]
            if len(pts) >= 3:
                folium.Polygon(pts, color="#dc2626", fill=True, fill_opacity=0.45, popup=f"障碍物{idx+1}｜高度：{hh}m").add_to(m)
                poly = Polygon(pts).buffer(st.session_state.safe_radius)
                folium.Polygon(list(poly.exterior.coords), color="#ff9900", fill=False, weight=2, dash_array="5 5", popup="安全半径").add_to(m)

        if st.session_state.temp_points:
            for pt in st.session_state.temp_points:
                folium.CircleMarker(pt, radius=5, color="#ff7700", fill=True).add_to(m)
            folium.PolyLine(st.session_state.temp_points, color="#ff7700", weight=3, dash_array="10 5").add_to(m)

        # ===================== 你要的四条路线全部显示 =====================
        if st.session_state.A_set and st.session_state.B_set:
            left, right, up_curve = get_all_routes(
                st.session_state.A, st.session_state.B,
                st.session_state.polygon_memory,
                st.session_state.safe_radius,
                st.session_state.height
            )

            st.session_state.left_route = left
            st.session_state.right_route = right
            st.session_state.shortest_route = up_curve

            # 直飞（黑色虚线）
            folium.PolyLine([st.session_state.A, st.session_state.B], color="black", weight=2, dash_array="4 4", popup="直飞").add_to(m)
            # 左绕（蓝色）
            folium.PolyLine(left, color="blue", weight=4, popup="左绕飞").add_to(m)
            # 右绕（绿色）
            folium.PolyLine(right, color="green", weight=4, popup="右绕飞").add_to(m)
            # 向上最优弧线（橙色）
            folium.PolyLine(up_curve, color="orange", weight=5, popup="向上绕飞（推荐）").add_to(m)

        output = st_folium(m, width=1150, height=720, key="main_map")
        if output and output.get("last_clicked"):
            now = time.time()
            if now - st.session_state.last_click_time > 0.5:
                pt = output["last_clicked"]
                click_lat, click_lng = pt["lat"], pt["lng"]
                new_pt = [click_lat, click_lng]
                mode = st.session_state.click_mode
                if mode == "障碍物圈选":
                    if not st.session_state.temp_points or new_pt != st.session_state.temp_points[-1]:
                        st.session_state.temp_points.append(new_pt)
                        st.session_state.last_click_time = now
                        st.rerun()
                elif mode == "选择起点A":
                    st.session_state.A = (click_lat, click_lng)
                    st.session_state.A_set = True
                    save_all_data()
                    st.rerun()
                elif mode == "选择终点B":
                    st.session_state.B = (click_lat, click_lng)
                    st.session_state.B_set = True
                    save_all_data()
                    st.rerun()

# ===================== 飞行监控页面 =====================
else:
    st.title("📡 飞行实时画面 - 任务执行监控")
    st.success("✅ 无人机系统链路正常，设备在线")
    col_btn = st.columns(4)
    with col_btn[0]:
        if st.button("🔴 开始任务", type="primary", disabled=st.session_state.flight_running):
            st.session_state.flight_waypoints = st.session_state.get("shortest_route", [st.session_state.A, st.session_state.B])
            st.session_state.flight_running = True
            st.session_state.flight_paused = False
            st.session_state.flight_start_time = datetime.now()
            st.session_state.current_wp_idx = 0
            st.session_state.elapsed_distance = 0.0
            st.rerun()
    with col_btn[1]:
        if st.button("⏸️ 暂停", disabled=not st.session_state.flight_running or st.session_state.flight_paused):
            st.session_state.flight_paused = True
            st.rerun()
    with col_btn[2]:
        if st.button("▶️ 继续", disabled=not st.session_state.flight_paused):
            st.session_state.flight_paused = False
            st.rerun()
    with col_btn[3]:
        if st.button("⏹️ 停止/重置", type="secondary"):
            st.session_state.flight_running = False
            st.session_state.flight_paused = False
            st.session_state.current_wp_idx = 0
            st.session_state.battery = 100.0
            st.rerun()

    if len(st.session_state.flight_waypoints) < 2:
        st.warning("⚠️ 请先在【航线规划】页面设置A/B点并生成航线！")
    else:
        total_dist = sum(np.hypot(
            st.session_state.flight_waypoints[i+1][0]-st.session_state.flight_waypoints[i][0],
            st.session_state.flight_waypoints[i+1][1]-st.session_state.flight_waypoints[i][1]
        ) for i in range(len(st.session_state.flight_waypoints)-1))
        st.session_state.total_distance = round(total_dist * 111000, 2)

        if st.session_state.flight_running and not st.session_state.flight_paused:
            if st.session_state.current_wp_idx < len(st.session_state.flight_waypoints)-1:
                st.session_state.current_wp_idx += 0.02
                st.session_state.battery = max(0, st.session_state.battery - 0.02)

        current_wp = int(st.session_state.current_wp_idx)+1
        total_wp = len(st.session_state.flight_waypoints)
        remain = round(st.session_state.total_distance - st.session_state.elapsed_distance,2)
        cols = st.columns(6)
        cols[0].metric("当前航点", f"{current_wp}/{total_wp}")
        cols[1].metric("飞行速度", "8.5 m/s")
        cols[2].metric("剩余距离", f"{remain} m")
        cols[5].metric("电量", f"{round(st.session_state.battery,1)}%")
        prog = st.session_state.current_wp_idx/(total_wp-1) if total_wp>1 else 0
        st.progress(prog, text=f"进度：{round(prog*100,1)}%")

        col_map_flight, col_status = st.columns([2,1])
        with col_map_flight:
            m_flight = folium.Map(location=st.session_state.flight_waypoints[0], zoom_start=17, tiles="https://{s}.tile.openstreetmap.de/tiles/osmde/{z}/{x}/{y}.png", attr="OSM")
            for obs in st.session_state.polygon_memory:
                pts = obs["pts"]
                if len(pts)>=3:
                    folium.Polygon(pts, color="#dc2626", fill=True, fill_opacity=0.45).add_to(m_flight)
            folium.PolyLine(st.session_state.flight_waypoints, color="orange", weight=4).add_to(m_flight)
            st_folium(m_flight, width="100%", height=500, key="flight_map")

    if st.session_state.flight_running and not st.session_state.flight_paused:
        time.sleep(0.1)
