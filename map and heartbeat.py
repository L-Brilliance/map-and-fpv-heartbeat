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

# ===================== 安全绕飞算法（左右弧线） =====================
def get_left_right_routes(pA, pB, obstacles, safe_dist):
    """
    返回 left_route, right_route, shortest_route
    若无障碍物冲突，返回直线作为三条路径
    """
    base_line = LineString([pA, pB])
    # 构建所有障碍物的安全缓冲区
    buffers = []
    for obs in obstacles:
        pts = obs["pts"]
        if len(pts) >= 3:
            poly = Polygon(pts).buffer(safe_dist)
            buffers.append(poly)
    if not buffers:
        straight = [pA, pB]
        return straight, straight, straight

    merged = unary_union(buffers)

    # 无冲突，返回直线
    if not base_line.intersects(merged):
        straight = [pA, pB]
        return straight, straight, straight

    # 获取合并缓冲区的外边界
    boundary = merged.exterior
    coords = list(boundary.coords)

    # 直线与边界的交点
    intersection = boundary.intersection(base_line)
    pts = []
    if isinstance(intersection, Point):
        pts = [intersection]
    elif isinstance(intersection, MultiPoint):
        pts = list(intersection.geoms)
    if len(pts) < 2:
        # 尝试延伸直线
        dx = pB[0] - pA[0]
        dy = pB[1] - pA[1]
        ext_line = LineString([(pA[0]-dx*0.01, pA[1]-dy*0.01),
                               (pB[0]+dx*0.01, pB[1]+dy*0.01)])
        intersection = boundary.intersection(ext_line)
        if isinstance(intersection, Point):
            pts = [intersection]
        elif isinstance(intersection, MultiPoint):
            pts = list(intersection.geoms)
        if len(pts) < 2:
            # 回退到直线
            straight = [pA, pB]
            return straight, straight, straight

    # 按直线方向排序交点
    pts.sort(key=lambda p: base_line.project(p))
    entry = pts[0]
    exit_ = pts[-1]

    # 找到交点在边界坐标中的索引
    def nearest_idx(point, coords):
        best, idx = float('inf'), 0
        for i, (x, y) in enumerate(coords):
            d = np.hypot(x - point.x, y - point.y)
            if d < best:
                best, idx = d, i
        return idx

    i_entry = nearest_idx(entry, coords)
    i_exit = nearest_idx(exit_, coords)

    # 边界拆分为两段
    if i_entry <= i_exit:
        seg1 = coords[i_entry:i_exit+1]
        seg2 = coords[i_exit:] + coords[:i_entry+1]
    else:
        seg1 = coords[i_entry:] + coords[:i_exit+1]
        seg2 = coords[i_exit:i_entry+1]

    # 判断左右：取段中点的侧向叉积
    dir_vec = (pB[0]-pA[0], pB[1]-pA[1])
    def cross(pt):
        vx, vy = pt[0]-pA[0], pt[1]-pA[1]
        return dir_vec[0]*vy - dir_vec[1]*vx

    mid1 = seg1[len(seg1)//2]
    mid2 = seg2[len(seg2)//2]
    if cross(mid1) > cross(mid2):
        left_seg = seg1
        right_seg = seg2
    else:
        left_seg = seg2
        right_seg = seg1

    # 构建完整路径（用边界点，不简化）
    entry_pt = (entry.x, entry.y)
    exit_pt = (exit_.x, exit_.y)

    left_path = [pA, entry_pt] + left_seg[1:-1] + [exit_pt, pB]
    right_path = [pA, entry_pt] + right_seg[1:-1] + [exit_pt, pB]

    # 计算路径长度（经纬度近似距离）
    def path_len(pts):
        total = 0.0
        for i in range(len(pts)-1):
            total += np.hypot(pts[i+1][0]-pts[i][0], pts[i+1][1]-pts[i][1])
        return total

    len_left = path_len(left_path)
    len_right = path_len(right_path)
    shortest_path = left_path if len_left < len_right else right_path

    return left_path, right_path, shortest_path

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
        if st.session_state.A_set and st.session_state.B_set:
            left_route, right_route, shortest_route = get_left_right_routes(
                st.session_state.A, st.session_state.B,
                st.session_state.polygon_memory,
                st.session_state.safe_radius
            )
            # 判断是否有绕飞
            direct = LineString([st.session_state.A, st.session_state.B])
            conflict = any(direct.intersects(Polygon(obs["pts"]).buffer(st.session_state.safe_radius))
                           for obs in st.session_state.polygon_memory if len(obs["pts"])>=3)
            if conflict:
                st.warning("🛡️ 航线与障碍物距离不足，已生成左右绕飞弧线")
            else:
                st.success("🛡️ 航线与障碍物安全距离充足")

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

        # 起点/终点
        if st.session_state.A_set:
            folium.CircleMarker(st.session_state.A, radius=12, color="red",
                                fill=True, fill_color="red", popup="起点A").add_to(m)
        if st.session_state.B_set:
            folium.CircleMarker(st.session_state.B, radius=12, color="green",
                                fill=True, fill_color="green", popup="终点B").add_to(m)

        # 障碍物及缓冲区
        for idx, obs in enumerate(st.session_state.polygon_memory):
            pts = obs["pts"]
            hh = obs["h"]
            if len(pts) >= 3:
                folium.Polygon(pts, color="#dc2626", fill=True, fill_opacity=0.45,
                               popup=f"障碍物{idx+1}｜高度：{hh}m").add_to(m)
                poly = Polygon(pts).buffer(st.session_state.safe_radius)
                folium.Polygon(list(poly.exterior.coords), color="#ff9900", fill=False,
                               weight=2, dash_array="5 5", popup="安全半径缓冲区").add_to(m)

        # 临时绘制点
        if st.session_state.temp_points:
            for pt in st.session_state.temp_points:
                folium.CircleMarker(pt, radius=5, color="#ff7700", fill=True, fill_color="#ff7700").add_to(m)
            folium.PolyLine(st.session_state.temp_points, color="#ff7700", weight=3, dash_array="10 5").add_to(m)

        # 航线显示
        if st.session_state.A_set and st.session_state.B_set:
            left_route, right_route, shortest_route = get_left_right_routes(
                st.session_state.A, st.session_state.B,
                st.session_state.polygon_memory,
                st.session_state.safe_radius
            )
            # 保存所有可能路线以便监控选择
            st.session_state.left_route = left_route
            st.session_state.right_route = right_route
            st.session_state.shortest_route = shortest_route

            # 直飞虚线
            folium.PolyLine([st.session_state.A, st.session_state.B], color="gray",
                            weight=2, dash_array="5,5", popup="直飞航线（不可用）").add_to(m)

            # 左绕弧线（蓝色）
            folium.PolyLine(left_route, color="blue", weight=4, popup="左绕弧线").add_to(m)
            # 右绕弧线（红色）
            folium.PolyLine(right_route, color="red", weight=4, popup="右绕弧线").add_to(m)
            # 最短路径（橙色）
            folium.PolyLine(shortest_route, color="orange", weight=5, popup="最短弧线（推荐）").add_to(m)

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
                    st.session_state.last_click_time = now
                    save_all_data()
                    st.rerun()
                elif mode == "选择终点B":
                    st.session_state.B = (click_lat, click_lng)
                    st.session_state.B_set = True
                    st.session_state.last_click_time = now
                    save_all_data()
                    st.rerun()

# ===================== 飞行监控页面 =====================
else:
    st.title("📡 飞行实时画面 - 任务执行监控")
    st.success("✅ 无人机系统链路正常，设备在线")
    st.subheader("监测区域：南京科技职业学院校内空域")

    col_btn = st.columns(4)
    with col_btn[0]:
        if st.button("🔴 开始任务", type="primary", disabled=st.session_state.flight_running):
            # 选择将使用的航线：默认最短弧线
            if hasattr(st.session_state, 'selected_route'):
                route = st.session_state.selected_route
            else:
                route = "shortest"
            if route == "left":
                st.session_state.flight_waypoints = st.session_state.get("left_route", [st.session_state.A, st.session_state.B])
            elif route == "right":
                st.session_state.flight_waypoints = st.session_state.get("right_route", [st.session_state.A, st.session_state.B])
            else:
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
            st.session_state.elapsed_distance = 0.0
            st.rerun()

    status_text = "任务运行中" if st.session_state.flight_running and not st.session_state.flight_paused else (
        "已暂停" if st.session_state.flight_paused else "已停止")
    st.caption(f"状态：{status_text}")

    if len(st.session_state.flight_waypoints) < 2:
        st.warning("⚠️ 请先在【航线规划】页面设置A/B点并生成航线！")
    else:
        total_dist = sum(
            np.hypot(st.session_state.flight_waypoints[i+1][0] - st.session_state.flight_waypoints[i][0],
                     st.session_state.flight_waypoints[i+1][1] - st.session_state.flight_waypoints[i][1])
            for i in range(len(st.session_state.flight_waypoints)-1)
        )
        st.session_state.total_distance = round(total_dist * 111000, 2)

        if st.session_state.flight_running and not st.session_state.flight_paused:
            if st.session_state.current_wp_idx < len(st.session_state.flight_waypoints)-1:
                st.session_state.current_wp_idx += 0.02
                st.session_state.battery = max(0, st.session_state.battery - 0.02)
                st.session_state.elapsed_distance = round(
                    st.session_state.current_wp_idx / (len(st.session_state.flight_waypoints)-1) * st.session_state.total_distance, 2
                )
            else:
                st.session_state.flight_running = False
                st.success("🎉 任务完成！已到达终点")

        current_wp = int(st.session_state.current_wp_idx) + 1
        total_wp = len(st.session_state.flight_waypoints)
        elapsed_dist = st.session_state.elapsed_distance
        remain_dist = round(st.session_state.total_distance - elapsed_dist, 2)

        if st.session_state.flight_start_time:
            elapsed_time = datetime.now() - st.session_state.flight_start_time
            elapsed_time_str = str(timedelta(seconds=int(elapsed_time.total_seconds()))).split(".")[0]
        else:
            elapsed_time_str = "00:00"

        eta_str = str(timedelta(seconds=int(remain_dist / st.session_state.flight_speed))).split(".")[0] if remain_dist > 0 and st.session_state.flight_speed > 0 else "00:00"
        battery = round(st.session_state.battery, 1)

        cols = st.columns(6)
        cols[0].metric("当前航点", f"{current_wp}/{total_wp}")
        cols[1].metric("飞行速度", f"{st.session_state.flight_speed} m/s")
        cols[2].metric("已用时间", elapsed_time_str)
        cols[3].metric("剩余距离", f"{remain_dist} m")
        cols[4].metric("预计到达", eta_str)
        cols[5].metric("电量模拟", f"{battery} %")

        progress = min(st.session_state.current_wp_idx / (total_wp - 1), 1.0) if total_wp > 1 else 0
        st.progress(progress, text=f"任务进度：{round(progress*100, 1)}%")

        col_map_flight, col_status = st.columns([2, 1])
        with col_map_flight:
            # 修复 attribution 缺失
            m_flight = folium.Map(
                location=st.session_state.flight_waypoints[0],
                zoom_start=17,
                tiles="https://{s}.tile.openstreetmap.de/tiles/osmde/{z}/{x}/{y}.png",
                attr="OpenStreetMap (WGS-84)"
            )
            for obs in st.session_state.polygon_memory:
                pts = obs["pts"]
                if len(pts) >= 3:
                    folium.Polygon(pts, color="#dc2626", fill=True, fill_opacity=0.45).add_to(m_flight)
            # 绘制整条航线（浅蓝）
            folium.PolyLine(st.session_state.flight_waypoints, color="#0066ff", weight=3, opacity=0.5).add_to(m_flight)
            flown_idx = int(st.session_state.current_wp_idx)
            if flown_idx >= 1:
                folium.PolyLine(st.session_state.flight_waypoints[:flown_idx+1], color="#22bb22", weight=4).add_to(m_flight)
            if len(st.session_state.flight_waypoints) > 0:
                drone_pos = st.session_state.flight_waypoints[min(flown_idx, len(st.session_state.flight_waypoints)-1)]
                folium.CircleMarker(drone_pos, radius=10, color="orange", fill=True, fill_color="orange").add_to(m_flight)
            st_folium(m_flight, width="100%", height=500, key="flight_map")
        with col_status:
            st.subheader("📡 通信链路拓扑与数据流")
            st.success("✅ GCS 在线")
            st.success("✅ OBC 在线")
            st.success("✅ FCU 在线")
            st.metric("飞行高度", f"{st.session_state.height} m")
            st.metric("安全半径", f"{st.session_state.safe_radius}")

    if st.session_state.flight_running and not st.session_state.flight_paused:
        time.sleep(0.1)
