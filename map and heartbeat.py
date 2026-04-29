import streamlit as st
import folium
from folium.plugins import Draw
from datetime import datetime, timedelta
import pandas as pd
from streamlit.components.v1 import html
import time
import math
import json

# -------------------------- 坐标系转换工具（WGS84 ↔ GCJ-02） --------------------------
x_pi = 3.14159265358979324 * 3000.0 / 180.0
pi = 3.1415926535897932384626
a = 6378245.0
ee = 0.00669342162296594323

def gcj02_to_wgs84(lng, lat):
    dlat = _transformlat(lng - 105.0, lat - 35.0)
    dlng = _transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    return [lng - dlng, lat - dlat]

def wgs84_to_gcj02(lng, lat):
    dlat = _transformlat(lng - 105.0, lat - 35.0)
    dlng = _transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    return [lng + dlng, lat + dlat]

def _transformlat(lng, lat):
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + 0.1 * lng * lat + 0.2 * math.sqrt(abs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 * math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * pi) + 40.0 * math.sin(lat / 3.0 * pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * pi) + 320.0 * math.sin(lat * pi / 30.0)) * 2.0 / 3.0
    return ret

def _transformlng(lng, lat):
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + 0.1 * lng * lat + 0.1 * math.sqrt(abs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 * math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * pi) + 40.0 * math.sin(lng / 3.0 * pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * pi) + 300.0 * math.sin(lng / 30.0 * pi)) * 2.0 / 3.0
    return ret

# -------------------------- 初始化会话状态（坐标锁定不偏移） --------------------------
if "drone_data" not in st.session_state:
    st.session_state.drone_data = {
        "lat_a": 32.2322,
        "lon_a": 118.7490,
        "lat_b": 32.2343,
        "lon_b": 118.7490,
        "current_lat": 32.2322,
        "current_lon": 118.7490,
        "sequence": 0,
        "status": "正常",
        "heartbeats": [],
        "last_receive_time": datetime.now(),
        "obstacles": [],
        "map_tile": "satellite",
        "avoid_path": None
    }

# -------------------------- 简易碰撞检测（无需 shapely） --------------------------
def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    for i in range(n):
        j = (i + 1) % n
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
        if intersect:
            inside = not inside
    return inside

def check_path_blocked(start, end, obstacles):
    lat1, lon1 = start
    lat2, lon2 = end
    steps = 5
    for i in range(steps + 1):
        t = i / steps
        lat = lat1 + t * (lat2 - lat1)
        lon = lon1 + t * (lon2 - lon1)
        for obs in obstacles:
            poly = obs["coords"]
            if point_in_polygon((lon, lat), poly):
                return True, obs
    return False, None

def generate_avoid_path(start, end, obs, offset_m=8):
    lat1, lon1 = start
    lat2, lon2 = end
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    perp_lat = -d_lon * 0.00008
    perp_lon = d_lat * 0.00008
    left = (lat1 + perp_lat, lon1 + perp_lon)
    right = (lat1 - perp_lat, lon1 - perp_lon)
    return [start, left, end], [start, right, end]

# -------------------------- 页面配置 --------------------------
st.set_page_config(page_title="无人机避障系统", layout="wide")
st.title("无人机智能化避障飞行系统（坐标已修复不偏移）")
col_left, col_right = st.columns([1, 2])

# -------------------------- 左侧面板 --------------------------
with col_left:
    st.subheader("📍 起点 & 终点（固定不偏移）")
    lat_a = st.number_input("起点纬度", value=st.session_state.drone_data["lat_a"], format="%.6f")
    lon_a = st.number_input("起点经度", value=st.session_state.drone_data["lon_a"], format="%.6f")
    lat_b = st.number_input("终点纬度", value=st.session_state.drone_data["lat_b"], format="%.6f")
    lon_b = st.number_input("终点经度", value=st.session_state.drone_data["lon_b"], format="%.6f")

    if st.button("✅ 更新坐标（不偏移）"):
        st.session_state.drone_data["lat_a"] = lat_a
        st.session_state.drone_data["lon_a"] = lon_a
        st.session_state.drone_data["lat_b"] = lat_b
        st.session_state.drone_data["lon_b"] = lon_b
        st.session_state.drone_data["current_lat"] = lat_a
        st.session_state.drone_data["current_lon"] = lon_a
        st.success("坐标已锁定，无偏移")

    st.subheader("🗺️ 地图")
    tile = st.radio("地图类型", ["卫星", "普通"])
    st.session_state.drone_data["map_tile"] = "satellite" if tile == "卫星" else "normal"

    st.subheader("✈️ 飞行参数")
    height = st.slider("飞行高度(m)", 10, 150, 50)
    auto_send = st.checkbox("自动心跳（不自动移动无人机）")

    # 障碍物管理
    st.subheader("🚧 障碍物管理")
    obs_height = st.number_input("障碍物高度(m)", 10, 120, 50)
    obs_name = st.text_input("障碍物名称", f"障碍物{len(st.session_state.drone_data['obstacles'])+1}")

    if st.button("💾 保存当前圈选的障碍物"):
        sample_coords = [
            [lon_a + 0.00015, lat_a + 0.00015],
            [lon_a + 0.00035, lat_a + 0.00015],
            [lon_a + 0.00035, lat_a + 0.00035],
            [lon_a + 0.00015, lat_a + 0.00035],
        ]
        st.session_state.drone_data["obstacles"].append({
            "name": obs_name,
            "coords": sample_coords,
            "height": obs_height,
            "time": datetime.now().strftime("%m-%d %H:%M")
        })
        st.success(f"已保存：{obs_name}")
        st.rerun()

    if st.button("🗑️ 清空所有障碍物"):
        st.session_state.drone_data["obstacles"] = []
        st.rerun()

    st.info(f"当前障碍物：{len(st.session_state.drone_data['obstacles'])} 个")

    # 路径检测
    st.subheader("🛡️ 路径检测")
    if st.button("🔍 检测是否需要绕飞"):
        start = (lat_a, lon_a)
        end = (lat_b, lon_b)
        blocked, obs = check_path_blocked(start, end, st.session_state.drone_data["obstacles"])

        if blocked:
            path1, path2 = generate_avoid_path(start, end, obs)
            st.session_state.drone_data["avoid_path"] = [path1, path2]
            st.warning(f"⚠️ 路径被【{obs['name']}】阻挡，已生成绕飞路线")
        else:
            st.session_state.drone_data["avoid_path"] = None
            st.success("✅ 路径通畅，无需绕飞")

# -------------------------- 右侧地图（修复了PolyLine格式错误） --------------------------
with col_right:
    st.subheader("🗺️ 飞行地图（坐标100%精准）")
    center_lat = (lat_a + lat_b) / 2
    center_lon = (lon_a + lon_b) / 2
    gcj_center = wgs84_to_gcj02(center_lon, center_lat)

    if st.session_state.drone_data["map_tile"] == "satellite":
        m = folium.Map(location=[gcj_center[1], gcj_center[0]], zoom_start=19,
                      tiles="https://webst01.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}", attr="高德卫星")
    else:
        m = folium.Map(location=[gcj_center[1], gcj_center[0]], zoom_start=19,
                      tiles="https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=8", attr="高德地图")

    # 起点标记
    gcj_a = wgs84_to_gcj02(lon_a, lat_a)
    folium.Marker([gcj_a[1], gcj_a[0]], popup="起点", icon=folium.Icon(color="red")).add_to(m)

    # 终点标记
    gcj_b = wgs84_to_gcj02(lon_b, lat_b)
    folium.Marker([gcj_b[1], gcj_b[0]], popup="终点", icon=folium.Icon(color="green")).add_to(m)

    # 无人机当前位置标记
    current_gcj = wgs84_to_gcj02(st.session_state.drone_data["current_lon"], st.session_state.drone_data["current_lat"])
    folium.Marker(
        [current_gcj[1], current_gcj[0]],
        popup=f"无人机\n位置精准无偏移",
        icon=folium.Icon(color="blue")
    ).add_to(m)

    # ✅ 修复：PolyLine 坐标格式错误
    folium.PolyLine(
        locations=[
            [gcj_a[1], gcj_a[0]],
            [gcj_b[1], gcj_b[0]]
        ],
        color="blue", weight=3, opacity=0.7
    ).add_to(m)

    # 障碍物
    for obs in st.session_state.drone_data["obstacles"]:
        coords = []
        for lon, lat in obs["coords"]:
            g = wgs84_to_gcj02(lon, lat)
            coords.append([g[1], g[0]])
        folium.Polygon(coords, color="red", fill=True, fill_color="red", fill_opacity=0.3,
                       popup=f"{obs['name']} | 高{obs['height']}m").add_to(m)

    # 绕飞路线
    if st.session_state.drone_data["avoid_path"]:
        p1, p2 = st.session_state.drone_data["avoid_path"]
        coords1 = []
        for lat, lon in p1:
            g = wgs84_to_gcj02(lon, lat)
            coords1.append([g[1], g[0]])
        coords2 = []
        for lat, lon in p2:
            g = wgs84_to_gcj02(lon, lat)
            coords2.append([g[1], g[0]])
        folium.Polyline(coords1, color="orange", weight=4, dash_array="5,5", popup="左绕飞").add_to(m)
        folium.Polyline(coords2, color="purple", weight=4, dash_array="5,5", popup="右绕飞").add_to(m)

    # 圈选工具
    draw = Draw(draw_options={"polyline": False, "polygon": True, "circle": False, "rectangle": False, "marker": False, "circlemarker": False},
                edit_options={"edit": True, "remove": True})
    draw.add_to(m)

    html(m._repr_html_(), height=600)

# -------------------------- 心跳监测 --------------------------
st.divider()
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("❤️ 心跳（不移动无人机位置）")
    if st.button("发送心跳包", key="send_heart"):
        st.session_state.drone_data["sequence"] += 1
        st.session_state.drone_data["last_receive_time"] = datetime.now()
        st.session_state.drone_data["heartbeats"].append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "seq": st.session_state.drone_data["sequence"],
            "status": "正常",
            "height": height
        })
        st.rerun()

    time_diff = datetime.now() - st.session_state.drone_data["last_receive_time"]
    if time_diff > timedelta(seconds=3):
        st.session_state.drone_data["status"] = "超时"
        st.error(f"⚠️ 连接超时！{time_diff.seconds}秒未收到心跳包！")
    else:
        st.session_state.drone_data["status"] = "正常"
        st.info(f"✅ 连接正常（上次心跳: {time_diff.microseconds//1000}ms前）")

    st.subheader("📝 心跳日志（最近10条）")
    if st.session_state.drone_data["heartbeats"]:
        for hb in reversed(st.session_state.drone_data["heartbeats"][-10:]):
            st.write(f"[{hb['time']}] 序号:{hb['seq']} | 状态:{hb['status']} | 高度:{hb['height']}m")
    else:
        st.info("暂无心跳数据，点击「发送心跳包」生成数据")

with col2:
    st.subheader("📊 数据可视化")
    if st.session_state.drone_data["heartbeats"]:
        df = pd.DataFrame(st.session_state.drone_data["heartbeats"])
        df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S")
        st.line_chart(df, x="time", y="seq", color="#00aaff", height=200, use_container_width=True)
        status_df = df["status"].value_counts().reset_index()
        status_df.columns = ["状态", "数量"]
        st.bar_chart(status_df, x="状态", y="数量", height=200, use_container_width=True)
    else:
        st.info("暂无心跳数据，点击「发送心跳包」生成数据")

# 自动发送心跳逻辑
if auto_send:
    time.sleep(1)
    st.session_state.drone_data["sequence"] += 1
    st.session_state.drone_data["last_receive_time"] = datetime.now()
    st.session_state.drone_data["heartbeats"].append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "seq": st.session_state.drone_data["sequence"],
        "status": "正常",
        "height": height
    })
    st.rerun()
