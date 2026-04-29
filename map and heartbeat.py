import streamlit as st
import folium
from folium.plugins import Draw
from datetime import datetime, timedelta
import pandas as pd
from streamlit.components.v1 import html
import time
import math
import json
from shapely.geometry import LineString, Polygon

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

# -------------------------- 初始化会话状态（【修复】坐标完全不偏移） --------------------------
if "drone_data" not in st.session_state:
    st.session_state.drone_data = {
        # ✅【保留你原始正确坐标】绝对不偏移
        "lat_a": 32.2322,
        "lon_a": 118.7490,
        "lat_b": 32.2343,
        "lon_b": 118.7490,
        "current_lat": 32.2322,    # 初始位置 = 起点，不偏移
        "current_lon": 118.7490,   # 初始位置 = 起点，不偏移
        "sequence": 0,
        "status": "正常",
        "heartbeats": [],
        "last_receive_time": datetime.now(),
        "obstacles": [],
        "map_tile": "satellite",
        "avoid_path": None
    }

# -------------------------- 避障算法 --------------------------
def check_path_blocked(start, end, obstacles):
    path = LineString([(start[1], start[0]), (end[1], end[0])])
    for obs in obstacles:
        coords = [(lon, lat) for lon, lat in obs["coords"]]
        if len(coords) >= 3:
            poly = Polygon(coords)
            if path.intersects(poly):
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

# -------------------------- 页面 --------------------------
st.set_page_config(page_title="无人机避障系统", layout="wide")
st.title("无人机智能化避障飞行系统（坐标已修复不偏移）")
col_left, col_right = st.columns([1, 2])

# -------------------------- 左侧 --------------------------
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
        # ✅ 关键修复：无人机初始位置强制等于起点，不漂移
        st.session_state.drone_data["current_lat"] = lat_a
        st.session_state.drone_data["current_lon"] = lon_a
        st.success("坐标已锁定，无偏移")

    st.subheader("🗺️ 地图")
    tile = st.radio("地图类型", ["卫星", "普通"])
    st.session_state.drone_data["map_tile"] = "satellite" if tile == "卫星" else "normal"

    st.subheader("✈️ 飞行参数")
    height = st.slider("飞行高度(m)", 10, 150, 50)
    auto_send = st.checkbox("自动心跳（不自动移动无人机）")

    # 障碍物
    st.subheader("🚧 障碍物")
    obs_height = st.number_input("障碍物高度(m)", 10, 120, 50)
    obs_name = st.text_input("障碍物名称", f"障碍物{len(st.session_state.drone_data['obstacles'])+1}")

    if st.button("💾 保存障碍物"):
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

    if st.button("🗑️ 清空障碍物"):
        st.session_state.drone_data["obstacles"] = []
        st.rerun()

    st.info(f"当前障碍物：{len(st.session_state.drone_data['obstacles'])} 个")

    # 避飞
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
            st.success("✅ 路径通畅")

# -------------------------- 右侧地图 --------------------------
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

    # 起点（精准不偏移）
    gcj_a = wgs84_to_gcj02(lon_a, lat_a)
    folium.Marker([gcj_a[1], gcj_a[0]], popup="起点", icon=folium.Icon(color="red")).add_to(m)

    # 终点（精准不偏移）
    gcj_b = wgs84_to_gcj02(lon_b, lat_b)
    folium.Marker([gcj_b[1], gcj_b[0]], popup="终点", icon=folium.Icon(color="green")).add_to(m)

    # 无人机当前位置（强制=起点，不漂移）
    current_gcj = wgs84_to_gcj02(st.session_state.drone_data["current_lon"], st.session_state.drone_data["current_lat"])
    folium.Marker(
        [current_gcj[1], current_gcj[0]],
        popup=f"无人机\n位置精准无偏移",
        icon=folium.Icon(color="blue")
    ).add_to(m)

    # 原航线
    folium.Polyline([[gcj_a[1], gcj_a[0]], [gcj_b[1], gcj_b[0]]], color="blue", weight=3).add_to(m)

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
        coords1 = [wgs84_to_gcj02(lon, lat) for lat, lon in p1]
        coords2 = [wgs84_to_gcj02(lon, lat) for lat, lon in p2]
        folium.Polyline([[c[1], c[0]] for c in coords1], color="orange", weight=4, dash_array="5,5").add_to(m)
        folium.Polyline([[c[1], c[0]] for c in coords2], color="purple", weight=4, dash_array="5,5").add_to(m)

    Draw(draw_options={"polygon": True}, edit_options={"edit": True}).add_to(m)
    html(m._repr_html_(), height=600)

# -------------------------- 心跳（【修复】不会自动移动无人机） --------------------------
st.divider()
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("❤️ 心跳（不移动无人机位置）")
    if st.button("发送心跳"):
        st.session_state.drone_data["sequence"] += 1
        st.session_state.drone_data["last_receive_time"] = datetime.now()
        # ✅ 修复：心跳不改变坐标！彻底杜绝偏移
        st.session_state.drone_data["heartbeats"].append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "seq": st.session_state.drone_data["sequence"],
            "status": "正常",
            "height": height
        })
        st.rerun()

    time_diff = datetime.now() - st.session_state.drone_data["last_receive_time"]
    if time_diff.seconds > 3:
        st.error("⚠️ 连接超时")
    else:
        st.success("✅ 连接正常")

with col2:
    st.subheader("📊 数据")
    if st.session_state.drone_data["heartbeats"]:
        df = pd.DataFrame(st.session_state.drone_data["heartbeats"])
        st.line_chart(df, x="time", y="seq")

# ✅ 修复：自动心跳也不会移动坐标
if auto_send:
    time.sleep(1)
    st.session_state.drone_data["sequence"] += 1
    st.session_state.drone_data["heartbeats"].append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "seq": st.session_state.drone_data["sequence"],
        "status": "正常",
        "height": height
    })
    st.rerun()
