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
# 火星坐标系(GCJ-02)与WGS84互转算法
x_pi = 3.14159265358979324 * 3000.0 / 180.0
pi = 3.1415926535897932384626
a = 6378245.0
ee = 0.00669342162296594323

def gcj02_to_wgs84(lng, lat):
    """GCJ-02转WGS84"""
    dlat = _transformlat(lng - 105.0, lat - 35.0)
    dlng = _transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    mglat = lat - dlat
    mglng = lng - dlng
    return [mglng, mglat]

def wgs84_to_gcj02(lng, lat):
    """WGS84转GCJ-02（火星坐标系，国内地图用）"""
    dlat = _transformlat(lng - 105.0, lat - 35.0)
    dlng = _transformlng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return [mglng, mglat]

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

# -------------------------- 初始化会话状态（新增障碍物记忆） --------------------------
if "drone_data" not in st.session_state:
    st.session_state.drone_data = {
        "lat_a": 32.2322, "lon_a": 118.749,  # 起点A（WGS84）
        "lat_b": 32.2343, "lon_b": 118.749,  # 终点B（WGS84）
        "current_lat": 32.2322, "current_lon": 118.749,
        "sequence": 0,
        "status": "正常",
        "heartbeats": [],
        "last_receive_time": datetime.now(),
        "obstacles": [],  # 存储障碍物多边形坐标（WGS84）
        "map_tile": "satellite"  # 默认卫星地图
    }

# 页面配置
st.set_page_config(page_title="无人机心跳+地图Demo", layout="wide")
st.title("无人机智能化应用 - 心跳监测 + 地图 + 障碍物圈选")

# ---------------------- 左侧：坐标设置 + 飞行参数 + 地图设置 ----------------------
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("📍 坐标设置（WGS84坐标系）")
    # 起点A输入
    st.markdown("**起点A**")
    lat_a = st.number_input("纬度A", value=st.session_state.drone_data["lat_a"], format="%.6f", key="lat_a")
    lon_a = st.number_input("经度A", value=st.session_state.drone_data["lon_a"], format="%.6f", key="lon_a")
    # 终点B输入
    st.markdown("**终点B**")
    lat_b = st.number_input("纬度B", value=st.session_state.drone_data["lat_b"], format="%.6f", key="lat_b")
    lon_b = st.number_input("经度B", value=st.session_state.drone_data["lon_b"], format="%.6f", key="lon_b")
    
    # 坐标转换校验
    if st.button("更新坐标并校验"):
        st.session_state.drone_data["lat_a"] = lat_a
        st.session_state.drone_data["lon_a"] = lon_a
        st.session_state.drone_data["lat_b"] = lat_b
        st.session_state.drone_data["lon_b"] = lon_b
        # 转换为GCJ-02用于地图显示
        gcj_a = wgs84_to_gcj02(lon_a, lat_a)
        gcj_b = wgs84_to_gcj02(lon_b, lat_b)
        st.success(f"坐标已更新！\n起点GCJ-02: {gcj_a[1]:.6f}, {gcj_a[0]:.6f}\n终点GCJ-02: {gcj_b[1]:.6f}, {gcj_b[0]:.6f}")

    st.subheader("🗺️ 地图设置")
    # 地图类型切换（卫星/普通）
    tile_option = st.radio("地图类型", ["卫星地图", "普通地图"], 
                          index=0 if st.session_state.drone_data["map_tile"] == "satellite" else 1)
    st.session_state.drone_data["map_tile"] = "satellite" if tile_option == "卫星地图" else "normal"

    st.subheader("✈️ 飞行参数")
    height = st.slider("设定飞行高度(m)", 10, 150, 50, key="height")
    auto_send = st.checkbox("自动发送心跳（每秒1次）", key="auto_send")

    # 障碍物管理
    st.subheader("🚧 障碍物管理")
    if st.session_state.drone_data["obstacles"]:
        st.info(f"已保存 {len(st.session_state.drone_data['obstacles'])} 个障碍物区域")
        if st.button("清空所有障碍物"):
            st.session_state.drone_data["obstacles"] = []
            st.success("障碍物已清空！")
            st.rerun()
    else:
        st.info("暂无障碍物，可在地图上用多边形圈选")

# ---------------------- 右侧：地图显示（卫星+圈选+坐标转换） ----------------------
with col_right:
    st.subheader("🗺️ 校园地图（GCJ-02坐标系）")
    # 地图中心取A/B中点（WGS84转GCJ-02）
    center_wgs_lat = (lat_a + lat_b) / 2
    center_wgs_lon = (lon_a + lon_b) / 2
    center_gcj = wgs84_to_gcj02(center_wgs_lon, center_wgs_lat)
    map_center = [center_gcj[1], center_gcj[0]]

    # 选择地图瓦片
    if st.session_state.drone_data["map_tile"] == "satellite":
        # 高德卫星地图（国内加载稳定，符合OpenStreetMap兼容要求）
        tile_url = "https://webst01.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}"
        tile_attr = "高德卫星地图"
    else:
        # 高德普通地图
        tile_url = "https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=8"
        tile_attr = "高德地图"

    # 初始化地图
    m = folium.Map(
        location=map_center,
        zoom_start=18,
        tiles=tile_url,
        attr=tile_attr
    )

    # 标记起点A（WGS84转GCJ-02）
    gcj_a = wgs84_to_gcj02(lon_a, lat_a)
    folium.Marker([gcj_a[1], gcj_a[0]], popup="起点A", icon=folium.Icon(color="red")).add_to(m)
    # 标记终点B
    gcj_b = wgs84_to_gcj02(lon_b, lat_b)
    folium.Marker([gcj_b[1], gcj_b[0]], popup="终点B", icon=folium.Icon(color="green")).add_to(m)
    # 标记无人机当前位置
    current_gcj = wgs84_to_gcj02(st.session_state.drone_data["current_lon"], st.session_state.drone_data["current_lat"])
    folium.Marker(
        [current_gcj[1], current_gcj[0]],
        popup=f"无人机\n序号: {st.session_state.drone_data['sequence']}\n状态: {st.session_state.drone_data['status']}",
        icon=folium.Icon(color="blue" if st.session_state.drone_data["status"] == "正常" else "red")
    ).add_to(m)
    # 绘制AB连线（模拟航线）
    folium.PolyLine(
        locations=[[gcj_a[1], gcj_a[0]], [gcj_b[1], gcj_b[0]]],
        color="blue", weight=3, opacity=0.7
    ).add_to(m)

    # 加载已保存的障碍物（WGS84转GCJ-02）
    for obs in st.session_state.drone_data["obstacles"]:
        # 转换每个顶点坐标
        gcj_coords = []
        for (lon, lat) in obs["coords"]:
            gcj = wgs84_to_gcj02(lon, lat)
            gcj_coords.append([gcj[1], gcj[0]])
        # 绘制多边形
        folium.Polygon(
            locations=gcj_coords,
            color="red",
            weight=2,
            fill_color="red",
            fill_opacity=0.3,
            popup=f"障碍物 {obs['name']}"
        ).add_to(m)

    # 多边形圈选工具（Draw插件）
    draw = Draw(
        draw_options={
            "polyline": False,
            "polygon": True,  # 仅保留多边形圈选
            "circle": False,
            "rectangle": False,
            "marker": False,
            "circlemarker": False
        },
        edit_options={"edit": True, "remove": True}
    )
    draw.add_to(m)

    # 渲染地图并获取圈选数据
    map_html = m._repr_html_()
    # 注入JS获取圈选的多边形坐标
    map_html += """
    <script>
    // 监听Draw事件，获取多边形坐标
    document.addEventListener('DOMContentLoaded', function() {
        const map = window[Object.keys(window).find(key => key.startsWith('map_'))];
        map.on('draw:created', function(e) {
            const layer = e.layer;
            const coords = layer.toGeoJSON().geometry.coordinates[0];
            // 发送坐标到Streamlit
            window.parent.postMessage({
                type: 'obstacle_coords',
                coords: coords
            }, '*');
        });
    });
    </script>
    """
    # 渲染地图
    html(map_html, height=600)

    # 处理圈选的障碍物（通过Streamlit回调）
    # 模拟获取圈选数据（实际部署可通过streamlit-folium优化，此处用手动保存兼容）
    with st.expander("📌 保存圈选的障碍物"):
        obstacle_name = st.text_input("障碍物名称", value=f"障碍物{len(st.session_state.drone_data['obstacles'])+1}")
        if st.button("保存当前圈选"):
            # 模拟获取圈选坐标（实际可通过JS回调获取，此处用示例坐标演示，真实使用替换为实际坐标）
            # 注意：实际部署建议用streamlit-folium组件获取坐标，此处为兼容IDLE运行
            sample_coords = [
                [lon_a+0.0001, lat_a+0.0001],
                [lon_a+0.0002, lat_a+0.0001],
                [lon_a+0.0002, lat_a+0.0002],
                [lon_a+0.0001, lat_a+0.0002]
            ]
            st.session_state.drone_data["obstacles"].append({
                "name": obstacle_name,
                "coords": sample_coords,
                "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            st.success(f"障碍物「{obstacle_name}」已保存！")
            st.rerun()

# ---------------------- 下方：心跳监测 + 掉线检测 + 可视化 ----------------------
st.divider()
col_heartbeat, col_chart = st.columns([1, 1])

with col_heartbeat:
    st.subheader("❤️ 心跳监测")
    # 手动发送心跳
    if st.button("发送心跳包", key="send_heart"):
        st.session_state.drone_data["sequence"] += 1
        st.session_state.drone_data["last_receive_time"] = datetime.now()
        # 模拟向终点移动（WGS84坐标）
        step_lat = (lat_b - lat_a) / 20
        step_lon = (lon_b - lon_a) / 20
        st.session_state.drone_data["current_lat"] = min(st.session_state.drone_data["current_lat"] + step_lat, lat_b)
        st.session_state.drone_data["current_lon"] = min(st.session_state.drone_data["current_lon"] + step_lon, lon_b)
        # 模拟状态
        st.session_state.drone_data["status"] = "正常"
        # 记录日志
        st.session_state.drone_data["heartbeats"].append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "seq": st.session_state.drone_data["sequence"],
            "status": st.session_state.drone_data["status"],
            "height": height
        })
        st.success(f"心跳 {st.session_state.drone_data['sequence']} 发送成功！")
        st.rerun()

    # 掉线检测（3秒未收到心跳则报警）
    time_diff = datetime.now() - st.session_state.drone_data["last_receive_time"]
    if time_diff > timedelta(seconds=3):
        st.session_state.drone_data["status"] = "超时"
        st.error(f"⚠️ 连接超时！{time_diff.seconds}秒未收到心跳包！")
    else:
        st.info(f"✅ 连接正常（上次心跳: {time_diff.microseconds//1000}ms前）")

    st.subheader("📝 心跳日志（最近10条）")
    if st.session_state.drone_data["heartbeats"]:
        for hb in reversed(st.session_state.drone_data["heartbeats"][-10:]):
            st.write(f"[{hb['time']}] 序号:{hb['seq']} | 状态:{hb['status']} | 高度:{hb['height']}m")
    else:
        st.info("暂无心跳数据，点击「发送心跳包」生成数据")

with col_chart:
    st.subheader("📊 数据可视化")
    if st.session_state.drone_data["heartbeats"]:
        df = pd.DataFrame(st.session_state.drone_data["heartbeats"])
        df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S")
        # 折线图：心跳序号随时间变化
        st.line_chart(df, x="time", y="seq", color="#00aaff", height=200, use_container_width=True)
        # 柱状图：状态统计
        status_df = df["status"].value_counts().reset_index()
        status_df.columns = ["状态", "数量"]
        color_map = {"正常": "#22c55e", "超时": "#ef4444"}
        status_df["color"] = status_df["状态"].map(color_map)
        st.bar_chart(status_df, x="状态", y="数量", color="color", height=200, use_container_width=True)
    else:
        st.info("暂无心跳数据，点击「发送心跳包」生成数据")

# 自动发送心跳逻辑
if auto_send:
    time.sleep(1)
    st.session_state.drone_data["sequence"] += 1
    st.session_state.drone_data["last_receive_time"] = datetime.now()
    # 模拟向终点移动
    step_lat = (lat_b - lat_a) / 20
    step_lon = (lon_b - lon_a) / 20
    st.session_state.drone_data["current_lat"] = min(st.session_state.drone_data["current_lat"] + step_lat, lat_b)
    st.session_state.drone_data["current_lon"] = min(st.session_state.drone_data["current_lon"] + step_lon, lon_b)
    st.session_state.drone_data["status"] = "正常"
    st.session_state.drone_data["heartbeats"].append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "seq": st.session_state.drone_data["sequence"],
        "status": st.session_state.drone_data["status"],
        "height": height
    })
    st.rerun()
