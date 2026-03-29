import streamlit as st
import folium
from datetime import datetime, timedelta
import pandas as pd
from streamlit.components.v1 import html
import time

# 初始化会话状态
if "drone_data" not in st.session_state:
    st.session_state.drone_data = {
        "lat_a": 32.2322, "lon_a": 118.749,  # 起点A（校园内）
        "lat_b": 32.2343, "lon_b": 118.749,  # 终点B（校园内）
        "current_lat": 32.2322, "current_lon": 118.749,
        "sequence": 0,
        "status": "正常",
        "heartbeats": [],
        "last_receive_time": datetime.now()  # 最后一次接收心跳时间
    }

# 页面配置
st.set_page_config(page_title="无人机心跳+地图Demo", layout="wide")
st.title("无人机智能化应用 - 心跳监测 + 地图")

# ---------------------- 左侧：坐标设置 + 飞行参数 ----------------------
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("📍 坐标设置")
    # 起点A输入
    st.markdown("**起点A**")
    lat_a = st.number_input("纬度A", value=st.session_state.drone_data["lat_a"], format="%.4f")
    lon_a = st.number_input("经度A", value=st.session_state.drone_data["lon_a"], format="%.4f")
    # 终点B输入
    st.markdown("**终点B**")
    lat_b = st.number_input("纬度B", value=st.session_state.drone_data["lat_b"], format="%.4f")
    lon_b = st.number_input("经度B", value=st.session_state.drone_data["lon_b"], format="%.4f")
    # 更新坐标
    if st.button("更新坐标"):
        st.session_state.drone_data["lat_a"] = lat_a
        st.session_state.drone_data["lon_a"] = lon_a
        st.session_state.drone_data["lat_b"] = lat_b
        st.session_state.drone_data["lon_b"] = lon_b
        st.success("坐标已更新！")

    st.subheader("✈️ 飞行参数")
    height = st.slider("设定飞行高度(m)", 10, 150, 50)
    auto_send = st.checkbox("自动发送心跳（每秒1次）")

# ---------------------- 右侧：地图显示 ----------------------
with col_right:
    st.subheader("🗺️ 校园地图")
    # 地图中心取A/B中点
    map_center = [(lat_a + lat_b)/2, (lon_a + lon_b)/2]
    # 使用高德地图瓦片（国内加载稳定）
    m = folium.Map(
        location=map_center,
        zoom_start=18,
        tiles="https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=8",
        attr="高德地图"
    )
    # 标记起点A
    folium.Marker([lat_a, lon_a], popup="起点A", icon=folium.Icon(color="red")).add_to(m)
    # 标记终点B
    folium.Marker([lat_b, lon_b], popup="终点B", icon=folium.Icon(color="green")).add_to(m)
    # 标记无人机当前位置
    folium.Marker(
        [st.session_state.drone_data["current_lat"], st.session_state.drone_data["current_lon"]],
        popup=f"无人机\n序号: {st.session_state.drone_data['sequence']}\n状态: {st.session_state.drone_data['status']}",
        icon=folium.Icon(color="blue" if st.session_state.drone_data["status"] == "正常" else "red")
    ).add_to(m)
    # 绘制AB连线（模拟航线）
    folium.PolyLine(
        locations=[[lat_a, lon_a], [lat_b, lon_b]],
        color="blue", weight=3, opacity=0.7
    ).add_to(m)
    # 渲染地图
    html(m._repr_html_(), height=500)

# ---------------------- 下方：心跳监测 + 掉线检测 + 可视化 ----------------------
st.divider()
col_heartbeat, col_chart = st.columns([1, 1])

with col_heartbeat:
    st.subheader("❤️ 心跳监测")
    # 手动发送心跳
    if st.button("发送心跳包"):
        st.session_state.drone_data["sequence"] += 1
        st.session_state.drone_data["last_receive_time"] = datetime.now()
        # 模拟向终点移动
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
            "status": st.session_state.drone_data["status"]
        })
        st.success(f"心跳 {st.session_state.drone_data['sequence']} 发送成功！")

    # 掉线检测（3秒未收到心跳则报警）
    if (datetime.now() - st.session_state.drone_data["last_receive_time"]) > timedelta(seconds=3):
        st.session_state.drone_data["status"] = "超时"
        st.error("⚠️ 连接超时！3秒未收到心跳包！")
    else:
        st.info("✅ 连接正常")

    st.subheader("📝 心跳日志")
    for hb in reversed(st.session_state.drone_data["heartbeats"][-10:]):
        st.write(f"[{hb['time']}] 序号:{hb['seq']} | {hb['status']}")

with col_chart:
    st.subheader("📊 数据可视化")
    if st.session_state.drone_data["heartbeats"]:
        df = pd.DataFrame(st.session_state.drone_data["heartbeats"])
        df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S")
        # 折线图：心跳序号随时间变化
        st.line_chart(df, x="time", y="seq", color="#00aaff", height=250)
        # 柱状图：状态统计
        # 转为 DataFrame 并指定颜色
status_df = df["status"].value_counts().reset_index()
status_df.columns = ["状态", "数量"]
color_map = {"正常": "#22c55e", "超时": "#ef4444"}
status_df["color"] = status_df["状态"].map(color_map)

st.bar_chart(status_df, x="状态", y="数量", color="color")
     else:
        st.info("暂无心跳数据，点击「发送心跳包」生成数据")

# 自动发送心跳逻辑
if auto_send:
    time.sleep(1)
    st.session_state.drone_data["sequence"] += 1
    st.session_state.drone_data["last_receive_time"] = datetime.now()
    step_lat = (lat_b - lat_a) / 20
    step_lon = (lon_b - lon_a) / 20
    st.session_state.drone_data["current_lat"] = min(st.session_state.drone_data["current_lat"] + step_lat, lat_b)
    st.session_state.drone_data["current_lon"] = min(st.session_state.drone_data["current_lon"] + step_lon, lon_b)
    st.session_state.drone_data["status"] = "正常"
    st.session_state.drone_data["heartbeats"].append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "seq": st.session_state.drone_data["sequence"],
        "status": st.session_state.drone_data["status"]
    })
    st.rerun()
