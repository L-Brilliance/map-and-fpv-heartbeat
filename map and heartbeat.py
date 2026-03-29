import streamlit as st
from streamlit_folium import st_folium
import folium

# 初始化会话状态（保存无人机状态）
if "drone_data" not in st.session_state:
    st.session_state.drone_data = {
        "lat": 32.2322,
        "lon": 118.749,
        "sequence": 0,
        "status": "正常",
        "heartbeats": []
    }

# 页面配置
st.set_page_config(page_title="无人机心跳地图Demo", layout="wide")
st.title("无人机智能化应用 - 心跳监测 + 地图")

# 分栏布局：地图 + 心跳日志
col_map, col_log = st.columns([2, 1])

# ---------------------- 左侧：实时地图 ----------------------
with col_map:
    st.subheader("无人机位置地图")
    map_center = [st.session_state.drone_data["lat"], st.session_state.drone_data["lon"]]
m = folium.Map(location=map_center, zoom_start=18)
folium.Marker(
    location=map_center,
    popup=f"序号: {st.session_state.drone_data['sequence']}\n状态: {st.session_state.drone_data['status']}",
    icon=folium.Icon(color="green" if st.session_state.drone_data["status"] == "正常" else "red")
).add_to(m)
from streamlit.components.v1 import html
html(m._repr_html_(), height=500)

# ---------------------- 右侧：心跳控制 ----------------------
with col_log:
    st.subheader("心跳监测")
    if st.button("发送心跳包"):
        # 更新心跳数据
        st.session_state.drone_data["sequence"] += 1
        st.session_state.drone_data["lat"] += 0.0001  # 模拟位置小幅度变化
        st.session_state.drone_data["lon"] += 0.0001
        st.session_state.drone_data["status"] = "正常" if st.session_state.drone_data["sequence"] % 5 != 0 else "异常"
        
        # 记录日志
        st.session_state.drone_data["heartbeats"].append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "seq": st.session_state.drone_data["sequence"],
            "status": st.session_state.drone_data["status"],
            "pos": f"({round(st.session_state.drone_data['lat'],4)}, {round(st.session_state.drone_data['lon'],4)})"
        })
        st.success(f"心跳 {st.session_state.drone_data['sequence']} 发送成功！")

    # 显示最近10条心跳日志
    st.subheader("心跳日志")
    for hb in reversed(st.session_state.drone_data["heartbeats"][-10:]):
        st.write(f"[{hb['time']}] 序号:{hb['seq']} | {hb['status']} | {hb['pos']}")
