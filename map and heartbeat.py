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
from shapely.geometry import LineString, Polygon

# ===================== 数据持久化 =====================
SAVE_FILE = "drone_data.json"
def load_all_data():
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE,"r",encoding="utf-8") as f:
            return json.load(f)
    return {"A":[32.2322,118.7490],"B":[32.2343,118.7490],"A_set":False,"B_set":False,"obstacles":[]}
def save_all_data():
    data={
        "A":list(st.session_state.A),"B":list(st.session_state.B),
        "A_set":st.session_state.A_set,"B_set":st.session_state.B_set,
        "obstacles":st.session_state.polygon_memory
    }
    with open(SAVE_FILE,"w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)

# ===================== 初始化状态 =====================
data=load_all_data()
default_states = {
    "A": tuple(data["A"]), "B": tuple(data["B"]),
    "A_set": data["A_set"], "B_set": data["B_set"],
    "height": 50, "heartbeat_data": [], "polygon_memory": data["obstacles"],
    "is_drawing": False, "temp_points": [], "obs_h": 20, "last_click_time": 0,
    "safe_radius": 0.0002,
    "flight_running": False, "flight_paused": False, "current_wp_idx": 0,
    "flight_speed": 8.5, "flight_start_time": None, "flight_waypoints": [],
    "battery": 100.0, "total_distance": 0.0, "elapsed_distance": 0.0
}
for key, val in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ===================== 页面配置 =====================
st.set_page_config(layout="wide",page_title="南科院无人机航线规划系统")

# ===================== 坐标转换 =====================
def gcj02_to_wgs84(lng:float,lat:float):
    a=6378245.0
    ee=0.00669342162296594323
    def transform_lat(x,y):
        ret=-100.0+2.0*x+3.0*y+0.2*y*y+0.1*x*y+0.2*np.sqrt(abs(x))
        ret+=(20.0*np.sin(6.0*x*np.pi)+20.0*np.sin(2.0*x*np.pi))*2.0/3.0
        ret+=(20.0*np.sin(y*np.pi)+40.0*np.sin(y/3.0*np.pi))*2.0/3.0
        ret+=(160.0*np.sin(y/12.0*np.pi)+320*np.sin(y/30.0*np.pi))*2.0/3.0
        return ret
def transform_lng(x,y):
        ret=300.0+x+2.0*y+0.1*x*x+0.1*x*y+0.1*np.sqrt(abs(x))
        ret+=(20.0*np.sin(6.0*x*np.pi)+20.0*np.sin(2.0*x*np.pi))*2.0/3.0
        ret+=(20.0*np.sin(x*np.pi)+40.0*np.sin(x/3.0*np.pi))*2.0/3.0
        ret+=(150.0*np.sin(x/12.0*np.pi)+300.0*np.sin(x/30.0*np.pi))*2.0/3.0
        return ret
    dlat=transform_lat(lng-105.0,lat-35.0)
    dlng=transform_lng(lng-105.0,lat-35.0)
    radlat=lat/180.0*np.pi
    magic=np.sin(radlat)
    magic=1-ee*magic*magic
    sqrtmagic=np.sqrt(magic)
    dlat=(dlat*180.0)/((a*(1-ee))/(magic*sqrtmagic)*np.pi)
    dlng=(dlng*180.0)/(a/sqrtmagic*np.cos(radlat)*np.pi)
    return lat-dlat,lng-dlng

# ===================== 安全半径航线生成 =====================
def calc_route_lines(pA,pB,offset=0.0001):
    latA,lonA=pA
    latB,lonB=pB
    dx=lonB-lonA
    dy=latB-latA
    L=np.hypot(dx,dy)
    if L<1e-8:L=1e-8
    left_off_x=-dy/L*offset
    left_off_y=dx/L*offset
    right_off_x=dy/L*offset
    right_off_y=-dx/L*offset
    left=[[latA,lonA],[latA+left_off_y,lonA+left_off_x],[latB+left_off_y,lonB+left_off_x],[latB,lonB]]
    right=[[latA,lonA],[latA+right_off_y,lonA+right_off_x],[latB+right_off_y,lonB+right_off_x],[latB,lonB]]
    return left,right

def get_safe_route(pA, pB, obstacles, safe_dist):
    base_line = LineString([pA, pB])
    obs_polygons = []
    for obs in obstacles:
        pts = obs["pts"]
        if len(pts)>=3:
            poly = Polygon(pts).buffer(safe_dist)
            obs_polygons.append(poly)
    conflict = False
    for poly in obs_polygons:
        if base_line.intersects(poly):
            conflict = True
            break
    if not conflict:
        return [pA, pB], False
    left_line, _ = calc_route_lines(pA, pB, offset=safe_dist)
    return left_line, True

# ===================== 侧边栏 =====================
with st.sidebar:
    st.title("🚁 无人机系统导航")
    page=option_menu("功能页面",["航线规划","飞行监控"],default_index=0)
    st.divider()
    st.subheader("坐标系转换")
    coord_type=st.radio("",["GCJ-02(火星坐标)","WGS-84(原始坐标)"])
    st.divider()
    st.subheader("系统点位状态")
    st.button("✅ A点已设置" if st.session_state.A_set else "❌ A点未设置",type="primary")
    st.button("✅ B点已设置" if st.session_state.B_set else "❌ B点未设置",type="primary")
    st.divider()
    st.subheader("🛡️ 安全半径配置")
    st.session_state.safe_radius = st.slider("航线与障碍物安全距离", 0.00005, 0.0005, value=st.session_state.safe_radius, step=0.00001, format="%.5f")

# ===================== 航线规划页面 =====================
if page=="航线规划":
    st.title("🚁 南京科技职业学院 无人机3D航线规划系统")
    col_map,col_ctrl=st.columns([3.2,1])
    with col_ctrl:
        st.subheader("🎛️ 点位与飞行参数")
        a_lat=st.number_input("起点A 纬度",value=st.session_state.A[0],format="%.6f")
        a_lon=st.number_input("起点A 经度",value=st.session_state.A[1],format="%.6f")
        b_lat=st.number_input("终点B 纬度",value=st.session_state.B[0],format="%.6f")
        b_lon=st.number_input("终点B 经度",value=st.session_state.B[1],format="%.6f")
        st.session_state.height=st.slider("无人机飞行高度 (m)",0,200,value=st.session_state.height)
        if st.button("确定设置起点A"):
            st.session_state.A=(a_lat,a_lon)
            st.session_state.A_set=True
            save_all_data()
            st.success("A点坐标已永久保存！")
        if st.button("确定设置终点B"):
            st.session_state.B=(b_lat,b_lon)
            st.session_state.B_set=True
            save_all_data()
            st.success("B点坐标已永久保存！")
        st.divider()
        st.subheader("🚧 障碍物区域圈选（带高度）")
        st.session_state.obs_h=st.number_input("障碍物高度(m)",0,300,value=st.session_state.obs_h)
        if st.session_state.is_drawing:
            st.warning(f"🖱️ 正在绘制障碍物，已点击点位：{len(st.session_state.temp_points)} 个")
        else:
            st.info("点击【开始绘制障碍物】，在地图上点击圈选禁飞区")
        btn1,btn2,btn3=st.columns(3)
        with btn1:
            if st.button("开始绘制"):
                st.session_state.is_drawing=True
                st.session_state.temp_points=[]
        with btn2:
            if st.button("撤销上一点"):
                if st.session_state.temp_points:st.session_state.temp_points.pop()
        with btn3:
            if st.button("取消绘制"):
                st.session_state.is_drawing=False
                st.session_state.temp_points=[]
        if st.button("✅ 完成圈选并保存"):
            if len(st.session_state.temp_points)>=3:
                st.session_state.polygon_memory.append({"pts":st.session_state.temp_points.copy(),"h":st.session_state.obs_h})
                save_all_data()
                st.success(f"障碍物已保存，高度：{st.session_state.obs_h}m")
            else:
                st.error("至少需要圈选3个点位！")
            st.session_state.is_drawing=False
            st.session_state.temp_points=[]
            st.rerun()
        if st.button("🗑️ 清空全部障碍物"):
            st.session_state.polygon_memory=[]
            st.session_state.temp_points=[]
            save_all_data()
            st.rerun()
        st.info(f"系统已记忆障碍物总数：{len(st.session_state.polygon_memory)} 个")
        st.divider()
        st.subheader("📊 高度+安全半径双重避障判断")
        need_avoid=False
        for idx,obs in enumerate(st.session_state.polygon_memory):
            oh=obs["h"]
            if st.session_state.height<oh:
                st.error(f"⚠️ 障碍物{idx+1}({oh}m) → 高度不足，需绕飞")
                need_avoid=True
            else:
                st.success(f"✅ 障碍物{idx+1}({oh}m) → 高度安全")
        if st.session_state.A_set and st.session_state.B_set:
            _, conflict = get_safe_route(st.session_state.A, st.session_state.B, st.session_state.polygon_memory, st.session_state.safe_radius)
            if conflict:
                st.warning(f"🛡️ 航线与障碍物距离不足，已自动生成安全绕飞路线")
                need_avoid = True
            else:
                st.success(f"🛡️ 航线与障碍物安全距离充足")
        st.divider()
        st.subheader("❤️ 无人机心跳监测")
        now=datetime.now().strftime("%H:%M:%S")
        st.metric("当前系统时间",now)
        st.session_state.heartbeat_data.append(time.time())
        if len(st.session_state.heartbeat_data)>30:st.session_state.heartbeat_data.pop(0)
        df=pd.DataFrame({"采样点":range(len(st.session_state.heartbeat_data)),"心跳时间戳":st.session_state.heartbeat_data})
        st.line_chart(df.set_index("采样点"),height=150)
        st.success("无人机链路心跳正常")
    with col_map:
        center_lat=(st.session_state.A[0]+st.session_state.B[0])/2
        center_lon=(st.session_state.A[1]+st.session_state.B[1])/2
        m=folium.Map(location=[center_lat,center_lon],zoom_start=19,tiles="https://{s}.tile.openstreetmap.de/tiles/osmde/{z}/{x}/{y}.png",attr="OpenStreetMap 3D",max_zoom=22)
        folium.plugins.Fullscreen(position="topright").add_to(m)
        if coord_type=="GCJ-02(火星坐标)":
            A_wgs=gcj02_to_wgs84(st.session_state.A[1],st.session_state.A[0])
            B_wgs=gcj02_to_wgs84(st.session_state.B[1],st.session_state.B[0])
        else:
            A_wgs=st.session_state.A
            B_wgs=st.session_state.B
        if st.session_state.A_set:folium.CircleMarker(A_wgs,radius=12,color="red",fill=True,fill_color="red",popup="飞行起点A").add_to(m)
        if st.session_state.B_set:folium.CircleMarker(B_wgs,radius=12,color="green",fill=True,fill_color="green",popup="飞行终点B").add_to(m)
        for idx,obs in enumerate(st.session_state.polygon_memory):
            pts=obs["pts"]
            hh=obs["h"]
            if len(pts)>=3:
                folium.Polygon(locations=pts,color="#dc2626",fill=True,fill_color="#dc2626",fill_opacity=0.45,popup=f"障碍物{idx+1}｜高度：{hh}m").add_to(m)
                poly = Polygon(pts).buffer(st.session_state.safe_radius)
                folium.Polygon(locations=list(poly.exterior.coords), color="#ff9900", fill=False, weight=2, dash_array="5 5", popup="安全半径缓冲区").add_to(m)
        if len(st.session_state.temp_points)>0:
            for point in st.session_state.temp_points:
                folium.CircleMarker(point,radius=5,color="#ff7700",fill=True,fill_color="#ff7700").add_to(m)
            folium.PolyLine(st.session_state.temp_points,color="#ff7700",weight=3,dash_array="10 5").add_to(m)
        if st.session_state.A_set and st.session_state.B_set:
            safe_waypoints, need_avoid = get_safe_route(A_wgs, B_wgs, st.session_state.polygon_memory, st.session_state.safe_radius)
            st.session_state.flight_waypoints = safe_waypoints
            if need_avoid:
                folium.PolyLine(safe_waypoints,color="#22bb22",weight=5,popup="⭐ 安全半径最优绕飞航线").add_to(m)
            else:
                folium.PolyLine(safe_waypoints,color="#0066ff",weight=4,opacity=0.8,popup="✅ 安全直飞航线").add_to(m)
        output=st_folium(m,width=1150,height=720,key="main_map_3d")
        if st.session_state.is_drawing and output and output.get("last_clicked"):
            now = time.time()
            if now - st.session_state.last_click_time > 0.5:
                pt = output["last_clicked"]
                click_lat, click_lng = pt["lat"], pt["lng"]
                new_pt = [click_lat, click_lng]
                if not st.session_state.temp_points or new_pt != st.session_state.temp_points[-1]:
                    st.session_state.temp_points.append(new_pt)
                    st.session_state.last_click_time = now
                    st.rerun()

# ===================== 飞行监控页面 =====================
else:
    st.title("📡 飞行实时画面 - 任务执行监控")
    st.success("✅ 无人机系统链路正常，设备在线")
    st.subheader("监测区域：南京科技职业学院校内空域")
    col_btn = st.columns(4)
    with col_btn[0]:
        if st.button("🔴 开始任务", type="primary", disabled=st.session_state.flight_running):
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
    st.caption("状态：" + ("任务运行中" if st.session_state.flight_running and not st.session_state.flight_paused else ("已暂停" if st.session_state.flight_paused else "已停止")))
    st.divider()
    if len(st.session_state.flight_waypoints) < 2:
        st.warning("⚠️ 请先在【航线规划】页面设置A/B点并生成航线！")
    else:
        total_dist = 0
        for i in range(len(st.session_state.flight_waypoints)-1):
            p1 = st.session_state.flight_waypoints[i]
            p2 = st.session_state.flight_waypoints[i+1]
            dist = np.hypot(p2[0]-p1[0], p2[1]-p1[1])
            total_dist += dist
        st.session_state.total_distance = round(total_dist * 111000, 2)
        if st.session_state.flight_running and not st.session_state.flight_paused:
            if st.session_state.current_wp_idx < len(st.session_state.flight_waypoints)-1:
                st.session_state.current_wp_idx += 0.02
                st.session_state.battery = max(0, st.session_state.battery - 0.02)
                st.session_state.elapsed_distance = round(st.session_state.current_wp_idx / (len(st.session_state.flight_waypoints)-1) * st.session_state.total_distance, 2)
            else:
                st.session_state.flight_running = False
                st.success("🎉 任务完成！已到达终点")
        current_wp = int(st.session_state.current_wp_idx) + 1
        total_wp = len(st.session_state.flight_waypoints)
        flight_speed = st.session_state.flight_speed
        elapsed_dist = st.session_state.elapsed_distance
        remain_dist = round(st.session_state.total_distance - elapsed_dist, 2)
        if st.session_state.flight_start_time:
            elapsed_time = datetime.now() - st.session_state.flight_start_time
            elapsed_time_str = str(timedelta(seconds=int(elapsed_time.total_seconds()))).split(".")[0]
        else:
            elapsed_time_str = "00:00"
        if remain_dist > 0 and flight_speed > 0:
            eta_sec = remain_dist / flight_speed
            eta_str = str(timedelta(seconds=int(eta_sec))).split(".")[0]
        else:
            eta_str = "00:00"
        battery = round(st.session_state.battery, 1)
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("当前航点", f"{current_wp}/{total_wp}")
        with col2:
            st.metric("飞行速度", f"{flight_speed} m/s")
        with col3:
            st.metric("已用时间", elapsed_time_str)
        with col4:
            st.metric("剩余距离", f"{remain_dist} m")
        with col5:
            st.metric("预计到达", eta_str)
        with col6:
            st.metric("电量模拟", f"{battery} %")
        progress = st.session_state.current_wp_idx / (len(st.session_state.flight_waypoints)-1) if len(st.session_state.flight_waypoints)>1 else 0
        st.progress(progress, text=f"任务进度：{round(progress*100, 1)}%")
        st.divider()
        col_map_flight, col_status = st.columns([2,1])
        with col_map_flight:
            st.subheader("🗺️ 实时飞行地图")
            m_flight = folium.Map(
                location=st.session_state.flight_waypoints[0],
                zoom_start=19,
                tiles="https://{s}.tile.openstreetmap.de/tiles/osmde/{z}/{x}/{y}.png",
                attr="OpenStreetMap 3D"
            )
            for idx,obs in enumerate(st.session_state.polygon_memory):
                pts=obs["pts"]
                hh=obs["h"]
                if len(pts)>=3:
                    folium.Polygon(locations=pts,color="#dc2626",fill=True,fill_color="#dc2626",fill_opacity=0.45,popup=f"障碍物{idx+1}｜高度：{hh}m").add_to(m_flight)
            folium.PolyLine(st.session_state.flight_waypoints, color="#0066ff", weight=3, opacity=0.5).add_to(m_flight)
            flown_idx = int(st.session_state.current_wp_idx)
            flown_waypoints = st.session_state.flight_waypoints[:flown_idx+1]
            if len(flown_waypoints)>=2:
                folium.PolyLine(flown_waypoints, color="#22bb22", weight=4, popup="✅ 已飞行路径").add_to(m_flight)
            if len(st.session_state.flight_waypoints) > 0:
                drone_pos = st.session_state.flight_waypoints[min(int(st.session_state.current_wp_idx), len(st.session_state.flight_waypoints)-1)]
                folium.CircleMarker(drone_pos, radius=10, color="orange", fill=True, fill_color="orange", popup="🚁 无人机当前位置").add_to(m_flight)
            st_folium(m_flight, width="100%", height=500, key="flight_map")
        with col_status:
            st.subheader("📡 通信链路拓扑与数据流")
            st.success("✅ GCS 在线")
            st.success("✅ OBC 在线")
            st.success("✅ FCU 在线")
            st.divider()
            st.info(f"📍 当前位置：纬度{drone_pos[0]:.6f}，经度{drone_pos[1]:.6f}")
            st.info(f"🛫 飞行高度：{st.session_state.height} m")
            st.info(f"🛡️ 安全半径：{st.session_state.safe_radius}")
            st.info(f"🚧 障碍物数量：{len(st.session_state.polygon_memory)} 个")
    if st.session_state.flight_running and not st.session_state.flight_paused:
        time.sleep(0.1)
        st.rerun()
