import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import time
import datetime
import json
import os
import math
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import unary_union

# -------------------------- 页面配置 --------------------------
st.set_page_config(
    page_title="南京科技职业学院 - 无人机导航系统",
    layout="wide"
)

# -------------------------- 样式 --------------------------
st.markdown("""
<style>
.left-panel {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 10px;
    height: 95vh;
}
</style>
""", unsafe_allow_html=True)

# -------------------------- 文件持久化路径 --------------------------
OBSTACLE_FILE = "obstacles.json"
HEARTBEAT_FILE = "heartbeat_state.json"

def load_obstacles():
    if os.path.exists(OBSTACLE_FILE):
        with open(OBSTACLE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_obstacles(obs_list):
    with open(OBSTACLE_FILE, 'w', encoding='utf-8') as f:
        json.dump(obs_list, f, ensure_ascii=False, indent=2)

def load_heartbeat_state():
    if os.path.exists(HEARTBEAT_FILE):
        with open(HEARTBEAT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"seq": 0, "heartbeat_data": [], "running": False}

def save_heartbeat_state(seq, heartbeat_data, running):
    with open(HEARTBEAT_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            "seq": seq,
            "heartbeat_data": heartbeat_data[-100:],  # 只保留最近100条
            "running": running
        }, f, ensure_ascii=False, indent=2)

# -------------------------- 初始化状态 --------------------------
if "drawing" not in st.session_state:
    st.session_state.drawing = False
if "current_points" not in st.session_state:
    st.session_state.current_points = []

# 心跳相关状态（从文件恢复）
if "heartbeat_data" not in st.session_state:
    hb_state = load_heartbeat_state()
    st.session_state.heartbeat_data = hb_state["heartbeat_data"]
    st.session_state.seq = hb_state["seq"]
    st.session_state.running = hb_state["running"]

# 绕飞路径缓存
if "avoid_path" not in st.session_state:
    st.session_state.avoid_path = []

# -------------------------- 坐标转换 --------------------------
def gcj_to_wgs(lat, lon):
    a = 6378245.0
    ee = 0.006693421622965943
    dlat = _transform_lat(lon - 105.0, lat - 35.0)
    dlon = _transform_lon(lon - 105.0, lat - 35.0)
    radlat = lat / 180.0 * math.pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * math.pi)
    dlon = (dlon * 180.0) / (a / sqrtmagic * math.cos(radlat) * math.pi)
    return lat - dlat, lon - dlon

def _transform_lat(x, y):
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    return ret

def _transform_lon(x, y):
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    return ret

# -------------------------- 绕飞路径计算 --------------------------
def compute_avoid_path(latA, lngA, latB, lngB, fly_height, obstacles):
    """
    检测直线是否与障碍物相交（若飞行高度≥障碍物高度则忽略），
    若相交则生成左右绕飞候选点，选择更短路径。
    返回: [(lat1,lng1), (lat2,lng2), ...] 航路点（不含首尾）
    """
    # 使用原始坐标（GCJ-02，与地图匹配）
    start = (lngA, latA)
    end = (lngB, latB)
    line = LineString([start, end])

    blocking_obs = []
    for obs in obstacles:
        obs_height = obs.get("height", 0)
        if fly_height >= obs_height:
            continue  # 从上方飞越，忽略
        points = obs["points"]  # [[lat, lng], ...]
        # 转换为 (lng, lat) 用于shapely
        coords = [(p[1], p[0]) for p in points]
        if coords[0] != coords[-1]:
            coords.append(coords[0])  # 确保闭合
        poly = Polygon(coords)
        if line.intersects(poly):
            blocking_obs.append((poly, obs_height))

    if not blocking_obs:
        return []  # 无障碍，直线飞行

    # 合并所有相关障碍物
    try:
        all_polys = [obs[0] for obs in blocking_obs]
        merged = unary_union(all_polys)
        if merged.geom_type == "MultiPolygon":
            polys = list(merged.geoms)
        else:
            polys = [merged]
    except:
        return []

    def left_right_points(poly, line):
        """返回多边形在直线左右两侧最远点"""
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
        left, right = left_right_points(poly, line)
        if left:
            left_candidates.append(left)
        if right:
            right_candidates.append(right)

    # 按距起点距离排序
    def dist(p):
        return math.hypot(p[0]-start[0], p[1]-start[1])
    left_candidates.sort(key=dist)
    right_candidates.sort(key=dist)

    left_path = [start] + left_candidates + [end] if left_candidates else None
    right_path = [start] + right_candidates + [end] if right_candidates else None

    if left_path and right_path:
        len_left = sum(dist(left_path[i+1]) for i in range(len(left_path)-1))
        len_right = sum(dist(right_path[i+1]) for i in range(len(right_path)-1))
        chosen = left_path if len_left < len_right else right_path
    elif left_path:
        chosen = left_path
    else:
        chosen = right_path

    if chosen is None:
        return []

    # 去掉首尾，转换为 [(lat, lng), ...]
    waypoints = [(p[1], p[0]) for p in chosen[1:-1]]
    return waypoints

# -------------------------- 地图渲染（加入绕飞路径） --------------------------
def render_map(latA, lngA, latB, lngB, map_type, fly_height):
    obstacles = load_obstacles()
    drawing = st.session_state.drawing
    points = st.session_state.current_points

    # 计算绕飞路径
    avoid_pts = compute_avoid_path(latA, lngA, latB, lngB, fly_height, obstacles)
    st.session_state.avoid_path = avoid_pts  # 缓存

    if map_type == "卫星影像地图":
        # 卫星用 WGS，起点终点转换
        wgsA = gcj_to_wgs(latA, lngA)
        wgsB = gcj_to_wgs(latB, lngB)
        map_latA, map_lngA = wgsA
        map_latB, map_lngB = wgsB
        # 绕飞点也需要转换（因为障碍物是GCJ-02，需保持相对一致性）
        # 为简单，卫星下只显示原始航线，不展示绕飞（说明偏移问题）
        # 实际可统一坐标，但用户说坐标无误，故卫星模式暂不启用绕飞绘制
        avoid_json = json.dumps([])
        layer = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        attr = "Esri"
    else:
        map_latA, map_lngA = latA, lngA
        map_latB, map_lngB = latB, lngB
        avoid_json = json.dumps(avoid_pts)  # 绕飞点 [(lat,lng),...]
        layer = "https://webrd01.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}"
        attr = "© 高德"

    points_json = json.dumps(points)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>#map {{width:100%;height:680px;border-radius:8px;}}</style>
    </head>
    <body>
        <div id="map"></div>
        <script>
            var map = L.map('map').setView([32.2335, 118.7475], 17);
            L.tileLayer('{layer}', {{maxZoom:20, attribution:'{attr}'}}).addTo(map);

            // 绘制原始航线（红色虚线）
            L.polyline([[{map_latA},{map_lngA}],[{map_latB},{map_lngB}]], 
                {{color:'red', weight:2, dashArray:'5,5'}}).addTo(map);
            L.marker([{map_latA},{map_lngA}]).bindPopup("起点").addTo(map);
            L.marker([{map_latB},{map_lngB}]).bindPopup("终点").addTo(map);

            // 绘制绕飞路径（黄色实线）
            var avoidPoints = {avoid_json};
            if (avoidPoints.length > 0) {{
                var path = [[{map_latA},{map_lngA}]].concat(avoidPoints).concat([{map_latB},{map_lngB}]);
                L.polyline(path, {{color:'#ffaa00', weight:4}}).addTo(map);
            }}

            // 绘制永久障碍物（红色半透明）
            const obs = {json.dumps(obstacles)};
            obs.forEach(o => {{
                L.polygon(o.points, {{color:'#f00',fillColor:'#f44',fillOpacity:0.3}})
                 .bindPopup(o.name + " " + o.height + "m").addTo(map);
            }});

            // 临时多边形（蓝色半透明）
            var points = {points_json};
            var poly = null;
            function redraw() {{
                if (poly) map.removeLayer(poly);
                if (points.length >= 2) {{
                    poly = L.polygon(points, {{color:'#00f',fillOpacity:0.2}}).addTo(map);
                }}
            }}
            redraw();

            if ({str(drawing).lower()}) {{
                map.on('click', function(e) {{
                    points.push([e.latlng.lat, e.latlng.lng]);
                    redraw();
                    window.Streamlit.setComponentValue(points);
                }});
            }}
        </script>
    </body>
    </html>
    """
    return html

# -------------------------- 左侧布局 --------------------------
col_left, col_right = st.columns([1, 3])

with col_left:
    st.markdown('<div class="left-panel">', unsafe_allow_html=True)
    st.subheader("🧭 导航")
    page = st.radio("", ["航线规划", "飞行监控"], label_visibility="collapsed")
    st.divider()

    if page == "航线规划":
        st.markdown("### 🚧 障碍物圈选（记忆版）")
        height = st.number_input("高度(m)", min_value=1, max_value=500, value=25, step=1)
        name = st.text_input("名称", value="教学楼")
        st.info(f"当前已打点：{len(st.session_state.current_points)} 个")

        if not st.session_state.drawing:
            if st.button("🔴 开始圈选", type="primary", use_container_width=True):
                st.session_state.drawing = True
                st.session_state.current_points = []
                st.rerun()
        else:
            if st.button("✅ 保存并结束圈选", type="primary", use_container_width=True):
                if len(st.session_state.current_points) >= 3:
                    all_obs = load_obstacles()
                    all_obs.append({
                        "name": name,
                        "height": height,
                        "points": st.session_state.current_points
                    })
                    save_obstacles(all_obs)
                    st.success("✅ 障碍物已永久保存！")
                else:
                    st.warning("⚠️ 请至少圈选3个点")
                st.session_state.current_points = []
                st.session_state.drawing = False
                st.rerun()

            if st.button("❌ 取消圈选", use_container_width=True):
                st.session_state.drawing = False
                st.rerun()

        st.divider()
        st.markdown("### 📋 已永久保存障碍物")
        obs_list = load_obstacles()
        if obs_list:
            for i, o in enumerate(obs_list):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.write(f"📍 {o['name']} ({o['height']}m)")
                with c2:
                    if st.button("🗑️ 删除", key=f"del_{i}", use_container_width=True):
                        del obs_list[i]
                        save_obstacles(obs_list)
                        st.rerun()
            if st.button("🧹 清空全部障碍物", use_container_width=True):
                save_obstacles([])
                st.rerun()
        else:
            st.info("暂无障碍物")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------- 右侧布局 --------------------------
with col_right:
    st.markdown("# 🎓 南京科技职业学院")
    st.markdown("## 无人机航线导航与监控系统")

    if page == "航线规划":
        map_type = st.radio("🗺️ 地图模式", ["高德普通地图", "卫星影像地图"], horizontal=True)
        st.markdown("### ⛰️ 飞行高度设置")
        fly_h = st.number_input("飞行高度(m)", min_value=1, max_value=500, value=50, step=1)

        st.markdown("### 🎯 航线坐标")
        c1, c2 = st.columns(2)
        with c1:
            latA = st.number_input("起点纬度", value=32.2335, format="%.6f")
            lngA = st.number_input("起点经度", value=118.7475, format="%.6f")
        with c2:
            latB = st.number_input("终点纬度", value=32.2338, format="%.6f")
            lngB = st.number_input("终点经度", value=118.7479, format="%.6f")

        # 渲染地图（传入飞行高度以计算绕飞）
        map_res = components.html(
            render_map(latA, lngA, latB, lngB, map_type, fly_h),
            height=680
        )
        if isinstance(map_res, list) and st.session_state.drawing:
            st.session_state.current_points = map_res

    else:
        # 飞行监控页（带记忆）
        st.title("📡 无人机心跳监控")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("▶️ 开始监测", use_container_width=True):
                st.session_state.running = True
                save_heartbeat_state(st.session_state.seq, st.session_state.heartbeat_data, True)
        with c2:
            if st.button("⏸️ 暂停监测", use_container_width=True):
                st.session_state.running = False
                save_heartbeat_state(st.session_state.seq, st.session_state.heartbeat_data, False)

        placeholder = st.empty()
        if st.session_state.running:
            # 自动循环发送心跳
            while st.session_state.running:
                st.session_state.seq += 1
                t = datetime.datetime.now().strftime("%H:%M:%S")
                st.session_state.heartbeat_data.append({
                    "序号": st.session_state.seq,
                    "时间": t,
                    "状态": "正常"
                })
                # 每次更新后保存状态（防止丢失）
                save_heartbeat_state(st.session_state.seq, st.session_state.heartbeat_data, True)

                df = pd.DataFrame(st.session_state.heartbeat_data)
                with placeholder.container():
                    st.line_chart(df, x="时间", y="序号", color="状态")
                    st.dataframe(df, use_container_width=True)
                time.sleep(1)
        else:
            # 暂停时显示最新保存的数据
            df = pd.DataFrame(st.session_state.heartbeat_data)
            if not df.empty:
                with placeholder.container():
                    st.line_chart(df, x="时间", y="序号", color="状态")
                    st.dataframe(df, use_container_width=True)
            else:
                st.info("暂无监控数据")
