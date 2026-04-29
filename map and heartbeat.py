import streamlit as st
import folium
from folium.plugins import Draw
from datetime import datetime, timedelta
import pandas as pd
from streamlit_folium import st_folium
import time
import math
import geopandas as gpd
from shapely.geometry import Polygon, LineString, Point, MultiLineString
from shapely.ops import unary_union

# -------------------------- 统一使用 GCJ-02 坐标系 --------------------------
# 国内地图（高德、腾讯等）直接使用 GCJ-02 坐标即可显示在正确位置，
# 所以程序内部完全采用 GCJ-02，不再做任何坐标转换。

# -------------------------- 初始化会话状态 --------------------------
if "drone_data" not in st.session_state:
    st.session_state.drone_data = {
        # 起点A（GCJ-02，南京科技职业学院附近坐标示例）
        "lat_a": 32.2322, "lon_a": 118.749,
        # 终点B（GCJ-02）
        "lat_b": 32.2343, "lon_b": 118.749,
        # 无人机当前位置（GCJ-02）
        "current_lat": 32.2322, "current_lon": 118.749,
        "sequence": 0,
        "status": "正常",
        "heartbeats": [],
        "last_receive_time": datetime.now(),
        "obstacles": [],        # 障碍物列表，结构：{"name":..., "coords":[[lon,lat],...], "height":...}
        "map_tile": "satellite", # 卫星/普通
        "waypoints": [],        # 绕飞生成的航路点（GCJ-02）
        "awaiting_obstacle": None, # 待保存的障碍物（临时存储 Draw 数据）
    }

# 页面配置
st.set_page_config(page_title="无人机心跳+地图Demo", layout="wide")
st.title("无人机智能化应用 - 心跳监测 + 地图 + 障碍物圈选（支持高度/绕飞）")

# ---------------------- 左侧：坐标设置 + 飞行参数 + 障碍物管理 ----------------------
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("📍 坐标设置（GCJ-02 坐标系）")
    st.markdown("**起点A**")
    lat_a = st.number_input("纬度A", value=st.session_state.drone_data["lat_a"], format="%.6f", key="lat_a")
    lon_a = st.number_input("经度A", value=st.session_state.drone_data["lon_a"], format="%.6f", key="lon_a")
    st.markdown("**终点B**")
    lat_b = st.number_input("纬度B", value=st.session_state.drone_data["lat_b"], format="%.6f", key="lat_b")
    lon_b = st.number_input("经度B", value=st.session_state.drone_data["lon_b"], format="%.6f", key="lon_b")

    if st.button("更新坐标并校验"):
        st.session_state.drone_data["lat_a"] = lat_a
        st.session_state.drone_data["lon_a"] = lon_a
        st.session_state.drone_data["lat_b"] = lat_b
        st.session_state.drone_data["lon_b"] = lon_b
        st.success("坐标已更新！（GCJ-02）")

    st.subheader("🗺️ 地图设置")
    tile_option = st.radio("地图类型", ["卫星地图", "普通地图"],
                          index=0 if st.session_state.drone_data["map_tile"] == "satellite" else 1)
    st.session_state.drone_data["map_tile"] = "satellite" if tile_option == "卫星地图" else "normal"

    st.subheader("✈️ 飞行参数")
    height = st.slider("设定飞行高度(m)", 10, 150, 50, key="height")
    auto_send = st.checkbox("自动发送心跳（每秒1次）", key="auto_send")

    st.subheader("🚧 障碍物管理")
    obstacle_count = len(st.session_state.drone_data["obstacles"])
    if obstacle_count:
        st.info(f"已保存 {obstacle_count} 个障碍物区域")
        if st.button("清空所有障碍物"):
            st.session_state.drone_data["obstacles"] = []
            st.success("已清空")
            st.rerun()
    else:
        st.info("暂无障碍物，可在地图上用多边形圈选")

# ---------------------- 右侧：地图显示（直接 GCJ-02） ----------------------
with col_right:
    st.subheader("🗺️ 校园地图（GCJ-02 坐标系）")
    # 地图中心取 A/B 中点
    center_lat = (lat_a + lat_b) / 2
    center_lon = (lon_a + lon_b) / 2
    map_center = [center_lat, center_lon]

    # 选择瓦片（高德地图，GCJ-02 原生匹配）
    if st.session_state.drone_data["map_tile"] == "satellite":
        tile_url = "https://webst01.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}"
        tile_attr = "高德卫星地图"
    else:
        tile_url = "https://wprd01.is.autonavi.com/appmaptile?x={x}&y={y}&z={z}&lang=zh_cn&size=1&scl=1&style=8"
        tile_attr = "高德地图"

    # 初始化地图
    m = folium.Map(
        location=map_center,
        zoom_start=18,
        tiles=tile_url,
        attr=tile_attr
    )

    # 起点、终点、无人机标记（均直接用 GCJ-02）
    folium.Marker([lat_a, lon_a], popup="起点A", icon=folium.Icon(color="red")).add_to(m)
    folium.Marker([lat_b, lon_b], popup="终点B", icon=folium.Icon(color="green")).add_to(m)

    current_lat = st.session_state.drone_data["current_lat"]
    current_lon = st.session_state.drone_data["current_lon"]
    folium.Marker(
        [current_lat, current_lon],
        popup=f"无人机\n序号:{st.session_state.drone_data['sequence']}\n状态:{st.session_state.drone_data['status']}",
        icon=folium.Icon(color="blue" if st.session_state.drone_data["status"] == "正常" else "red")
    ).add_to(m)

    # 绘制原始 A->B 直线（浅蓝色虚线）
    folium.PolyLine(
        locations=[[lat_a, lon_a], [lat_b, lon_b]],
        color="dodgerblue", weight=2, opacity=0.5, dash_array="5,5"
    ).add_to(m)

    # 绘制绕飞路径（如果有）
    waypoints = st.session_state.drone_data.get("waypoints", [])
    if waypoints:
        # 路径从起点开始，经过所有航路点，到终点
        path = [[lat_a, lon_a]] + [[wp[1], wp[0]] for wp in waypoints] + [[lat_b, lon_b]]
        folium.PolyLine(path, color="orange", weight=4, opacity=0.9).add_to(m)

    # 加载已保存的障碍物（GCJ-02 坐标）
    for obs in st.session_state.drone_data["obstacles"]:
        coords = [[lat, lon] for (lon, lat) in obs["coords"]]  # 注意我们的存储是 [lon, lat]
        # 根据高度调整颜色深度
        h = obs.get("height", 0)
        opacity = min(0.3 + h / 150, 0.7)
        folium.Polygon(
            locations=coords,
            color="red",
            weight=2,
            fill_color="red",
            fill_opacity=opacity,
            popup=f"{obs['name']} (高度:{h}m)"
        ).add_to(m)

    # 多边形圈选工具
    draw = Draw(
        draw_options={
            "polyline": False,
            "polygon": True,
            "circle": False,
            "rectangle": False,
            "marker": False,
            "circlemarker": False
        },
        edit_options={"edit": True, "remove": True}
    )
    draw.add_to(m)

    # 使用 st_folium 展示地图并捕获绘图数据
    map_data = st_folium(m, width=700, height=500, key="map",
                         returned_objects=["all_drawings"])

    # 处理新绘制的障碍物
    if map_data and "all_drawings" in map_data and map_data["all_drawings"]:
        drawings = map_data["all_drawings"]
        # 检查是否有新增的绘图（与已保存数量比较，粗略判断）
        current_draw_count = len(drawings)
        saved_count = len(st.session_state.drone_data["obstacles"])
        if current_draw_count > saved_count:
            # 最新绘制的多边形作为待保存障碍物
            latest_drawing = drawings[-1]
            if "geometry" in latest_drawing and latest_drawing["geometry"]["type"] == "Polygon":
                coords_raw = latest_drawing["geometry"]["coordinates"][0]
                # 转换为 [lon, lat] 列表
                coords = [[pt[0], pt[1]] for pt in coords_raw]
                st.session_state.drone_data["awaiting_obstacle"] = coords

    # 保存障碍物的弹出窗
    if st.session_state.drone_data.get("awaiting_obstacle"):
        with st.expander("📌 保存新圈选的障碍物", expanded=True):
            obstacle_name = st.text_input("障碍物名称", value=f"障碍物{len(st.session_state.drone_data['obstacles'])+1}")
            obstacle_height = st.number_input("障碍物高度(m)", min_value=0, value=20, step=5)
            if st.button("保存障碍物"):
                new_obs = {
                    "name": obstacle_name,
                    "coords": st.session_state.drone_data["awaiting_obstacle"],
                    "height": obstacle_height,
                    "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.drone_data["obstacles"].append(new_obs)
                st.session_state.drone_data["awaiting_obstacle"] = None
                st.success(f"障碍物「{obstacle_name}」已保存！")
                st.rerun()

# ---------------------- 下方：心跳监测 + 掉线检测 + 可视化 ----------------------
st.divider()
col_heartbeat, col_chart = st.columns([1, 1])

with col_heartbeat:
    st.subheader("❤️ 心跳监测")

    # 手动发送心跳
    if st.button("发送心跳包", key="send_heart"):
        # 调用绕飞算法更新航路点
        update_waypoints = True
        st.session_state.drone_data["sequence"] += 1
        st.session_state.drone_data["last_receive_time"] = datetime.now()

        # 如果有障碍物，计算绕飞路径
        obstacles = st.session_state.drone_data.get("obstacles", [])
        if obstacles and update_waypoints:
            start = (lon_a, lat_a)        # (lon, lat) 格式
            end = (lon_b, lat_b)
            new_waypoints = compute_avoidance_path(start, end, obstacles, flight_height=height)
            st.session_state.drone_data["waypoints"] = new_waypoints

        # 推进无人机位置
        move_drone(step=0.0001)  # 每次移动一小步
        # 记录心跳日志
        st.session_state.drone_data["heartbeats"].append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "seq": st.session_state.drone_data["sequence"],
            "status": st.session_state.drone_data["status"],
            "height": height
        })
        st.success(f"心跳 {st.session_state.drone_data['sequence']} 发送成功！")
        st.rerun()

    # 掉线检测
    time_diff = datetime.now() - st.session_state.drone_data["last_receive_time"]
    if time_diff > timedelta(seconds=3):
        st.session_state.drone_data["status"] = "超时"
        st.error(f"⚠️ 连接超时！{time_diff.seconds}秒未收到心跳包！")
    else:
        st.info(f"✅ 连接正常（上次心跳: {time_diff.microseconds//1000}ms前）")

    # 心跳日志
    st.subheader("📝 心跳日志（最近10条）")
    heartbeats = st.session_state.drone_data["heartbeats"]
    if heartbeats:
        for hb in reversed(heartbeats[-10:]):
            st.write(f"[{hb['time']}] 序号:{hb['seq']} | 状态:{hb['status']} | 高度:{hb['height']}m")
    else:
        st.info("暂无心跳数据")

with col_chart:
    st.subheader("📊 数据可视化")
    if heartbeats:
        df = pd.DataFrame(heartbeats)
        df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S")
        st.line_chart(df, x="time", y="seq", color="#00aaff", height=200, use_container_width=True)
        status_df = df["status"].value_counts().reset_index()
        status_df.columns = ["状态", "数量"]
        color_map = {"正常": "#22c55e", "超时": "#ef4444"}
        status_df["color"] = status_df["状态"].map(color_map)
        st.bar_chart(status_df, x="状态", y="数量", color="color", height=200, use_container_width=True)
    else:
        st.info("暂无心跳数据")

# ---------------------- 自动发送心跳逻辑 ----------------------
if auto_send:
    time.sleep(1)
    st.session_state.drone_data["sequence"] += 1
    st.session_state.drone_data["last_receive_time"] = datetime.now()

    # 自动计算绕飞路径
    obstacles = st.session_state.drone_data.get("obstacles", [])
    if obstacles:
        new_waypoints = compute_avoidance_path(
            (st.session_state.drone_data["lon_a"], st.session_state.drone_data["lat_a"]),
            (st.session_state.drone_data["lon_b"], st.session_state.drone_data["lat_b"]),
            obstacles, flight_height=height
        )
        st.session_state.drone_data["waypoints"] = new_waypoints

    move_drone(step=0.0001)
    st.session_state.drone_data["heartbeats"].append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "seq": st.session_state.drone_data["sequence"],
        "status": st.session_state.drone_data["status"],
        "height": height
    })
    st.rerun()


# -------------------------- 辅助函数：无人机移动、绕飞算法 --------------------------
def move_drone(step=0.0001):
    """将无人机向当前目标点移动一小步（GCJ-02 坐标）"""
    waypoints = st.session_state.drone_data.get("waypoints", [])
    lat_a = st.session_state.drone_data["lat_a"]
    lon_a = st.session_state.drone_data["lon_a"]
    lat_b = st.session_state.drone_data["lat_b"]
    lon_b = st.session_state.drone_data["lon_b"]

    # 目标点列表：起点 -> 航路点 -> 终点
    targets = [(lon_a, lat_a)] + waypoints + [(lon_b, lat_b)]
    current = (st.session_state.drone_data["current_lon"], st.session_state.drone_data["current_lat"])

    # 寻找下一个未到达的目标
    target = None
    for pt in targets:
        if distance(current, pt) > 1e-6:
            target = pt
            break
    if target is None:
        # 已到达终点，停留
        return

    # 向目标移动一小步
    dx = target[0] - current[0]
    dy = target[1] - current[1]
    dist = math.sqrt(dx*dx + dy*dy)
    if dist < step:
        # 直接到达目标点
        st.session_state.drone_data["current_lon"] = target[0]
        st.session_state.drone_data["current_lat"] = target[1]
    else:
        ratio = step / dist
        st.session_state.drone_data["current_lon"] = current[0] + dx * ratio
        st.session_state.drone_data["current_lat"] = current[1] + dy * ratio

    # 更新状态
    st.session_state.drone_data["status"] = "正常"


def distance(p1, p2):
    """两点间的欧氏距离（近似，小范围内）"""
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def compute_avoidance_path(start, end, obstacles, flight_height):
    """
    计算避开障碍物的二维路径。
    start, end: (lon, lat) 元组
    obstacles: 障碍物列表，每个包含 "coords" ([[lon,lat],...]) 和 "height"
    flight_height: 当前飞行高度
    返回: (lon, lat) 列表的航路点（不包括起点和终点）
    """
    line = LineString([start, end])
    # 收集所有需要避开的障碍物多边形
    blocking_obs = []
    for obs in obstacles:
        obs_height = obs.get("height", 0)
        if flight_height >= obs_height:
            continue  # 可从上方飞过，忽略
        poly_coords = [(lon, lat) for (lon, lat) in obs["coords"]]
        # 确保多边形闭合
        if poly_coords[0] != poly_coords[-1]:
            poly_coords.append(poly_coords[0])
        poly = Polygon(poly_coords)
        if line.intersects(poly):
            blocking_obs.append((poly, obs_height))

    if not blocking_obs:
        return []  # 无障碍或均在上方飞越，直线即可

    # 对每个障碍物，尝试左右绕行
    # 简化处理：取所有障碍物合并的外边界，然后找到左右偏移点
    try:
        all_polys = [obs[0] for obs in blocking_obs]
        merged = unary_union(all_polys)
        # 如果合并后为 MultiPolygon 则取各个部分
        if merged.geom_type == "MultiPolygon":
            polys = list(merged.geoms)
        else:
            polys = [merged]
    except:
        return []  # 出错时保持直线

    # 对于每个部分，计算左右绕行候选点
    def left_right_points(poly, line):
        """
        返回多边形在直线两侧的最远点（左、右）
        左侧定义为从起点看向终点的左侧（叉积为正）
        """
        coords = list(poly.exterior.coords)
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        left_pts, right_pts = [], []
        for pt in coords:
            # 计算向量 (pt - start) 与方向向量的叉积
            vx = pt[0] - start[0]
            vy = pt[1] - start[1]
            cross = dx * vy - dy * vx
            if cross > 0:
                left_pts.append((cross, pt))
            elif cross < 0:
                right_pts.append((-cross, pt))  # 右边取绝对值方便比较
        # 取离直线最远的点
        best_left = max(left_pts, key=lambda x: x[0])[1] if left_pts else None
        best_right = max(right_pts, key=lambda x: x[0])[1] if right_pts else None
        return best_left, best_right

    # 收集所有可能的绕行候选点
    left_candidates, right_candidates = [], []
    for poly in polys:
        l, r = left_right_points(poly, line)
        if l:
            left_candidates.append(l)
        if r:
            right_candidates.append(r)

    # 简单策略：生成左绕路径 起点->左点1->...->左点k->终点（按距离排序）
    if left_candidates:
        # 按距离起点的远近排序
        left_candidates.sort(key=lambda pt: distance(start, pt))
        left_path = [start] + left_candidates + [end]
    else:
        left_path = None

    if right_candidates:
        right_candidates.sort(key=lambda pt: distance(start, pt))
        right_path = [start] + right_candidates + [end]
    else:
        right_path = None

    # 选总长度较短的路径
    if left_path and right_path:
        len_left = sum(distance(left_path[i], left_path[i+1]) for i in range(len(left_path)-1))
        len_right = sum(distance(right_path[i], right_path[i+1]) for i in range(len(right_path)-1))
        chosen = left_path if len_left < len_right else right_path
    elif left_path:
        chosen = left_path
    else:
        chosen = right_path

    if chosen is None:
        return []

    # 去掉首尾（起点和终点），保留中间航路点
    waypoints = chosen[1:-1]
    return waypoints
