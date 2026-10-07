#!/usr/bin/env python3
import math
import os
import re
import time

import cv2
import numpy as np
import yaml
from scipy.interpolate import splev, splprep
from scipy.ndimage import distance_transform_edt

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path


def load_occupancy(yaml_path):
    with open(yaml_path, 'r') as f:
        meta = yaml.safe_load(f)
    base_dir = os.path.dirname(os.path.abspath(yaml_path))
    image_path = meta['image']
    if not os.path.isabs(image_path):
        image_path = os.path.join(base_dir, image_path)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resolution = float(meta['resolution'])
    origin = meta['origin']
    negate = int(meta.get('negate', 0))
    mode = meta.get('mode', 'trinary')
    px = img.astype(np.float64) if negate == 0 else (255.0 - img.astype(np.float64))
    if mode == 'trinary':
        free_mask = px >= 250.0
        occ_mask = px <= 10.0
    else:
        occupied_thresh = float(meta.get('occupied_thresh', 0.65))
        free_thresh = float(meta.get('free_thresh', 0.25))
        occ_prob = 1.0 - px / 255.0
        free_mask = occ_prob < free_thresh
        occ_mask = occ_prob > occupied_thresh
    return free_mask, occ_mask, resolution, (float(origin[0]), float(origin[1]))


def rasterize_walls(free_mask, resolution, origin, walls):
    h, w = free_mask.shape
    combined = free_mask.copy()
    for cx, cy, sx, sy in walls:
        x0, y0 = cx - sx / 2, cy - sy / 2
        x1, y1 = cx + sx / 2, cy + sy / 2
        px0, py0 = world_to_pixel(x0, y1, h, resolution, origin)
        px1, py1 = world_to_pixel(x1, y0, h, resolution, origin)
        ix0, ix1 = sorted((int(math.floor(px0)), int(math.ceil(px1))))
        iy0, iy1 = sorted((int(math.floor(py0)), int(math.ceil(py1))))
        ix0, iy0 = max(ix0, 0), max(iy0, 0)
        ix1, iy1 = min(ix1, w - 1), min(iy1, h - 1)
        if ix1 >= ix0 and iy1 >= iy0:
            combined[iy0:iy1 + 1, ix0:ix1 + 1] = False
    return combined


def load_world_walls(world_path):
    text = open(world_path).read()
    blocks = re.findall(
        r'<collision name="wall_\d+">\s*<pose>([^<]+)</pose>\s*<geometry>\s*<box>\s*<size>([^<]+)</size>',
        text,
    )
    walls = []
    for pose, size in blocks:
        vals = [float(v) for v in pose.split()]
        sx, sy, _ = [float(v) for v in size.split()]
        walls.append((vals[0], vals[1], sx, sy))
    return np.array(walls) if walls else np.zeros((0, 4))


def wall_clearance(walls, x, y):
    if len(walls) == 0:
        return 1e9
    cx, cy, sx, sy = walls[:, 0], walls[:, 1], walls[:, 2], walls[:, 3]
    dx = np.maximum(np.abs(x - cx) - sx / 2.0, 0.0)
    dy = np.maximum(np.abs(y - cy) - sy / 2.0, 0.0)
    return float(np.sqrt(dx ** 2 + dy ** 2).min())


def pixel_to_world(px, py, h, resolution, origin):
    wx = origin[0] + px * resolution
    wy = origin[1] + (h - 1 - py) * resolution
    return wx, wy


def world_to_pixel(wx, wy, h, resolution, origin):
    px = (wx - origin[0]) / resolution
    py = (h - 1) - (wy - origin[1]) / resolution
    return px, py


def sample_mask(mask, px, py):
    h, w = mask.shape
    ix = np.clip(np.round(px).astype(np.int64), 0, w - 1)
    iy = np.clip(np.round(py).astype(np.int64), 0, h - 1)
    return mask[iy, ix]


def sample_field(field, px, py):
    h, w = field.shape
    ix = np.clip(np.round(px).astype(np.int64), 0, w - 1)
    iy = np.clip(np.round(py).astype(np.int64), 0, h - 1)
    return field[iy, ix]


def extract_centerline(free_mask, resolution, origin, n_angles=360):
    h, w = free_mask.shape
    clearance_px = distance_transform_edt(free_mask)
    ys, xs = np.nonzero(free_mask)
    cx_px = xs.mean()
    cy_px = ys.mean()
    max_r = math.hypot(w, h) * resolution
    step = resolution * 0.5
    n_steps = int(max_r / step)
    r = (np.arange(1, n_steps + 1) * step)
    points = []
    for i in range(n_angles):
        theta = 2.0 * math.pi * i / n_angles
        px = cx_px + (r / resolution) * math.cos(theta)
        py = cy_px + (r / resolution) * (-math.sin(theta))
        inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        if not np.any(inside):
            continue
        free_hits = np.zeros(len(r), dtype=bool)
        free_hits[inside] = sample_mask(free_mask, px[inside], py[inside])
        clear_vals = np.zeros(len(r), dtype=np.float64)
        clear_vals[inside] = sample_field(clearance_px, px[inside], py[inside])

        best_peak = -1.0
        best_peak_idx = -1
        k = 0
        n = len(r)
        while k < n:
            if free_hits[k]:
                j = k
                while j < n and free_hits[j]:
                    j += 1
                seg = clear_vals[k:j]
                peak = seg.max()
                if peak > best_peak:
                    best_peak = peak
                    best_peak_idx = k + int(seg.argmax())
                k = j
            else:
                k += 1
        if best_peak_idx < 0:
            continue
        wx, wy = pixel_to_world(px[best_peak_idx], py[best_peak_idx], h, resolution, origin)
        points.append((theta, wx, wy))

    points.sort(key=lambda t: t[0])
    pts = np.array([[p[1], p[2]] for p in points])
    return pts


def project_clear(pts, free_mask, resolution, origin, search_radius=1.2):
    h, w = free_mask.shape
    out = pts.copy()
    ys_free, xs_free = np.nonzero(free_mask)
    free_wx = origin[0] + xs_free * resolution
    free_wy = origin[1] + (h - 1 - ys_free) * resolution
    for i in range(len(pts)):
        x, y = pts[i]
        px, py = world_to_pixel(x, y, h, resolution, origin)
        ix, iy = int(round(px)), int(round(py))
        ok = 0 <= ix < w and 0 <= iy < h and free_mask[iy, ix]
        if ok:
            continue
        d2 = (free_wx - x) ** 2 + (free_wy - y) ** 2
        j = int(np.argmin(d2))
        out[i] = (free_wx[j], free_wy[j])
    return out


def resample_closed_curve(pts, n_out, smooth_factor):
    pts_ext = np.vstack([pts, pts[0:1]])
    tck, u = splprep([pts_ext[:, 0], pts_ext[:, 1]], per=1, s=smooth_factor)
    u_fine = np.linspace(0, 1, 2000, endpoint=False)
    xf, yf = splev(u_fine, tck)
    xf = np.asarray(xf)
    yf = np.asarray(yf)
    seg = np.hypot(np.diff(xf, append=xf[0]), np.diff(yf, append=yf[0]))
    s = np.concatenate([[0.0], np.cumsum(seg)])[:-1]
    total = s[-1] + seg[-1]
    s_target = np.linspace(0, total, n_out, endpoint=False)
    x_out = np.interp(s_target, s, xf, period=total)
    y_out = np.interp(s_target, s, yf, period=total)
    return np.stack([x_out, y_out], axis=1)


def compute_normals(pts):
    nxt = np.roll(pts, -1, axis=0)
    prv = np.roll(pts, 1, axis=0)
    tangent = nxt - prv
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    norm[norm < 1e-9] = 1e-9
    tangent = tangent / norm
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    return tangent, normal


def ray_clearance(free_mask, resolution, origin, origin_xy, direction_xy, max_dist):
    h, w = free_mask.shape
    step = resolution * 0.5
    n_steps = max(1, int(max_dist / step))
    d = np.arange(1, n_steps + 1) * step
    wx = origin_xy[0] + direction_xy[0] * d
    wy = origin_xy[1] + direction_xy[1] * d
    px, py = world_to_pixel(wx, wy, h, resolution, origin)
    inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    if not np.any(inside):
        return 0.0
    occ = ~sample_mask(free_mask, px, py)
    hit_idx = np.where(occ & inside)[0]
    first_outside = np.where(~inside)[0]
    limit = hit_idx[0] if len(hit_idx) > 0 else n_steps
    if len(first_outside) > 0:
        limit = min(limit, first_outside[0])
    return d[limit - 1] if limit > 0 else 0.0


def compute_track_bounds(pts, normals, free_mask, resolution, origin, max_width=3.0):
    left = np.zeros(len(pts))
    right = np.zeros(len(pts))
    for i in range(len(pts)):
        left[i] = ray_clearance(free_mask, resolution, origin, pts[i], normals[i], max_width)
        right[i] = ray_clearance(free_mask, resolution, origin, pts[i], -normals[i], max_width)
    return left, right


def optimize_raceline(pts, normals, left_bound, right_bound, margin, iterations, step_size):
    n = len(pts)
    offset = np.zeros(n)
    lo = np.minimum(0.0, -(right_bound - margin))
    hi = np.maximum(0.0, left_bound - margin)
    for _ in range(iterations):
        cur = pts + normals * offset[:, None]
        prv = np.roll(cur, 1, axis=0)
        nxt = np.roll(cur, -1, axis=0)
        lap = 0.5 * (prv + nxt) - cur
        delta = np.einsum('ij,ij->i', lap, normals)
        offset = offset + step_size * delta
        offset = np.clip(offset, lo, hi)
    final_pts = pts + normals * offset[:, None]
    return final_pts


def path_curvature(pts):
    nxt = np.roll(pts, -1, axis=0)
    prv = np.roll(pts, 1, axis=0)
    ds_next = np.linalg.norm(nxt - pts, axis=1)
    ds_prev = np.linalg.norm(pts - prv, axis=1)
    heading = np.arctan2(nxt[:, 1] - pts[:, 1], nxt[:, 0] - pts[:, 0])
    heading_prev = np.arctan2(pts[:, 1] - prv[:, 1], pts[:, 0] - prv[:, 0])
    dtheta = np.arctan2(np.sin(heading - heading_prev), np.cos(heading - heading_prev))
    ds = 0.5 * (ds_next + ds_prev)
    ds[ds < 1e-6] = 1e-6
    kappa = dtheta / ds
    return kappa, ds, heading


def velocity_profile(kappa, ds, v_max, a_lat_max, a_long_max, v_floor, yaw_rate_max):
    kappa_abs = np.abs(kappa) + 1e-6
    v_curve = np.sqrt(a_lat_max / kappa_abs)
    v_yaw = yaw_rate_max / kappa_abs
    v = np.minimum(np.minimum(v_curve, v_yaw), v_max)
    n = len(v)
    for _ in range(4):
        for i in range(n):
            ip = (i - 1) % n
            v[i] = min(v[i], math.sqrt(v[ip] ** 2 + 2.0 * a_long_max * ds[i]))
    for _ in range(4):
        for i in range(n - 1, -1, -1):
            inx = (i + 1) % n
            v[i] = min(v[i], math.sqrt(v[inx] ** 2 + 2.0 * a_long_max * ds[i]))
    v = np.maximum(v, v_floor)
    return v


def yaw_to_quat(yaw):
    return (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5))


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class RaceNodeComplete(Node):
    def __init__(self):
        super().__init__('race_node_complete')
        self.declare_parameter('map_yaml', '/home/wj/LIMO_GAZEBO/ws_limo_humble/maps/track_newmap.yaml')
        self.declare_parameter('world_file', '/home/wj/LIMO_GAZEBO/ws_limo_humble/src/limo_ros2/limo_car/worlds/roboracer/ifac_roboracer.world')
        self.declare_parameter('spawn_x', -0.2)
        self.declare_parameter('spawn_y', 0.80)
        self.declare_parameter('frame_id', 'odom')
        self.declare_parameter('vehicle_half_width', 0.11)
        self.declare_parameter('safety_margin', 0.05)
        self.declare_parameter('max_speed', 0.3)
        self.declare_parameter('a_lat_max', 0.5)
        self.declare_parameter('a_long_max', 0.6)
        self.declare_parameter('v_floor', 0.2)
        self.declare_parameter('path_points', 400)
        self.declare_parameter('smoothing_iterations', 400)
        self.declare_parameter('smoothing_step', 0.3)
        self.declare_parameter('lookahead_base', 0.25)
        self.declare_parameter('lookahead_gain', 0.3)
        self.declare_parameter('wheelbase', 0.24)
        self.declare_parameter('max_steer', 0.42)
        self.declare_parameter('max_steer_cmd', 0.12)
        self.declare_parameter('steer_rate_limit', 0.6)
        self.declare_parameter('yaw_rate_max', 0.18)
        self.declare_parameter('control_rate', 30.0)
        self.declare_parameter('speed_cmd_gain', 1.0)

        map_yaml = self.get_parameter('map_yaml').value
        world_file = self.get_parameter('world_file').value
        self.frame_id = self.get_parameter('frame_id').value
        margin = float(self.get_parameter('vehicle_half_width').value) + float(self.get_parameter('safety_margin').value)
        self.max_speed = float(self.get_parameter('max_speed').value)
        self.speed_cmd_gain = float(self.get_parameter('speed_cmd_gain').value)
        a_lat_max = float(self.get_parameter('a_lat_max').value)
        a_long_max = float(self.get_parameter('a_long_max').value)
        v_floor = float(self.get_parameter('v_floor').value)
        n_out = int(self.get_parameter('path_points').value)
        iters = int(self.get_parameter('smoothing_iterations').value)
        step_size = float(self.get_parameter('smoothing_step').value)
        self.ld_base = float(self.get_parameter('lookahead_base').value)
        self.ld_gain = float(self.get_parameter('lookahead_gain').value)
        control_rate = float(self.get_parameter('control_rate').value)
        self.wheelbase = float(self.get_parameter('wheelbase').value)
        max_steer_cmd = float(self.get_parameter('max_steer_cmd').value)
        self.max_curvature = math.tan(max_steer_cmd) / self.wheelbase
        self.steer_rate_limit = float(self.get_parameter('steer_rate_limit').value)
        yaw_rate_max = float(self.get_parameter('yaw_rate_max').value)
        self.prev_steer = 0.0
        self.last_control_time = None

        self.get_logger().info(f'loading map {map_yaml}')
        free_mask, occ_mask, resolution, origin = load_occupancy(map_yaml)
        walls = load_world_walls(world_file)
        self.get_logger().info(f'loaded {len(walls)} real wall boxes from world file')
        combined_mask = rasterize_walls(free_mask, resolution, origin, walls)

        raw_pts = extract_centerline(free_mask, resolution, origin)
        raw_pts = project_clear(raw_pts, combined_mask, resolution, origin)
        gaps = np.linalg.norm(np.diff(raw_pts, axis=0, append=raw_pts[0:1]), axis=1)
        raw_pts = raw_pts[gaps > 1e-4]
        center_pts = resample_closed_curve(raw_pts, n_out, smooth_factor=0.02 * len(raw_pts))
        tangent, normals = compute_normals(center_pts)
        left_b, right_b = compute_track_bounds(center_pts, normals, combined_mask, resolution, origin)
        race_pts = optimize_raceline(center_pts, normals, left_b, right_b, margin, iters, step_size)
        race_pts = resample_closed_curve(race_pts, n_out, smooth_factor=0.01 * n_out)
        race_pts = np.loadtxt('/root/path_C.csv', delimiter=',')
        kappa, ds, heading = path_curvature(race_pts)
        speed = velocity_profile(kappa, ds, self.max_speed, a_lat_max, a_long_max, v_floor, yaw_rate_max)

        self.path_xy = race_pts
        self.path_heading = heading
        self.path_speed = speed
        self.path_s = np.concatenate([[0.0], np.cumsum(ds)])[:-1]
        self.path_len = ds.sum()
        self.n_path = len(race_pts)

        min_real_clear = min(wall_clearance(walls, x, y) for x, y in race_pts)
        self.get_logger().info(
            f'path ready: {self.n_path} points, length {self.path_len:.2f} m, '
            f'max speed {speed.max():.2f} m/s, min speed {speed.min():.2f} m/s, '
            f'max |kappa|={np.abs(kappa).max():.3f}, min real-wall clearance on path={min_real_clear:.3f} m'
        )

        latched_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.path_pub = self.create_publisher(Path, '/race_path', latched_qos)
        self.publish_path()

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.on_odom, 10)

        self.have_odom = False
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.nearest_idx = 0
        self.lap_start_time = None
        self.lap_count = 0
        self.initialized = False

        self.timer = self.create_timer(1.0 / control_rate, self.control_step)

    def publish_path(self):
        msg = Path()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        for i in range(self.n_path):
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(self.path_xy[i, 0])
            ps.pose.position.y = float(self.path_xy[i, 1])
            qx, qy, qz, qw = yaw_to_quat(float(self.path_heading[i]))
            ps.pose.orientation.x = qx
            ps.pose.orientation.y = qy
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
            msg.poses.append(ps)
        msg.poses.append(msg.poses[0])
        self.path_pub.publish(msg)

    def on_odom(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = yaw_from_quat(msg.pose.pose.orientation)
        self.have_odom = True

    def find_nearest_global(self):
        d = np.hypot(self.path_xy[:, 0] - self.x, self.path_xy[:, 1] - self.y)
        return int(np.argmin(d))

    def find_nearest(self):
        window = 40
        lo = self.nearest_idx - window
        idxs = (np.arange(lo, lo + 2 * window) % self.n_path)
        d = np.hypot(self.path_xy[idxs, 0] - self.x, self.path_xy[idxs, 1] - self.y)
        best = idxs[np.argmin(d)]
        return int(best)

    def target_index(self, start_idx, lookahead):
        s0 = self.path_s[start_idx]
        target_s = (s0 + lookahead) % self.path_len
        idx = int(np.searchsorted(self.path_s, target_s) % self.n_path)
        return idx

    def min_speed_over_arc(self, start_idx, end_idx):
        if end_idx >= start_idx:
            return float(self.path_speed[start_idx:end_idx + 1].min())
        wrapped = np.concatenate([self.path_speed[start_idx:], self.path_speed[:end_idx + 1]])
        return float(wrapped.min())

    def control_step(self):
        if not self.have_odom:
            return

        if not self.initialized:
            self.nearest_idx = self.find_nearest_global()
            self.lap_start_time = time.time()
            self.initialized = True
        else:
            idx = self.find_nearest()
            if idx < self.nearest_idx - self.n_path // 2:
                now = time.time()
                self.lap_count += 1
                self.get_logger().info(f'lap {self.lap_count} time: {now - self.lap_start_time:.2f} s')
                self.lap_start_time = now
            self.nearest_idx = idx
        idx = self.nearest_idx

        speed_est = max(self.path_speed[idx], 0.1)
        lookahead = self.ld_base + self.ld_gain * speed_est
        tgt_idx = self.target_index(idx, lookahead)
        tx = self.path_xy[tgt_idx, 0]
        ty = self.path_xy[tgt_idx, 1]

        dx = tx - self.x
        dy = ty - self.y
        local_x = math.cos(-self.yaw) * dx - math.sin(-self.yaw) * dy
        local_y = math.sin(-self.yaw) * dx + math.cos(-self.yaw) * dy
        ld2 = local_x * local_x + local_y * local_y
        if ld2 < 1e-6:
            curvature = 0.0
        else:
            curvature = 2.0 * local_y / ld2
        curvature = max(-self.max_curvature, min(self.max_curvature, curvature))

        now_t = time.time()
        dt = 1.0 / 30.0 if self.last_control_time is None else max(1e-3, now_t - self.last_control_time)
        self.last_control_time = now_t
        steer_angle = math.atan(self.wheelbase * curvature)
        max_delta = self.steer_rate_limit * dt
        steer_angle = max(self.prev_steer - max_delta, min(self.prev_steer + max_delta, steer_angle))
        self.prev_steer = steer_angle
        curvature = math.tan(steer_angle) / self.wheelbase

        cmd_speed = min(self.min_speed_over_arc(idx, tgt_idx), self.max_speed)
        cmd_speed = max(cmd_speed, 0.1) * self.speed_cmd_gain
        angular_z = curvature * cmd_speed

        out = Twist()
        out.linear.x = float(cmd_speed)
        out.angular.z = float(angular_z)
        self.cmd_pub.publish(out)


def main():
    rclpy.init()
    node = RaceNodeComplete()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
