#!/usr/bin/env python3
"""Build the same global raceline race_node_complete uses and publish it on /pp_path
for the GIU-F1Tenth pure_pursuit controller (velocity encoded in orientation.w)."""
import os
import sys
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from race_node_complete import (
    load_occupancy, load_world_walls, rasterize_walls, extract_centerline,
    project_clear, resample_closed_curve, compute_normals, compute_track_bounds,
    optimize_raceline, path_curvature, velocity_profile, wall_clearance,
)

MAP_YAML = '/home/wj/LIMO_GAZEBO/ws_limo_humble/maps/track_newmap.yaml'
WORLD_FILE = '/home/wj/LIMO_GAZEBO/ws_limo_humble/src/limo_ros2/limo_car/worlds/roboracer/ifac_roboracer.world'
VEHICLE_HALF_WIDTH = 0.11
SAFETY_MARGIN = 0.05
MAX_SPEED = 0.3
A_LAT_MAX = 0.5
A_LONG_MAX = 0.6
V_FLOOR = 0.2
PATH_POINTS = 400
SMOOTHING_ITERATIONS = 400
SMOOTHING_STEP = 0.3
YAW_RATE_MAX = 0.18


def build_path():
    margin = VEHICLE_HALF_WIDTH + SAFETY_MARGIN
    free_mask, occ_mask, resolution, origin = load_occupancy(MAP_YAML)
    walls = load_world_walls(WORLD_FILE)
    combined_mask = rasterize_walls(free_mask, resolution, origin, walls)

    raw_pts = extract_centerline(free_mask, resolution, origin)
    raw_pts = project_clear(raw_pts, combined_mask, resolution, origin)
    gaps = np.linalg.norm(np.diff(raw_pts, axis=0, append=raw_pts[0:1]), axis=1)
    raw_pts = raw_pts[gaps > 1e-4]
    center_pts = resample_closed_curve(raw_pts, PATH_POINTS, smooth_factor=0.02 * len(raw_pts))
    tangent, normals = compute_normals(center_pts)
    left_b, right_b = compute_track_bounds(center_pts, normals, combined_mask, resolution, origin)
    race_pts = optimize_raceline(center_pts, normals, left_b, right_b, margin, SMOOTHING_ITERATIONS, SMOOTHING_STEP)
    race_pts = resample_closed_curve(race_pts, PATH_POINTS, smooth_factor=0.01 * PATH_POINTS)
    kappa, ds, heading = path_curvature(race_pts)
    speed = velocity_profile(kappa, ds, MAX_SPEED, A_LAT_MAX, A_LONG_MAX, V_FLOOR, YAW_RATE_MAX)

    min_real_clear = min(wall_clearance(walls, x, y) for x, y in race_pts)
    return race_pts, speed, min_real_clear


class PPPathPublisher(Node):
    def __init__(self):
        super().__init__('pp_path_publisher')
        self.declare_parameter('frame_id', 'odom')
        self.declare_parameter('v_min', 0.2)
        self.declare_parameter('v_max', 0.3)
        frame_id = self.get_parameter('frame_id').value
        v_min = float(self.get_parameter('v_min').value)
        v_max = float(self.get_parameter('v_max').value)

        race_pts, speed, min_real_clear = build_path()
        self.get_logger().info(
            f'built path: {len(race_pts)} pts, raw speed range '
            f'[{speed.min():.2f},{speed.max():.2f}] m/s, min real-wall clearance={min_real_clear:.3f} m'
        )

        s_min, s_max = speed.min(), speed.max()
        if s_max > s_min:
            v_scaled = v_min + (speed - s_min) / (s_max - s_min) * (v_max - v_min)
        else:
            v_scaled = np.full_like(speed, v_min)
        self.get_logger().info(f'rescaled speed to pp_path range [{v_scaled.min():.2f},{v_scaled.max():.2f}] m/s')

        latched_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.pub = self.create_publisher(Path, '/pp_path', latched_qos)

        msg = Path()
        msg.header.frame_id = frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        for i in range(len(race_pts)):
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(race_pts[i, 0])
            ps.pose.position.y = float(race_pts[i, 1])
            ps.pose.orientation.w = float(v_scaled[i])
            msg.poses.append(ps)
        msg.poses.append(msg.poses[0])
        self.pub.publish(msg)
        self.timer = self.create_timer(1.0, self.republish)
        self.msg = msg

    def republish(self):
        self.pub.publish(self.msg)


def main():
    rclpy.init()
    node = PPPathPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
