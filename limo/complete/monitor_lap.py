#!/usr/bin/env python3
"""Log odom pose, rear wheel joint velocities, and /drive commands for one
pure_pursuit attempt, and report completion/stop/divergence."""
import argparse
import csv
import math
import os
import sys
import time

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from ackermann_msgs.msg import AckermannDriveStamped

sys.path.insert(0, '/home/wj/LIMO_GAZEBO/ws_limo_humble/diagnostics')
from world_walls import load_walls, min_clearance


class LapMonitor(Node):
    def __init__(self, out_path, start_x, start_y, duration, divergence_threshold):
        super().__init__('monitor_lap')
        self.out_file = open(out_path, 'w', newline='')
        self.writer = csv.writer(self.out_file)
        self.writer.writerow(['t', 'x', 'y', 'rear_left', 'rear_right', 'drive_speed', 'drive_steer', 'wall_clear'])
        self.walls = load_walls()
        self.start_x = start_x
        self.start_y = start_y
        self.duration = duration
        self.divergence_threshold = divergence_threshold
        self.t0 = time.time()
        self.last_joint = (0.0, 0.0)
        self.last_drive = (0.0, 0.0)
        self.last_pos = None
        self.last_move_t = 0.0
        self.min_wall_clear = 1e9
        self.max_rl = 0.0
        self.max_rr = 0.0
        self.max_steer = 0.0
        self.steer_history = []
        self.left_start = False
        self.result = None
        self.result_t = None
        self.result_pos = None
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.create_subscription(JointState, '/joint_states', self.on_joint, 10)
        self.create_subscription(AckermannDriveStamped, '/drive', self.on_drive, 10)

    def on_joint(self, msg):
        try:
            rl = msg.velocity[msg.name.index('rear_left_wheel_joint')]
            rr = msg.velocity[msg.name.index('rear_right_wheel_joint')]
            self.last_joint = (rl, rr)
            self.max_rl = max(self.max_rl, abs(rl))
            self.max_rr = max(self.max_rr, abs(rr))
        except ValueError:
            pass

    def on_drive(self, msg):
        self.last_drive = (msg.drive.speed, msg.drive.steering_angle)
        self.max_steer = max(self.max_steer, abs(msg.drive.steering_angle))
        self.steer_history.append((time.time() - self.t0, msg.drive.steering_angle))

    def on_odom(self, msg):
        if self.result is not None:
            return
        t = time.time() - self.t0
        p = msg.pose.pose.position
        x, y = p.x, p.y
        clear = min_clearance(self.walls, x, y)
        self.min_wall_clear = min(self.min_wall_clear, clear)
        rl, rr = self.last_joint
        spd, steer = self.last_drive
        self.writer.writerow([f'{t:.4f}', f'{x:.5f}', f'{y:.5f}', f'{rl:.4f}', f'{rr:.4f}',
                               f'{spd:.4f}', f'{steer:.4f}', f'{clear:.4f}'])
        self.out_file.flush()

        if self.last_pos is not None:
            d = math.hypot(x - self.last_pos[0], y - self.last_pos[1])
            if d > 0.01:
                self.last_move_t = t
        self.last_pos = (x, y)

        dist_from_start = math.hypot(x - self.start_x, y - self.start_y)
        if t > 5.0 and dist_from_start > 1.5:
            self.left_start = True
        if self.left_start and t > 8.0 and dist_from_start < 0.4:
            self.result = 'completed'
            self.result_t = t
            self.result_pos = (x, y)
            return

        if max(abs(rl), abs(rr)) >= self.divergence_threshold:
            self.result = 'diverged'
            self.result_t = t
            self.result_pos = (x, y)
            return

        if t > 10.0 and (t - self.last_move_t) > 4.0:
            self.result = 'stopped'
            self.result_t = t
            self.result_pos = (x, y)
            return

        if t >= self.duration:
            self.result = 'timeout'
            self.result_t = t
            self.result_pos = (x, y)
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--start-x', type=float, required=True)
    ap.add_argument('--start-y', type=float, required=True)
    ap.add_argument('--duration', type=float, default=180.0)
    ap.add_argument('--threshold', type=float, default=50.0)
    args = ap.parse_args()

    rclpy.init()
    node = LapMonitor(args.out, args.start_x, args.start_y, args.duration, args.threshold)
    start = time.time()
    while rclpy.ok() and node.result is None and (time.time() - start) < args.duration + 5.0:
        rclpy.spin_once(node, timeout_sec=0.2)

    result = node.result or 'timeout'
    result_t = node.result_t if node.result_t is not None else (time.time() - start)
    pos = node.result_pos if node.result_pos is not None else node.last_pos
    print(f'RESULT={result}')
    print(f'result_t={result_t:.2f}')
    print(f'result_pos={pos}')
    print(f'max_rear_left={node.max_rl:.2f}')
    print(f'max_rear_right={node.max_rr:.2f}')
    print(f'min_wall_clear={node.min_wall_clear:.4f}')
    print(f'max_steer={node.max_steer:.4f}')
    if len(node.steer_history) > 1:
        rates = []
        for i in range(1, len(node.steer_history)):
            dt = node.steer_history[i][0] - node.steer_history[i-1][0]
            if dt > 1e-4:
                rates.append(abs(node.steer_history[i][1] - node.steer_history[i-1][1]) / dt)
        if rates:
            print(f'mean_steer_rate={sum(rates)/len(rates):.4f}')
            print(f'max_steer_rate={max(rates):.4f}')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
