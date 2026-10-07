import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry


class Repro(Node):
    def __init__(self):
        super().__init__('teleop_repro')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(JointState, '/joint_states', self.on_js, 10)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.v = float('nan')
        self.x = self.y = float('nan')
        self.px = self.py = None
        self.dist = 0.0
        self.idx = None
        self.t0 = None
        self.diverged = None
        self.last_print = -1
        self.create_timer(0.1, self.tick)

    def on_js(self, m):
        if self.idx is None:
            print('joints:', list(m.name))
            for i, n in enumerate(m.name):
                if 'rear' in n.lower() and 'left' in n.lower():
                    self.idx = i
                    print('rear_left joint =', n)
                    break
            if self.idx is None:
                self.idx = -1
        if self.idx >= 0 and self.idx < len(m.velocity):
            self.v = m.velocity[self.idx]

    def on_odom(self, m):
        self.x = m.pose.pose.position.x
        self.y = m.pose.pose.position.y
        if self.px is not None:
            self.dist += math.hypot(self.x - self.px, self.y - self.py)
        self.px, self.py = self.x, self.y

    def tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = now
        t = now - self.t0
        cmd = Twist()
        if t < 90.0:
            cmd.linear.x = 0.1
            cmd.angular.z = 0.3
        self.pub.publish(cmd)
        if self.diverged is None and abs(self.v) > 50.0:
            self.diverged = t
            print(f'*** DIVERGED t={t:.1f}s v={self.v:.1f} x={self.x:.2f} y={self.y:.2f} dist={self.dist:.2f}m')
        sec = int(t)
        if sec != self.last_print:
            self.last_print = sec
            print(f't={t:5.1f} rear_left_v={self.v:8.2f} x={self.x:6.2f} y={self.y:6.2f} dist={self.dist:5.2f}')
        if t >= 92.0:
            print('RESULT:', 'no divergence in 90s' if self.diverged is None else f'diverged at {self.diverged:.1f}s')
            raise SystemExit


def main():
    rclpy.init()
    n = Repro()
    try:
        rclpy.spin(n)
    except SystemExit:
        pass
    n.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
