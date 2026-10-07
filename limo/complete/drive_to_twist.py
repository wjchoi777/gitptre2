#!/usr/bin/env python3
"""Convert AckermannDriveStamped (/drive) to Twist (/cmd_vel) for the unmodified
ackermann_twist_adapter.py. yaw_rate = speed * tan(steering_angle) / wheelbase is the
exact algebraic inverse of that adapter's angle = atan(wheelbase * yaw_rate / speed)."""
import math
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Twist

WHEELBASE = 0.24


class DriveToTwist(Node):
    def __init__(self):
        super().__init__('drive_to_twist')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(AckermannDriveStamped, '/drive', self.on_drive, 10)

    def on_drive(self, msg):
        speed = msg.drive.speed
        steer = msg.drive.steering_angle
        out = Twist()
        out.linear.x = speed
        if abs(speed) > 1e-6:
            out.angular.z = speed * math.tan(steer) / WHEELBASE
        self.pub.publish(out)


def main():
    rclpy.init()
    node = DriveToTwist()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
