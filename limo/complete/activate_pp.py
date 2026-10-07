#!/usr/bin/env python3
"""Keep publishing the activation string pure_pursuit_node needs on /control_selector."""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class Activator(Node):
    def __init__(self):
        super().__init__('activate_pp')
        self.pub = self.create_publisher(String, '/control_selector', 10)
        self.create_timer(0.5, self.tick)

    def tick(self):
        self.pub.publish(String(data='pure_pursuit'))


def main():
    rclpy.init()
    node = Activator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
