#!/usr/bin/env python3
"""ROS 2 node wrapping the tested LineController (camera geometry from your URDF)."""
import time

import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from line_follower import vision
from line_follower.control import DEFAULTS, LineController


class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower')
        d = ParameterDescriptor(dynamic_typing=True)     # accept int or float in YAML

        def p(name, default):
            return self.declare_parameter(name, default, d).value

        self.image_topic = str(p('image_topic', '/camera/image_raw'))
        self.cmd_topic = str(p('cmd_vel_topic', '/cmd_vel'))
        self.stamped = bool(p('use_stamped', False))
        self.img_timeout = float(p('image_timeout', 1.0))      # wall-clock seconds
        self.debug = bool(p('debug', True))
        # camera geometry (must match the URDF): height above ground, x ahead of the
        # wheel axle, pitch down (rad), horizontal field of view (rad)
        self.cam_h = float(p('cam_height', 0.10))
        self.cam_x = float(p('cam_x', 0.11))
        self.cam_pitch = float(p('cam_pitch', 0.35))
        self.cam_hfov = float(p('cam_hfov', 1.3962634))
        self.stride = int(p('image_stride', 2))          # 640x480 -> 320x240
        params = {}
        for k, v in DEFAULTS.items():
            val = p(k, v)
            params[k] = str(val) if isinstance(v, str) else float(val)
        self.ctl = LineController(**params)
        self.model = None
        self.model_size = None

        self.pub = self.create_publisher(TwistStamped if self.stamped else Twist, self.cmd_topic, 10)
        self.create_subscription(Image, self.image_topic, self.on_image, qos_profile_sensor_data)
        self.create_timer(0.1, self.watchdog, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.last_img = time.monotonic()
        self.last_log = 0.0
        self.t_fallback = 0.0
        self.get_logger().info(f'line_follower: {self.image_topic} -> {self.cmd_topic} (stamped={self.stamped})')

    def send(self, v, w):
        if self.stamped:
            m = TwistStamped()
            m.header.stamp = self.get_clock().now().to_msg()
            m.twist.linear.x, m.twist.angular.z = float(v), float(w)
        else:
            m = Twist()
            m.linear.x, m.angular.z = float(v), float(w)
        self.pub.publish(m)

    def watchdog(self):
        if time.monotonic() - self.last_img > self.img_timeout:
            self.send(0.0, 0.0)

    def on_image(self, msg):
        self.last_img = time.monotonic()
        # control time = camera timestamp (works with or without /clock being bridged)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if t <= 0.0:
            self.t_fallback += 1.0 / 30.0
            t = self.t_fallback
        gray = vision.to_gray(msg.data, msg.height, msg.width, msg.step, msg.encoding, self.stride)
        if gray is None:
            self.get_logger().error(f'Unsupported image encoding: {msg.encoding}',
                                    throttle_duration_sec=5.0)
            self.send(0.0, 0.0)
            return
        h, w = gray.shape
        if self.model is None or self.model_size != (w, h):
            self.model = vision.CamModel(w, h, self.cam_hfov, self.cam_h, self.cam_x, self.cam_pitch)
            self.model_size = (w, h)
        v, wz = self.ctl.update(gray, t, self.model)
        self.send(v, wz)
        if self.debug and time.monotonic() - self.last_log > 1.0:
                self.last_log = time.monotonic()
                fit = self.ctl.fit or (0.0, 0.0, 0.0)
                self.get_logger().info(
                    f'state={self.ctl.state:6s} y0={fit[0]*100:+5.1f}cm psi={fit[1]*57.3:+5.1f}deg '
                    f'curv={fit[2]:+4.2f} cmd v={v:.2f} w={wz:+.2f}')
def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.send(0.0, 0.0)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
