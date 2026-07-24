#!/usr/bin/env python3
"""Verify the MC-MPC closed-loop simulator topics and truth trajectory."""

import math
import sys

import rclpy
from px4_msgs.msg import (
    ActuatorMotors,
    HoverThrustEstimate,
    RateCtrlStatus,
    TakeoffStatus,
    VehicleAngularVelocity,
    VehicleAttitudeSetpoint,
    VehicleLandDetected,
    VehicleLocalPositionSetpoint,
    VehicleOdometry,
    VehicleRatesSetpoint,
    VehicleTorqueSetpoint,
)
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy


TOPICS = (
    ("odometry", VehicleOdometry, "/fmu/out/vehicle_odometry"),
    ("local_position_setpoint", VehicleLocalPositionSetpoint,
     "/fmu/out/vehicle_local_position_setpoint"),
    ("attitude_setpoint", VehicleAttitudeSetpoint,
     "/fmu/out/vehicle_attitude_setpoint_v1"),
    ("rates_setpoint", VehicleRatesSetpoint,
     "/fmu/out/vehicle_rates_setpoint"),
    ("hover_thrust", HoverThrustEstimate, "/fmu/out/hover_thrust_estimate"),
    ("takeoff_status", TakeoffStatus, "/fmu/out/takeoff_status"),
    ("land_detected", VehicleLandDetected, "/fmu/out/vehicle_land_detected"),
    ("angular_velocity", VehicleAngularVelocity,
     "/fmu/out/vehicle_angular_velocity"),
    ("rate_ctrl_status", RateCtrlStatus, "/fmu/out/rate_ctrl_status"),
    ("torque_setpoint", VehicleTorqueSetpoint,
     "/fmu/out/vehicle_torque_setpoint"),
    ("actuator_motors", ActuatorMotors, "/fmu/out/actuator_motors"),
)


def finite(values):
    return all(math.isfinite(float(value)) for value in values)


class SimulatorVerifier(Node):
    def __init__(self):
        super().__init__("verify_mcmpc_sim")
        self.declare_parameter("duration", 8.0)
        self.declare_parameter("minimum_motion", 0.05)
        self.received = {name: 0 for name, _, _ in TOPICS}
        self.invalid = []
        self.first_position = None
        self.last_position = None
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._topic_subscriptions = []
        for name, message_type, topic in TOPICS:
            callback = self.odometry_callback if name == "odometry" else (
                lambda message, topic_name=name: self.topic_callback(
                    topic_name, message))
            self._topic_subscriptions.append(
                self.create_subscription(message_type, topic, callback, qos))
        duration = float(self.get_parameter("duration").value)
        self.timer = self.create_timer(duration, self.finish)
        self.done = False
        self.exit_code = 1
        self.get_logger().info(
            f"checking {len(TOPICS)} simulator topics for {duration:.1f} s")

    def topic_callback(self, name, _message):
        self.received[name] += 1
        message = _message
        extractors = {
            "local_position_setpoint": lambda msg: (
                msg.x, msg.y, msg.z, msg.vx, msg.vy, msg.vz,
                *msg.acceleration, msg.yaw),
            "attitude_setpoint": lambda msg: (*msg.q_d, *msg.thrust_body),
            "rates_setpoint": lambda msg: (
                msg.roll, msg.pitch, msg.yaw, *msg.thrust_body),
            "hover_thrust": lambda msg: (msg.hover_thrust,),
            "takeoff_status": lambda msg: (msg.tilt_limit,),
            "land_detected": lambda _msg: (),
            "angular_velocity": lambda msg: (
                *msg.xyz, *msg.xyz_derivative),
            "rate_ctrl_status": lambda msg: (
                msg.rollspeed_integ, msg.pitchspeed_integ,
                msg.yawspeed_integ),
            "torque_setpoint": lambda msg: tuple(msg.xyz),
            "actuator_motors": lambda msg: tuple(msg.control[:4]),
        }
        values = extractors[name](message)
        if not finite(values):
            self.invalid.append(f"{name} contains NaN/Inf")

    def odometry_callback(self, message):
        self.received["odometry"] += 1
        values = list(message.q) + list(message.angular_velocity)
        values += list(message.position) + list(message.velocity)
        if not finite(values):
            self.invalid.append("odometry contains NaN/Inf")
            return
        quaternion_norm = math.sqrt(sum(float(value) ** 2 for value in message.q))
        if abs(quaternion_norm - 1.0) > 0.02:
            self.invalid.append(
                f"quaternion norm is {quaternion_norm:.4f}, expected 1")
        position = tuple(float(value) for value in message.position)
        if self.first_position is None:
            self.first_position = position
        self.last_position = position

    def finish(self):
        if self.done:
            return
        self.done = True
        missing = [name for name, count in self.received.items() if count == 0]
        motion = 0.0
        if self.first_position is not None and self.last_position is not None:
            motion = math.dist(self.first_position, self.last_position)
        minimum_motion = float(self.get_parameter("minimum_motion").value)
        failures = []
        if missing:
            failures.append("missing topics: " + ", ".join(missing))
        if self.invalid:
            failures.extend(sorted(set(self.invalid)))
        if motion < minimum_motion:
            failures.append(
                f"truth motion {motion:.3f} m is below {minimum_motion:.3f} m")
        counts = ", ".join(
            f"{name}={count}" for name, count in self.received.items())
        self.get_logger().info("received: " + counts)
        self.get_logger().info(f"truth displacement: {motion:.3f} m")
        if failures:
            for failure in failures:
                self.get_logger().error(failure)
            self.exit_code = 1
        else:
            self.get_logger().info(
                "PASS: model truth and every MC-MPC feedback topic are active")
            self.exit_code = 0
        self.timer.cancel()


def main(args=None):
    rclpy.init(args=args)
    verifier = SimulatorVerifier()
    while rclpy.ok() and not verifier.done:
        rclpy.spin_once(verifier, timeout_sec=0.1)
    exit_code = verifier.exit_code
    verifier.destroy_node()
    rclpy.shutdown()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
