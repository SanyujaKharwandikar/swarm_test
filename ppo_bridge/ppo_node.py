import math
import json
import sys
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped

import torch

# --- TorchRL imports for PolicyNet ---
from torchrl.modules import ProbabilisticActor, TanhNormal
from tensordict.nn import TensorDictModule
from torchrl.modules.distributions import NormalParamExtractor
from torchrl.modules.models.multiagent import MultiAgentMLP


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class PolicyNet(ProbabilisticActor):
    def _init_(self, obs_size, action_size, num_agents=2, device='cpu'):
        actor_net = torch.nn.Sequential(
            MultiAgentMLP(
                n_agent_inputs=obs_size,
                n_agent_outputs=2 * action_size,
                n_agents=num_agents,
                centralised=False,
                share_params=True,
                device=device,
                depth=2,
                num_cells=256,
                activation_class=torch.nn.Tanh,
            ),
            NormalParamExtractor(),
        )

        policy_module = TensorDictModule(
            module=actor_net,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "loc"), ("agents", "scale")],
        )

        super()._init_(
            module=policy_module,
            in_keys=[("agents", "loc"), ("agents", "scale")],
            out_keys=[("agents", "action")],
            distribution_class=TanhNormal,
            return_log_prob=False,
            spec=None
        )


class MultiAgentPPONode(Node):
    def _init_(self) -> None:
        super()._init_('multi_agent_ppo_node')

        # Parameters
        self.declare_parameter('policy_path', '/home/vimal-pranav/Downloads/multi_agent_policy.pth')
        self.declare_parameter('robot_namespaces', ['tb1', 'tb2'])
        default_goals_str = json.dumps([[1.0, 0.0], [-1.0, 0.0]])
        self.declare_parameter('goal_positions', default_goals_str)
        self.declare_parameter('lidar_topic', 'scan')
        self.declare_parameter('control_rate_hz', 10.0)
        self.declare_parameter('v_max', 0.26)
        self.declare_parameter('w_max', 1.82)
        self.declare_parameter('device', 'cpu')

        policy_path = self.get_parameter('policy_path').get_parameter_value().string_value
        self.namespaces: List[str] = [str(s) for s in self.get_parameter('robot_namespaces').value]
        goals_str = self.get_parameter('goal_positions').get_parameter_value().string_value
        try:
            self.goal_positions: List[List[float]] = json.loads(goals_str)
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse goal_positions JSON: {e}")
            raise RuntimeError("Invalid goal_positions parameter format.")

        self.lidar_topic: str = self.get_parameter('lidar_topic').get_parameter_value().string_value
        self.control_rate_hz: float = float(self.get_parameter('control_rate_hz').value)
        self.v_max: float = float(self.get_parameter('v_max').value)
        self.w_max: float = float(self.get_parameter('w_max').value)
        self.device: str = self.get_parameter('device').get_parameter_value().string_value

        if not policy_path:
            self.get_logger().error('Parameter policy_path is required (TorchScript .pt/.ts recommended).')
            raise RuntimeError('Missing policy_path')

        if len(self.goal_positions) != len(self.namespaces):
            self.get_logger().warn(
                'goal_positions length != robot_namespaces length. Will repeat/trim goals as needed.'
            )

        self.num_agents = len(self.namespaces)
        self.odom: Dict[str, Odometry] = {ns: None for ns in self.namespaces}
        self.scan: Dict[str, LaserScan] = {ns: None for ns in self.namespaces}
        self.cmd_pubs: Dict[str, rclpy.publisher.Publisher] = {}

        # Load policy
        self.policy = self._load_policy(policy_path, self.device)
        self.policy.eval()

        # QoS
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.history = HistoryPolicy.KEEP_LAST

        # Subscriptions and publishers
        for ns in self.namespaces:
            self.create_subscription(Odometry, f'/{ns}/odom', lambda msg, ns=ns: self._odom_cb(ns, msg), qos)
            if self.lidar_topic:
                self.create_subscription(LaserScan, f'/{ns}/{self.lidar_topic}', lambda msg, ns=ns: self._scan_cb(ns, msg), qos)
            self.cmd_pubs[ns] = self.create_publisher(TwistStamped, f'/{ns}/cmd_vel', 10)

        self.timer = self.create_timer(1.0 / self.control_rate_hz, self._control_step)
        self.get_logger().info(f"Loaded policy from {policy_path} for {self.num_agents} agents: {self.namespaces} (device={self.device})")

    def _odom_cb(self, ns: str, msg: Odometry) -> None:
        self.odom[ns] = msg

    def _scan_cb(self, ns: str, msg: LaserScan) -> None:
        self.scan[ns] = msg

    def _load_policy(self, path: str, device: str):
        try:
            policy = torch.jit.load(path, map_location=device)
            self.get_logger().info('Loaded TorchScript policy.')
            return policy
        except Exception as e_js:
            self.get_logger().warn(f'TorchScript load failed: {e_js}. Trying torch.load...')
            obj = torch.load(path, map_location=device)

            if hasattr(obj, 'eval') and callable(getattr(obj, 'eval')):
                self.get_logger().info('Loaded pickled PyTorch policy object.')
                return obj
            elif isinstance(obj, dict):
                obs_size = 18
                action_size = 2
                num_agents = self.num_agents
                policy = PolicyNet(obs_size, action_size, num_agents, device=device)
                if 'model_state_dict' in obj:
                    policy.load_state_dict(obj['model_state_dict'])
                else:
                    policy.load_state_dict(obj)
                policy.to(device)
                self.get_logger().info('Loaded PolicyNet from state_dict.')
                return policy
            else:
                self.get_logger().error('Unsupported policy format.')
                raise RuntimeError('Unsupported policy format')

    def _control_step(self) -> None:
        if any(self.odom[ns] is None for ns in self.namespaces):
            return

        obs = []
        poses: Dict[str, Tuple[float, float, float]] = {}
        vels: Dict[str, Tuple[float, float, float]] = {}
        for ns in self.namespaces:
            od = self.odom[ns]
            px = od.pose.pose.position.x
            py = od.pose.pose.position.y
            qx = od.pose.pose.orientation.x
            qy = od.pose.pose.orientation.y
            qz = od.pose.pose.orientation.z
            qw = od.pose.pose.orientation.w
            yaw = quat_to_yaw(qx, qy, qz, qw)
            vx = od.twist.twist.linear.x
            vy = od.twist.twist.linear.y
            wz = od.twist.twist.angular.z
            poses[ns] = (px, py, yaw)
            vels[ns] = (vx, vy, wz)

        for i, ns in enumerate(self.namespaces):
            px, py, yaw = poses[ns]
            vx, vy, wz = vels[ns]

            goal = self.goal_positions[i % len(self.goal_positions)]
            gx, gy = float(goal[0]), float(goal[1])
            dx = gx - px
            dy = gy - py
            goal_dist = math.hypot(dx, dy)

            other_rel_pos = (0.0, 0.0)
            other_rel_vel = (0.0, 0.0)
            for ns2 in self.namespaces:
                if ns2 == ns:
                    continue
                px2, py2, _ = poses[ns2]
                vx2, vy2, _ = vels[ns2]
                other_rel_pos = (px2 - px, py2 - py)
                other_rel_vel = (vx2 - vx, vy2 - vy)
                break

            lidar_vals = [1.0] * 5
            if self.lidar_topic and self.scan[ns] is not None:
                sc: LaserScan = self.scan[ns]
                num = len(sc.ranges)
                if num > 0 and sc.angle_increment != 0.0:
                    def idx_for_angle(angle_rad: float) -> int:
                        angle = clip(angle_rad, sc.angle_min, sc.angle_max)
                        return int((angle - sc.angle_min) / sc.angle_increment)
                    sample_angles = [-math.pi/2, -math.pi/4, 0.0, math.pi/4, math.pi/2]
                    for k, a in enumerate(sample_angles):
                        idx = clip(idx_for_angle(a), 0, num - 1)
                        rng = sc.ranges[int(idx)]
                        if math.isfinite(rng):
                            denom = sc.range_max if sc.range_max > 0.0 else 1.0
                            lidar_vals[k] = clip(rng / denom, 0.0, 1.0)
                        else:
                            lidar_vals[k] = 1.0

            obs_i = [
                px, py, vx, vy, yaw, wz,
                dx, dy, goal_dist,
                other_rel_pos[0], other_rel_pos[1],
                other_rel_vel[0], other_rel_vel[1],
                *lidar_vals
            ]
            if len(obs_i) != 18:
                self.get_logger().error(f'Observation length is {len(obs_i)}, expected 18.')
                return
            obs.append(obs_i)

        obs_tensor = torch.tensor([obs], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            actions = self.policy(obs_tensor)
        if isinstance(actions, (list, tuple)):
            actions = actions[0]
        if torch.is_tensor(actions):
            act = actions.squeeze(0).detach().cpu().numpy()
        else:
            self.get_logger().error('Policy output is not a tensor.')
            return

        for i, ns in enumerate(self.namespaces):
            a_lin = float(clip(act[i][0], -1.0, 1.0))
            a_ang = float(clip(act[i][1], -1.0, 1.0))
            v = a_lin * self.v_max
            w = a_ang * self.w_max
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            msg.twist.linear.x = v
            msg.twist.angular.z = w
            self.cmd_pubs[ns].publish(msg)


def main(argv=None):
    rclpy.init(args=argv)
    try:
        node = MultiAgentPPONode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if _name_ == '_main_':
    main()
