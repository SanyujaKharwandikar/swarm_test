# RESEARCH (ROBOTICS CLUB MULTI-DRONE PROJECT)

## Multi-Drone Mapping Overview

**Collaborative SLAM (C-SLAM)** is the foundation of multi-drone systems, allowing multiple robots to explore and build a shared map of an environment accurately and efficiently.

### Best SLAM System for the Project: SLAM Toolbox (ROS 2)

* Best for: Simplicity, real-time mapping, modularity
* Works well in ROS 2 (officially supported)
* Supports map serialization, pose graph optimization, online/offline modes
* Lightweight and ideal for TurtleBot3
* Each robot runs SLAM Toolbox locally
* MARL agents can explore and periodically merge maps into a central/global one

---

## Map Merging in Multi-Robot Systems

In C-SLAM, each robot builds a local map which is later merged into a global map.

### A. Occupancy Grid Stitching

Each robot maintains its own `nav_msgs/OccupancyGrid`.
Map merging involves:

* Aligning maps based on estimated relative transforms (from TF, GPS, or known landmarks)
* Transforming the occupancy grids into a global frame
* Combining them by overlaying or averaging occupancy values

**Example Tools:**

* `multirobot_map_merge`
* Custom node using `tf2` + occupancy grid math

**Needs:**

* Consistent TF tree (e.g., `/robot_1/map → /robot_1/odom → /robot_1/base_link`)
* Good pose estimation or initial pose sharing between robots

---

### B. Pose Graph Merging

Each robot builds a pose graph SLAM (keyframes + constraints).
Map merging involves:

* Sharing pose graphs
* Identifying loop closures across robots
* Running global graph optimization (e.g., g2o, GTSAM)

---

### C. Landmark-Based Matching

Robots detect common features (visual landmarks or LIDAR signatures) and use them to align maps.
**Useful when:**

* Robots don’t share odometry or TF
* Vision-based SLAM is used

---

### D. Cloud-Based or Server Merging

Robots send map data to a central server which:

* Aligns submaps
* Optimizes poses
* Returns global map to each robot

**Implementation Steps:**

* Run SLAM Toolbox on each robot
* Use a merging node (or script) to combine `/map` topics using transforms
* Set initial relative positions manually or use TF broadcasters to simulate known origin offsets

---

## Common Test Environments

For collective mapping and MARL projects, testing typically occurs in three environment types:

### A. Structured Environments

Examples: Offices, labs, warehouses, corridors
**Reason:**

* Easier for debugging and baseline tests
* Clearly defined walls and known layouts
* Better SLAM performance with fewer dynamic obstacles

### B. Semi-Structured Environments

Examples: Houses with furniture, halls with obstacles
**Reason:**

* Realistic indoor scenarios
* Some clutter and dynamic elements

### C. Unstructured Environments

Examples: Outdoors, forests, disaster zones
**Reason:**

* Suitable for advanced systems with robust SLAM + sensor fusion
* Higher uncertainty: uneven terrain, dynamic lighting, moving objects

---

## MARL Algorithms Used in Navigation and Mapping

| Algorithm                          | Type                | Description                                              | Use in Robotics                          |
| ---------------------------------- | ------------------- | -------------------------------------------------------- | ---------------------------------------- |
| MADDPG (Multi-Agent DDPG)          | Actor-Critic        | Centralized training with decentralized execution (CTDE) | Cooperative exploration and coordination |
| MAPPO (Multi-Agent PPO)            | Actor-Critic        | Stable, scalable PPO for multiple agents                 | Exploration and area coverage            |
| QMIX                               | Value-based         | Learns value decomposition across agents                 | Team navigation and patrol               |
| VDN (Value Decomposition Networks) | Value-based         | Simpler than QMIX, no mixing network                     | Cooperative reward optimization          |
| DGN (Dynamic Graph Networks)       | GNN-based           | Models inter-agent communication                         | Navigation with dynamic role-switching   |
| CommNet / TarMAC                   | Communication-based | Agents learn to share state information                  | Task/goal-based navigation               |

**Example Studies:**

* Cooperative Deep Reinforcement Learning for Multi-Agent Mapping and Exploration → uses MADDPG
* Graph-based MARL for Swarm Navigation → uses DGN
* MAPPO in Multi-Agent Path Finding → performs well in dense environments

---

## Centralized vs Decentralized Training

| Feature             | Centralized Training              | Decentralized Training           |
| ------------------- | --------------------------------- | -------------------------------- |
| Coordination        | Strong                            | Weak                             |
| Learning Stability  | Higher                            | Lower (non-stationarity)         |
| Scalability         | Medium (depends on communication) | High (lightweight)               |
| Real-time Execution | Bottlenecked                      | Scalable                         |
| Best For            | CTDE setups (MADDPG, MAPPO)       | Simple teams, weak communication |

Most practical systems use **Centralized Training → Decentralized Execution (CTDE)**, forming the foundation of MADDPG, MAPPO, etc.

---

## Simulation Frameworks

| Framework                    | Use Case                   | ROS Support                               | Notes                                                   |
| ---------------------------- | -------------------------- | ----------------------------------------- | ------------------------------------------------------- |
| Gazebo (Classic or Ignition) | 3D robotics simulation     | Full ROS 1/2                              | Widely used with TurtleBot3, Nav2, slam_toolbox         |
| stage_ros                    | 2D lightweight sim         | ROS 1 only (unofficial ROS 2 forks exist) | Efficient for swarm MARL, good for exploration research |
| gym-gazebo2                  | MARL + RL interface        | ROS 2 + OpenAI Gym                        | Ideal for RL training, limited maintenance              |
| Webots                       | Visual + physics           | ROS 2 driver available                    | Better GUI than Gazebo                                  |
| Unity ML + ROS               | Realistic sim + perception | ROS bridge via ros_unity                  | Used in vision-based MARL                               |

---

## SLAM Techniques Best Supported

| SLAM Package | ROS 2 Support | Features                                   | Multi-Robot | Recommended For                   |
| ------------ | ------------- | ------------------------------------------ | ----------- | --------------------------------- |
| slam_toolbox | Excellent     | Real-time + offline modes (manual merging) | Partial     | TurtleBot3/4 Nav2 + Gazebo        |
| Cartographer | Moderate      | 2D/3D SLAM + loop closure                  | Partial     | Advanced mapping, higher accuracy |
| Gmapping     | ROS 1 only    | Basic 2D SLAM                              | —           | Legacy TB3 setups                 |

---

## Multi-Robot Launch and Coordination

| Component                | Built-in Support      | Custom Work Needed                                       |
| ------------------------ | --------------------- | -------------------------------------------------------- |
| Multi-TurtleBot Spawning | Only 1 TB3 by default | Launch files with namespace + tf_prefix                  |
| Nav2 Multi-Robot         | Supported officially  | Requires namespace and topic remapping                   |
| SLAM Multi-Robot         | No built-in support   | Manual map merging or separate instances                 |
| TF Tree Separation       | Not automatic         | Use `PushRosNamespace` and `GroupAction` in launch files |

You’ll need custom launch files and namespaces to simulate 2+ TurtleBots in Gazebo with independent navigation and SLAM.

---

## Sensor Types

| Sensor   | Description                                   |
| -------- | --------------------------------------------- |
| Lidar    | 360° scanning for SLAM and obstacle detection |
| IMU      | Measures acceleration and orientation         |
| Odometry | Derived from wheel encoders                   |
| Camera   | Used for visual SLAM or object detection      |

---

## Incorporating Reinforcement Learning in Multi-Robot Systems

### What is MARL and Why Use It?

Multi-Agent Reinforcement Learning (MARL) enables multiple robots (agents) to learn collaborative behaviors in a shared environment.
In mapping, MARL helps robots collectively explore, share information, and efficiently navigate to goals.

### Steps to Incorporate RL

1. **Define the Problem**

   * State Space: Robot’s own position, sensor data, partial map
   * Action Space: Movement or mapping actions
   * Reward Function: Area mapped, collisions avoided, path efficiency

2. **Choose an RL Algorithm**

   * MADDPG, DQN, PPO are popular options
   * Algorithms may be centralized (shared critic) or decentralized

3. **Simulation Environment**

   * Use Gazebo or OpenAI Gym for pre-deployment training

4. **Training and Testing**

   * Train in simulation, transfer to real robots, adjust reward functions as needed

---

## Reinforcement Learning Core Concepts

### Q-Learning Update Rule

[
Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]
]

* ( Q(s,a) ): Current Q-value
* ( \alpha ): Learning rate
* ( r ): Reward
* ( \gamma ): Discount factor
* ( s' ): Next state
* ( \max_{a'} Q(s', a') ): Best future Q-value

### Bellman Optimality Equation

[
Q^*(s, a) = \mathbb{E}*{s'} [r + \gamma \max*{a'} Q^*(s', a')]
]

### DQN Loss Function

[
\mathcal{L}(\theta) = [r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta)]^2
]

Where:

* ( \theta ): Parameters of the Q-network
* ( \theta^- ): Target network parameters

---

## Multi-Agent Q-Learning Variants

* **Independent Q-Learning (IQL):** Each agent learns individually; may face non-stationarity.
* **Centralized Training with Decentralized Execution (CTDE):** Shared training information; decentralized deployment.

This is the basis of algorithms like **MADDPG** and **MAPPO**.
