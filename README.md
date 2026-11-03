# Jackal: find a car, identify its side, approach and face its rear

A Clearpath Jackal (ROS 2 Jazzy, Gazebo) scouts clockwise until a car fills the frame, works out
which end is the rear, measures the car with the depth camera, plans an A* path to a spot behind
the rear bumper, drives there, turns to face the car and backs up a little for a clear view.

The script is `jackal_mirror_rear_astar.py`. Your run command calls it `cv_detection.py`, so
either copy it to that name or change the command:

```bash
cp jackal_mirror_rear_astar.py ~/Downloads/cv_detection.py
```

---

## How it works

```
 1 SCOUT        rotate clockwise only, in small yaw steps, running YOLO on every fresh frame
      |
 2 FRAME-FILL   first car box that fits entirely inside the image (2 px margin) -> HOLD, robot stops
      |
 3 IDENTIFY     once the robot is stationary, a CFV orientation model classifies the car crop
      |         and its mirror image. The result must be confident, self-consistent and a
      |         side/diagonal view. Then:
      |            mirror RIGHT -> rear is on the image/robot LEFT  (robot will go left)
      |            mirror LEFT  -> rear is on the image/robot RIGHT (robot will go right)
      |
 4 DEPTH        median of a few depth frames; the car surface inside the box is projected onto
      |         the ground plane and a RANSAC line fit finds the long side. This gives:
      |            distance from camera, car centre, orientation, length
      |
 5 REAR POINT   of the two car ends, the one on the CFV image side is the rear
      |
 6 PLAN         A* in the odom frame around the car's box (inflated 0.65 m) to a goal 1.0 m
      |         behind the rear bumper, facing the car
      |
 7 DRIVE        turn / drive along the waypoints using odometry, final align to face the car,
                then reverse 0.5 m for a better view
```

No Gazebo ground-truth poses are used. The car's pose comes from the depth camera, and the robot's
pose from `/j100_0000/platform/odom`.

---

## Start-up (one terminal per step)

Run the steps in this order. Wait for each to finish before starting the next.

### 1. Launch the world

```bash
source /opt/ros/jazzy/setup.bash
export GZ_SIM_RESOURCE_PATH="$HOME/jackal_sim/models${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
gz sim -r ~/jackal_sim/worlds/jackal_lab1.sdf
```

### 2. Spawn the Jackal and camera

```bash
source /opt/ros/jazzy/setup.bash
ros2 launch clearpath_gz robot_spawn.launch.py \
  world:=jackal_lab \
  setup_path:="$HOME/clearpath" \
  x:=0.0 y:=0.0 z:=0.3 yaw:=0.0
```

Give it 10 to 20 seconds after the robot appears so the controllers can start.

### 3. Clock bridge (required)

The script uses simulation time and waits for `/clock`. Without this it never starts moving.

```bash
source /opt/ros/jazzy/setup.bash
ros2 run ros_gz_bridge parameter_bridge \
  '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'
```

### 4. Spawn the Prius

```bash
source /opt/ros/jazzy/setup.bash
ros2 launch ros_gz_sim gz_spawn_model.launch.py \
  world:=jackal_lab \
  file:="$HOME/jackal_sim/models/prius_hybrid/prius_static.sdf" \
  entity_name:=prius_01 \
  x:=5.0 y:=2.0 z:=0.1 yaw:=0.0
```

### 5. Check the sensors (first run, or after any change to the robot setup)

```bash
source /opt/ros/jazzy/setup.bash
ros2 topic list | grep camera_0
ros2 topic hz /j100_0000/sensors/camera_0/depth/image
ros2 topic echo /j100_0000/sensors/camera_0/color/camera_info --once
```

The script expects these topics by default:

| Purpose | Default topic |
|---|---|
| Colour image | `/j100_0000/sensors/camera_0/color/image` |
| Depth image | `/j100_0000/sensors/camera_0/depth/image` |
| Camera info | `/j100_0000/sensors/camera_0/color/camera_info` |
| Odometry | `/j100_0000/platform/odom` |
| Velocity out | `/j100_0000/platform/cmd_vel` |

The depth and camera-info names are inferred from the colour topic's naming. If yours differ, pass
`--depth-topic` and `--camera-info-topic` (see Options). The depth image must be aligned with the
colour image, as a Gazebo RGB-D camera normally is.

Optional viewers and checks:

```bash
# camera viewer (the script also publishes its annotated view on /clockwise_scan/image)
ros2 run rqt_image_view rqt_image_view

# odometry
ros2 topic echo /j100_0000/platform/odom --once \
  --qos-reliability reliable --qos-durability transient_local
```

### 6. Teleop (optional, for positioning only)

Use it to place the Jackal somewhere sensible before the run, then **close it**.

```bash
source /opt/ros/jazzy/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args \
  -p stamped:=true \
  -p frame_id:=base_link \
  -p speed:=0.2 \
  -p turn:=0.4 \
  -r cmd_vel:=/j100_0000/platform/cmd_vel
```

Do not leave teleop open while the script runs. Both publish to the same command topic and will
fight each other.

### 7. Run the detection and approach

Source ROS first, then activate the virtual environment.

```bash
source /opt/ros/jazzy/setup.bash
source ~/jackal_sim/smolvlm_venv/bin/activate

python -u ~/Downloads/cv_detection.py \
  --execute \
  --speed 0.05 \
  --cfv-confidence 0.70 \
  --confirm-frames 1
```

`--front-axis-deg -90` is no longer an option. The car's axis now comes from the depth data, so
drop it from your command.

Press `q` in the OpenCV window or Ctrl+C to stop. The robot is told to stop on exit.

**Preview without moving:** leave out `--execute`. The script then publishes no commands, so the
robot will not scout. Use it with the car already in view to check the detection, depth estimate and
planned path.

---

## What you should see

| Console message | Meaning |
|---|---|
| `EXECUTION ENABLED` | Rotation is allowed |
| `CLOCKWISE \| yaw=... target=...` | Scouting |
| `HOLD latched: full_bbox=True` | A car fits entirely in the frame; the robot stops |
| `Raw CFV: ... \| mirror: ...` and `Rear likely IMAGE LEFT/RIGHT` | Side identified |
| `Side confirmation: 1/1` then `IDENTIFIED -> depth estimate -> A* PLANNING` | Hand-off to depth and planning |
| `CAR DISTANCE FROM CAMERA: nearest ... m, median ... m` | Depth-based range |
| `Car in robot frame: centre=(... fwd, ... left) m, length=...` | Measured car geometry |
| `goal is X m to the robot's LEFT/RIGHT and Y m ahead` | Direction chosen from the mirror result |
| `Waypoint n: (x, y)` | The A* path |
| `Aligned with the car rear; backing up 0.50 m` | Final step |
| `Jackal is ... m behind the rear bumper ... facing the car rear.` | Done |

A top-down image of the depth points, car box, path and goal is written to
`~/jackal_sim/rear_plan_debug.png`. Check it before trusting a new setup. The depth points (blue)
should lie along the car's near side, and the goal (orange arrow) should sit behind the end marked
by the red dot, pointing at the car.

---

## Options

| Option | Default | Notes |
|---|---|---|
| `--execute` | off | Allow motion. Without it nothing is published |
| `--speed` | 0.05 | Scouting rotation speed, rad/s (max 0.3) |
| `--yaw-step-deg` | 10 | Size of each clockwise yaw goal (2 to 45) |
| `--cfv-confidence` | 0.70 | Minimum CFV score for both the original and mirrored crop |
| `--confirm-frames` | 1 | Consecutive identical side results required |
| `--margin` | 2 | Pixels the car box must stay inside the image edges |
| `--weights` | `~/jackal_sim/yolo11n.pt` | YOLO weights (must have a `car` class) |
| `--depth-topic` | `/j100_0000/sensors/camera_0/depth/image` | 32FC1 metres or 16UC1 millimetres |
| `--camera-info-topic` | `/j100_0000/sensors/camera_0/color/camera_info` | Camera intrinsics |
| `--hfov-deg` | none | Fallback horizontal FOV if there is no camera info |
| `--depth-frames` | 3 | Depth frames median-stacked for the estimate |
| `--depth-timeout` | 6 | Seconds to wait for depth frames |
| `--backup-distance` | 0.5 | Metres to reverse after aligning; 0 disables (max 2) |
| `--cmd-topic` | `/j100_0000/platform/cmd_vel` | A `TwistStamped` topic. Use a `twist_mux` input to let the mux arbitrate |
| `--debug-image` | `~/jackal_sim/rear_plan_debug.png` | Empty string disables |
| `--stale-seconds` | 2 | Maximum age of camera data |

Fixed in code: goal 1.0 m behind the rear bumper (`REAR_CLEARANCE`), planning margin 0.65 m
(`INFLATION`), hard-stop margin 0.40 m while driving (`TRACK_INFLATION`), nominal car size
4.5 m x 1.8 m.

---

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| Stays at `WAIT - fresh camera, odometry and /clock required` | The clock bridge (step 3) is not running, or camera or odometry is stale |
| `No camera intrinsics` | Wrong `--camera-info-topic`. Run `ros2 topic list \| grep camera_0`, or pass `--hfov-deg` |
| `No fresh depth frames on ...` | Wrong `--depth-topic`, or the depth stream is not publishing |
| `End-on view: both car ends project to the same image column` | The car is seen nearly from the front or back. Teleop to a position that shows its side, then restart |
| `No car-length side face found in depth` | The side is too grazing or partly hidden, or the car is beyond the depth range. Check the depth view in the debug image |
| `DEPTH ESTIMATE FAILED - not moving` | Any of the above. The robot does not move after this |
| `Tracking departed from clear route; stopped` | The robot drifted into the car's safety margin. Restart; the plan is recomputed |
| `Missing controller subscriber, odometry or simulation clock` | Steps 2 or 3 are not fully up |
| Robot ignores commands | See the next section |

### Robot does not move

1. Make sure teleop is closed and no old copy of the script is running:
   `pgrep -af "cv_detection|jackal_mirror_rear_astar|teleop_twist_keyboard"`.
2. Run `bash cmd_vel_diagnose.sh` with the sim up. It lists who publishes to the command topic, the
   `twist_mux` input topics, the controller state, `/clock` and the odometry velocity.
3. On Clearpath robots `twist_mux` publishes `/j100_0000/platform/cmd_vel`. If commands sent
   directly to that topic are overridden, send them to a `twist_mux` input of type `TwistStamped`
   instead (teleop: `-r cmd_vel:=<input topic>`; script: `--cmd-topic <input topic>`).

---

## Assumptions and limits

- One stationary car in an otherwise empty area. A* treats only the detected car as an obstacle.
- Fixed forward camera at base x = 0.2106 m, y = 0.0175 m, level, no yaw offset.
- Needs a side or diagonal view. Near front or rear views cannot tell the two ends apart and are
  refused.
- The car must fit entirely in the frame when the robot stops, within the depth camera's range
  (default gate 0.3 to 12 m).
- The CFV rule (mirror right means rear on the left) is your calibration for this camera and model,
  applied to the original image direction.
- The reverse step moves straight back along the facing direction and does not check for other
  obstacles behind the robot.
- The depth estimation, A* planning and navigation loop were tested on synthetic depth images and
  a simulated robot. They have not yet been run in your Gazebo world, so verify the topic names
  (step 5) and the debug image on the first run.
