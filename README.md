# Pose estimation and rear alignment

working video: 
https://drive.google.com/file/d/177wl_-af6_aVhrmi-ClD8hpTfykzzgOY/view?usp=sharing

A Clearpath Jackal (ROS 2 Jazzy, Gazebo) scouts clockwise until a car fills the frame, works out
which end is the rear, measures the car with the depth camera, plans an A* path to a spot behind
the rear bumper, drives there, turns to face the car and backs up a little for a clear view.


## How it works

 1 SCOUT - rotate clockwise only, in small yaw steps, running YOLO on every fresh frame
      
 2 FRAME-FILL - first car box that fits entirely inside the image (2 px margin) and HOLD, robot stops
      
 3 IDENTIFY - once the robot is stationary, a CFV orientation model classifies the car crop and its mirror image. The result must be confident, self-consistent and a side/diagonal view. Then:
 mirror RIGHT = rear is on the image/robot LEFT  (robot will go left)
 mirror LEFT  = rear is on the image/robot RIGHT (robot will go right)
 4 DEPTH - median of a few depth frames; the car surface inside the box is projected onto the ground plane and a RANSAC line fit finds the long side. This gives: distance from camera, car centre, orientation, length
      
 5 REAR POINT - of the two car ends, the one on the CFV image side is the rear
      
 6 PLAN - A* in the odom frame around the car's box (inflated 0.65 m) to a goal 1.0 m behind the rear bumper, facing the car
      
 7 DRIVE - turn / drive along the waypoints using odometry, final align to face the car, then reverse 0.5 m for a better view


No Gazebo ground-truth poses are used. The car's pose comes from the depth camera, and the robot's
pose from `/j100_0000/platform/odom`.

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


Optional viewers and checks:

```bash

ros2 run rqt_image_view rqt_image_view



### 6. Teleop (for positioning only)

Use it to place the Jackal somewhere sensible before the run.

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




