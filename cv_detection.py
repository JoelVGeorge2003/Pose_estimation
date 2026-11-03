#!/usr/bin/env python3

import argparse
import math
import heapq
import threading
import time
import warnings
from collections import namedtuple
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from ultralytics import YOLO

import os
import sys
from types import SimpleNamespace
import torch
import numpy as np
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path.home() / 'jackal_sim/cfv/car-orientation'))
from models.multitask_models import ConvNextSmallMultiModel

VIEWS = [
    'front', 'front-left', 'left side', 'rear-left',
    'rear', 'rear-right', 'right side', 'front-right',
]


FLIP_INDEX = [0, 7, 6, 5, 4, 3, 2, 1]


def red_cue(crop):
    """Experimental color cue, NOT a trained taillight detector."""
    h, w = crop.shape[:2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 90, 35), (12, 255, 255))
    mask |= cv2.inRange(hsv, (168, 90, 35), (179, 255, 255))
    # Ignore central bodywork and lowest wheel/ground region.
    mask[int(h*.75):, :] = 0
    mask[:, int(w*.30):int(w*.70)] = 0
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    areas = [0., 0.]
    points = []
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < max(3, int(h*w*.00015)):
            continue
        cx, cy = centroids[i]
        side = 0 if cx < w*.5 else 1
        areas[side] += area
        points.append((int(cx), int(cy)))
    total = sum(areas)
    if total < max(6, h*w*.0003):
        return 'uncertain', points, 'Too little red evidence'
    if total > h*w*.04:
        return 'uncertain', points, 'Too much red: possible body paint'
    winner = int(areas[1] > areas[0])
    if areas[winner] < 3*max(areas[1-winner], 1):
        return 'uncertain', points, 'Red evidence at both ends'
    return ('IMAGE LEFT' if winner == 0 else 'IMAGE RIGHT'), points, 'Red end cue only'


def interpret_view(i, j, score, mirror_score, cue, threshold):
    if not all(math.isfinite(v) for v in (score, mirror_score)):
        return ('UNCERTAIN', 'Invalid model scores')
    if min(score, mirror_score) < threshold:
        return ('UNCERTAIN', 'Low model score')
    if j != FLIP_INDEX[i]:
        return ('UNCERTAIN', 'Original and mirror labels disagree')
    if j in (5,6,7):
        direction = 'IMAGE LEFT'
    elif j in (1,2,3):
        direction = 'IMAGE RIGHT'
    else:
        return ('FRONT/REAR UNVERIFIED', 'Need a side or diagonal side label')
    return (f'Rear likely {direction} - USER MAPPING',
            'CFV selects end; depth camera supplies geometry. Red cue is diagnostic only.')


def classify_car(cfv, frame, bbox, threshold):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0,x1), max(0,y1)
    x2, y2 = min(width,x2), min(height,y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return ['Invalid car crop'], []
    rgb = cv2.cvtColor(cv2.resize(crop, (224,224)), cv2.COLOR_BGR2RGB)
    original = torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255.
    with torch.inference_mode():
        logits, _ = cfv(torch.stack([original, original.flip(-1)]))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    i, j = int(probs[0].argmax()), int(probs[1].argmax())
    score, mirror_score = float(probs[0,i]), float(probs[1,j])
    consistent = j == FLIP_INDEX[i]
    cue, points, reason = red_cue(crop)
    status, explanation = interpret_view(i, j, score, mirror_score, cue, threshold)
    lines = [
        status,
        f'Raw CFV: {VIEWS[i]} {score:.3f} | mirror: {VIEWS[j]} {mirror_score:.3f}',
        f'Mirror consistent: {consistent} | red cue: {cue}',
        explanation,
    ]
    return lines, [(x1+x, y1+y) for x,y in points]


NS='/j100_0000'


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def yaw_of(msg):
    q=msg.pose.pose.orientation
    return math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))


def full_box(box,width,height,margin):
    x1,y1,x2,y2=box
    return (x2>x1 and y2>y1 and x1>=margin and y1>=margin
            and x2<=width-margin and y2<=height-margin)



class Scanner(Node):
    def __init__(self,args):
        super().__init__('jackal_clockwise_car_scan')
        self.args=args
        self.set_parameters([Parameter('use_sim_time',value=True)])
        self.bridge=CvBridge()
        self.lock=threading.RLock()
        self.latest=None
        self.depth=None
        self.info=None
        self.odom=None
        self.odom_received=0.
        self.target_yaw=None
        self.last_frame_time=0.
        self.holding=False
        self.hold_time=0.
        self.state='WAITING FOR CAMERA AND ODOMETRY'
        self.command_time=time.monotonic()
        self.publisher=self.create_publisher(TwistStamped,args.cmd_topic,1)
        self.images=self.create_publisher(Image,'/clockwise_scan/image',1)
        self.create_subscription(Image,NS+'/sensors/camera_0/color/image',
                                 self.receive_image,qos_profile_sensor_data)
        self.create_subscription(Image,args.depth_topic,self.receive_depth,qos_profile_sensor_data)
        self.create_subscription(CameraInfo,args.camera_info_topic,self.receive_info,qos_profile_sensor_data)
        self.create_subscription(Odometry,NS+'/platform/odom',self.receive_odom,10)
        self.create_timer(.05,self.control)

    def receive_image(self,msg):
        with self.lock:
            self.latest=(msg,time.monotonic())

    def receive_depth(self,msg):
        with self.lock:
            self.depth=(msg,time.monotonic())

    def receive_info(self,msg):
        with self.lock:
            self.info=msg

    def receive_odom(self,msg):
        with self.lock:
            self.odom=msg
            self.odom_received=time.monotonic()

    def send(self,w):
        if not rclpy.ok() or not self.args.execute:
            return   # observation only: publish nothing, so teleop / twist_mux are never overridden
        msg=TwistStamped()
        msg.header.stamp=self.get_clock().now().to_msg()
        msg.header.frame_id='base_link'
        msg.twist.linear.x=0.
        msg.twist.angular.z=float(w) if self.args.execute else 0.
        try:
            self.publisher.publish(msg)
        except Exception:
            if rclpy.ok():
                raise

    def hold(self,score):
        with self.lock:
            if not self.holding:
                print(f'HOLD latched: full_bbox=True (detector score={score:.6f}; no stop-score threshold)',flush=True)
            if not self.holding:
                self.hold_time=time.monotonic()
            self.holding=True
            self.state='HOLD - full_bbox=True'
            self.send(0.)

    def control(self):
        with self.lock:
            now=time.monotonic()
            self.command_time=now
            if self.holding:
                self.send(0.)
                return
            if (self.odom is None or now-self.odom_received>.5 or
                now-self.last_frame_time>self.args.stale_seconds or
                self.get_clock().now().nanoseconds==0):
                self.state='WAIT - fresh camera, odometry and /clock required'
                self.target_yaw=None
                self.send(0.)
                return
            a=yaw_of(self.odom)
            if self.target_yaw is None:
                self.target_yaw=wrap(a-math.radians(self.args.yaw_step_deg))
            error=wrap(self.target_yaw-a)
            if error>=-.015:
                # Select another clockwise yaw goal; never command positive yaw.
                self.target_yaw=wrap(a-math.radians(self.args.yaw_step_deg))
                error=wrap(self.target_yaw-a)
            w=-min(self.args.speed,max(.01,.8*abs(error)))
            self.state=f'CLOCKWISE | yaw={math.degrees(a):.1f} target={math.degrees(self.target_yaw):.1f} deg'
            self.send(w)

    def watchdog(self,stop):
        while not stop.wait(.1):
            with self.lock:
                if time.monotonic()-self.command_time>.4:
                    self.send(0.)


def scan_main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true',help='Enable rotation; otherwise observation only')
    parser.add_argument('--speed',type=float,default=.05,help='Clockwise angular speed limit, rad/s')
    parser.add_argument('--yaw-step-deg',type=float,default=10.)
    parser.add_argument('--confidence',type=float,default=None,help='Legacy option; ignored. A full bbox always stops motion.')
    parser.add_argument('--margin',type=int,default=2,help='BBox margin from image edges, pixels')
    parser.add_argument('--weights',default=str(Path.home()/'jackal_sim/yolo11n.pt'))
    parser.add_argument('--stale-seconds',type=float,default=2.)
    parser.add_argument('--cfv-confidence',type=float,default=.70)
    parser.add_argument('--cmd-topic',default=NS+'/platform/cmd_vel',
                        help='TwistStamped velocity topic to publish on (twist_mux OUTPUT by default; '
                             'pass a twist_mux INPUT topic to let the mux arbitrate)')
    parser.add_argument('--depth-topic',default=NS+'/sensors/camera_0/depth/image',
                        help='Depth image aligned with the colour image (32FC1 m or 16UC1 mm)')
    parser.add_argument('--camera-info-topic',default=NS+'/sensors/camera_0/color/camera_info')
    parser.add_argument('--hfov-deg',type=float,default=None,
                        help='Fallback horizontal FOV if no CameraInfo arrives')
    parser.add_argument('--depth-frames',type=int,default=3,help='Depth frames median-stacked for the estimate')
    parser.add_argument('--depth-timeout',type=float,default=6.,help='Seconds to wait for depth frames')
    parser.add_argument('--backup-distance',type=float,default=.5,
                        help='Metres to reverse after aligning with the car rear, for a better view; 0 disables')
    parser.add_argument('--debug-image',default=str(Path.home()/'jackal_sim/rear_plan_debug.png'),
                        help='Top-down PNG of the depth estimate and A* path; empty string disables')
    parser.add_argument('--confirm-frames',type=int,default=1)
    args,ros_args=parser.parse_known_args()
    if args.confirm_frames < 1 or args.depth_frames < 1 or args.depth_timeout <= 0:
        parser.error('Require positive confirm-frames, depth-frames and depth-timeout')
    if not 0 <= args.backup_distance <= 2:
        parser.error('backup-distance must be between 0 and 2 m')
    confirmed = False
    plan_inputs = None
    rear_side = None
    streak = 0
    previous_hint = None
    print('DEPTH-ASSISTED TRIAL: CFV mirror mapping selects rear end; the depth camera supplies metric geometry.', flush=True)
    print('A* models only the detected car as an obstacle, not other objects.', flush=True)
    print('Mapping uses ORIGINAL camera image: mirror right -> rear left; mirror left -> rear right.', flush=True)
    print('Assumes fixed forward camera at base x=.2106,y=.0175 and a stationary car.', flush=True)
    if not 0 <= args.cfv_confidence <= 1:
        parser.error('cfv-confidence must be between 0 and 1')
    if not 0<args.speed<=.3 or not 2<=args.yaw_step_deg<=45:
        parser.error('Require 0<speed<=0.3 rad/s and yaw-step-deg between 2 and 45')
    if args.margin<0 or args.stale_seconds<=0:
        parser.error('Require nonnegative margin and positive stale-seconds')
    print('Loading YOLO car detector...',flush=True)
    model=YOLO(args.weights)
    car_ids=[int(i) for i,name in model.names.items() if name=='car']
    if not car_ids:
        raise RuntimeError('Weights have no car class')
    print('Loading CFV on CPU...', flush=True)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    checkpoint = hf_hub_download(
        repo_id='fort-cyber/Car-orientation-image',
        filename='convnext-multi-weak.pth',
    )
    cfv = ConvNextSmallMultiModel(SimpleNamespace(pretrained=False))
    cfv.load_state_dict(
        torch.load(checkpoint, map_location='cpu', weights_only=True),
        strict=True,
    )
    cfv.eval()
    cfv_lines = ['Waiting for full car box']
    cue_points = []
    held_size = None
    held_bbox = None
    last_cfv_time = 0.0
    rclpy.init(args=ros_args)
    node=Scanner(args)
    executor=SingleThreadedExecutor()
    executor.add_node(node)
    def spin():
        try:
            executor.spin()
        except ExternalShutdownException:
            pass
    thread=threading.Thread(target=spin,daemon=True)
    thread.start()
    stop=threading.Event()
    watchdog=threading.Thread(target=node.watchdog,args=(stop,),daemon=True)
    watchdog.start()
    print(f'Stop rule: full_bbox=True; clockwise speed <= {args.speed} rad/s',flush=True)
    if args.confidence is not None:
        print('--confidence is ignored: every detected full bbox triggers HOLD.',flush=True)
    print('EXECUTION ENABLED' if args.execute else 'OBSERVATION ONLY; add --execute to rotate',flush=True)
    print('Q or Ctrl+C: stop. HOLD waits for identification, then hands off to A*.',flush=True)
    last_stamp=None
    try:
        while rclpy.ok():
            if cv2.waitKey(1)&255==ord('q'):
                break
            with node.lock:
                item=node.latest
                node.latest=None
            if item is None:
                time.sleep(.01)
                continue
            msg,received=item
            source_stamp=(msg.header.stamp.sec,msg.header.stamp.nanosec)
            if source_stamp==last_stamp:
                continue
            last_stamp=source_stamp
            if time.monotonic()-received>args.stale_seconds:
                continue
            frame=node.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8').copy()
            height,width=frame.shape[:2]
            raw_frame = frame.copy()
            depth_m = None
            with node.lock:
                depth_item = node.depth
            if depth_item is not None and time.monotonic()-depth_item[1] <= args.stale_seconds:
                depth_m = depth_to_meters(node.bridge.imgmsg_to_cv2(depth_item[0],desired_encoding='passthrough'))
            result=model.predict(frame,classes=car_ids,conf=.10,imgsz=416,
                                 device='cpu',verbose=False)[0]
            fresh=time.monotonic()-received<=args.stale_seconds
            best=0.
            for bbox,score in zip(result.boxes.xyxy.cpu().numpy(),
                                  result.boxes.conf.cpu().numpy()):
                score=float(score)
                best=max(best,score)
                inside=full_box(bbox,width,height,args.margin)
                x1,y1,x2,y2=map(int,bbox)
                color=(0,255,0) if inside else (0,180,255)
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                range_text=''
                if depth_m is not None:
                    d=bbox_range(depth_m,scale_box(bbox,(width,height),(depth_m.shape[1],depth_m.shape[0])))
                    if d is not None:
                        range_text=f' range~{d:.2f}m'
                cv2.putText(frame,f'car {score:.6f} full_bbox={inside}{range_text}',
                            (max(0,x1),max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
                if inside:
                    node.hold(score)
                    if held_bbox is None:
                        held_bbox = tuple(float(v) for v in bbox)
                        held_size = (width,height)
            if node.holding and held_bbox is not None:
                with node.lock:
                    odom = node.odom
                    stationary = (
                        odom is not None and time.monotonic()-node.odom_received < .5
                        and abs(odom.twist.twist.angular.z) < .025
                        and math.hypot(odom.twist.twist.linear.x,
                                       odom.twist.twist.linear.y) < .02
                        and received > node.hold_time + 1.0
                    )
                if stationary and time.monotonic()-last_cfv_time >= 1.0:
                    ow, oh = held_size
                    box = (held_bbox[0]*width/ow, held_bbox[1]*height/oh,
                           held_bbox[2]*width/ow, held_bbox[3]*height/oh)
                    cfv_lines, cue_points = classify_car(
                        cfv, raw_frame, box, args.cfv_confidence
                    )
                    last_cfv_time = time.monotonic()
                    print(' | '.join(cfv_lines), flush=True)
                    hint = cfv_lines[0]
                    if hint.startswith('Rear likely IMAGE'):
                        streak = streak + 1 if hint == previous_hint else 1
                        previous_hint = hint
                    else:
                        streak = 0
                        previous_hint = None
                    print(f'Side confirmation: {streak}/{args.confirm_frames}', flush=True)
                    if streak >= args.confirm_frames:
                        node.send(0.)
                        confirmed = True
                        print('IDENTIFIED -> depth estimate -> A* PLANNING', flush=True)
                        break
                elif not stationary:
                    streak = 0
                    previous_hint = None
                    cfv_lines = ['HOLD: waiting for fresh stationary odometry']
                    cue_points = []
                for px,py in cue_points:
                    cv2.circle(frame, (px,py), 6, (255,0,255), 2)
            with node.lock:
                # Only fully processed fresh camera data permits ongoing rotation.
                if fresh:
                    node.last_frame_time=received
                status=node.state
            cv2.putText(frame,status,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1)
            frame = cv2.copyMakeBorder(frame,0,100,0,0,cv2.BORDER_CONSTANT,value=(25,25,25))
            for line_no, line in enumerate(cfv_lines):
                cv2.putText(frame,line,(8,height+20+line_no*23),
                            cv2.FONT_HERSHEY_SIMPLEX,.43,(0,255,255),1)
            output=node.bridge.cv2_to_imgmsg(frame,encoding='bgr8')
            output.header=msg.header
            node.images.publish(output)
            cv2.imshow('CFV + experimental red rear cue - Q stops',frame)
            print(f'{status} | best raw car score={best:.9f} | stop rule: full_bbox=True',flush=True)
        if confirmed:
            rear_side = 'left' if 'IMAGE LEFT' in previous_hint else 'right'
            try:
                plan_inputs = acquire_car_estimate(node,args,box,(width,height),rear_side)
            except RuntimeError as exc:
                print(f'DEPTH ESTIMATE FAILED - not moving: {exc}',flush=True)
                confirmed = False
    except KeyboardInterrupt:
        pass
    finally:
        with node.lock:
            node.holding=True
            node.send(0.)
        stop.set()
        executor.shutdown()
        thread.join(timeout=2)
        watchdog.join(timeout=1)
        node.send(0.)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()


    if confirmed and plan_inputs is not None:
        car, robot_pose, est = plan_inputs
        # Scanner publisher and executor are destroyed before navigation starts.
        nav_main(args.execute, car, robot_pose, est, rear_side, args.debug_image, args.backup_distance, args.cmd_topic)



CAMERA_X = .2106           # camera mount in base_link, forward (m)
CAMERA_Y = .0175           # camera mount in base_link, left (m)
CMD_TOPIC = NS + '/platform/cmd_vel'
ODOM_TOPIC = NS + '/platform/odom'

CAR_LENGTH = 4.5           # nominal; used for plausibility bounds only
CAR_WIDTH = 1.8            # used to push the visible side face back to the centreline
REAR_CLEARANCE = 1.0       # rear bumper -> robot centre at the goal (m)

MIN_DEPTH = 0.3            # depth gate (m)
MAX_DEPTH = 12.0
UPPER_FRACTION = 0.70      # use top 70% of the box: body panels, not wheels/floor
MIN_CAR_PIXELS = 150
LINE_TOL = 0.08            # RANSAC inlier distance in the ground plane (m)

MAX_LINEAR = 0.60
MAX_ANGULAR = 0.60
YAW_TOL = 0.06
NAV_TIMEOUT = 900.0
GRID_RESOLUTION = 0.15
MAP_MARGIN = 5.0
INFLATION = 0.65           # planning margin around the car box (m)
TRACK_INFLATION = 0.40     # hard-stop margin while driving; > Jackal radius, < planning margin

# Car box in the ODOM frame. rear_yaw = direction from the car centre towards its rear.
Car = namedtuple('Car', 'x y rear_yaw length')


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(value, limit):
    return max(-limit, min(limit, value))


def robot_to_world(p, pose):
    rx, ry, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return (rx + c*p[0] - s*p[1], ry + s*p[0] + c*p[1])



def depth_to_meters(raw):
    """32FC1 (m) or 16UC1 (mm) -> float32 metres, invalid -> NaN."""
    d = np.asarray(raw)
    d = d.astype(np.float32)/1000. if d.dtype == np.uint16 else d.astype(np.float32)
    with np.errstate(invalid='ignore'):
        d[~np.isfinite(d) | (d <= 0.)] = np.nan
    return d


def scale_box(box, src_wh, dst_wh):
    sx, sy = dst_wh[0]/src_wh[0], dst_wh[1]/src_wh[1]
    x1, y1, x2, y2 = box
    return (x1*sx, y1*sy, x2*sx, y2*sy)


def intrinsics_for(info, hfov_deg, dw, dh):
    """(fx, fy, cx, cy) at depth-image size, from CameraInfo or a horizontal FOV."""
    if info is not None and info.k[0] > 0 and info.width > 0 and info.height > 0:
        sx, sy = dw/info.width, dh/info.height
        return (info.k[0]*sx, info.k[4]*sy, info.k[2]*sx, info.k[5]*sy)
    if hfov_deg:
        fx = (dw/2.)/math.tan(math.radians(hfov_deg)/2.)
        return (fx, fx, dw/2., dh/2.)
    return None


def bbox_range(depth, box):
    """Median depth over the central half of a box. Quick live readout only."""
    h, w = depth.shape
    x1, y1, x2, y2 = box
    cx, cy, hw, hh = (x1+x2)/2., (y1+y2)/2., (x2-x1)/4., (y2-y1)/4.
    xa, xb = int(max(0, cx-hw)), int(min(w, cx+hw))
    ya, yb = int(max(0, cy-hh)), int(min(h, cy+hh))
    if xb <= xa or yb <= ya:
        return None
    roi = depth[ya:yb, xa:xb]
    roi = roi[np.isfinite(roi)]
    return float(np.median(roi)) if roi.size else None


def smooth_core(Z, rel=.08, absolute=.10):
    """Valid pixels whose right and lower neighbours have a similar depth.
    A pixel on the near side of a depth jump is dropped, so a connected component
    of the result is one continuous surface (the car), separate from the wall behind."""
    h, w = Z.shape
    thr = rel*Z + absolute
    right = np.ones((h, w), bool)
    down = np.ones((h, w), bool)
    with np.errstate(invalid='ignore'):
        right[:, :-1] = np.abs(Z[:, 1:] - Z[:, :-1]) < thr[:, :-1]
        down[:-1, :] = np.abs(Z[1:, :] - Z[:-1, :]) < thr[:-1, :]
    return np.isfinite(Z) & right & down


def ransac_line(pts, rng, iters=250, tol=LINE_TOL):
    n = len(pts)
    if n < 20:
        return None
    best, best_n = None, 0
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        d = pts[j] - pts[i]
        norm = float(np.hypot(d[0], d[1]))
        if norm < .5:
            continue
        normal = np.array([-d[1], d[0]])/norm
        inl = np.abs((pts - pts[i]) @ normal) < tol
        count = int(inl.sum())
        if count > best_n:
            best, best_n = inl, count
    return best


def fit_line(pts):
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    return c, vt[0]/np.linalg.norm(vt[0])


def estimate_car_from_depth(depth, box, intr, rear_image_side):
    """depth: HxW metres. box: car bbox in DEPTH pixel coords. intr: (fx,fy,cx,cy).
    rear_image_side: 'left'|'right' from CFV. Returns a dict; everything is in base_link
    (x forward, y left) at the moment the depth frame was taken."""
    fx, fy, cx, cy = intr
    h, w = depth.shape
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2, y2 = min(w, int(math.ceil(box[2]))), min(h, int(math.ceil(box[3])))
    y2 = y1 + int((y2-y1)*UPPER_FRACTION)
    if x2-x1 < 8 or y2-y1 < 8:
        raise RuntimeError('Car box too small in the depth image')
    Z = depth[y1:y2, x1:x2]
    with np.errstate(invalid='ignore'):
        core = smooth_core(Z) & (Z > MIN_DEPTH) & (Z < MAX_DEPTH)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), connectivity=4)
    if count < 2:
        raise RuntimeError('No valid depth inside the car box (check depth topic / max range)')
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[label, cv2.CC_STAT_AREA] < MIN_CAR_PIXELS:
        raise RuntimeError('Largest depth surface in the car box is too small')
    vs, us = np.nonzero(labels == label)
    z = Z[vs, us].astype(np.float64)
    xo = (us + x1 - cx)*z/fx                      # optical X (right)
    pts = np.column_stack((CAMERA_X + z, CAMERA_Y - xo))   # base_link ground plane
    planar = np.hypot(xo, z)                      # horizontal range from the camera

    rng = np.random.RandomState(0)
    remaining, line = pts, None
    for _ in range(3):
        sample = remaining if len(remaining) <= 4000 else remaining[rng.choice(len(remaining), 4000, replace=False)]
        inl = ransac_line(sample, rng)
        if inl is None:
            break
        c, u = fit_line(sample[inl])
        dist = np.abs((remaining - c) @ np.array([-u[1], u[0]]))
        members = remaining[dist < LINE_TOL*1.5]
        c, u = fit_line(members)
        t = (members - c) @ u
        t_lo, t_hi = np.percentile(t, [0.5, 99.5])
        if .6*CAR_LENGTH <= t_hi - t_lo <= 1.4*CAR_LENGTH:
            line = (c, u, float(t_lo), float(t_hi))
            break
        remaining = remaining[dist >= LINE_TOL*1.5]     # probably the end face; try the next line
        if len(remaining) < 20:
            break
    if line is None:
        raise RuntimeError('No car-length side face found in depth (car partly hidden or too far?)')
    c, u, t_lo, t_hi = line
    length = t_hi - t_lo

    cam = np.array([CAMERA_X, CAMERA_Y])
    n = np.array([-u[1], u[0]])
    if float(n @ (c - cam)) < 0:
        n = -n                                         # normal pointing away from the camera
    center = c + u*(t_lo + t_hi)/2. + n*CAR_WIDTH/2.   # visible face lies W/2 in front of the centreline

    ratios = {}                                        # image-right per metre forward, per visible end
    for sign, t_end in ((-1, t_lo), (1, t_hi)):
        p = c + u*t_end
        forward = p[0] - CAMERA_X
        if forward <= .1:
            raise RuntimeError('Car end is behind the camera; cannot pick the rear from the image side')
        ratios[sign] = -(p[1] - CAMERA_Y)/forward
    if abs(ratios[1] - ratios[-1]) < .10:
        raise RuntimeError('End-on view: both car ends project to the same image column')
    rear_sign = (min if rear_image_side == 'left' else max)(ratios, key=ratios.get)
    rear_vec = rear_sign*u
    return {
        'car': Car(float(center[0]), float(center[1]), math.atan2(rear_vec[1], rear_vec[0]), float(length)),
        'range_near': float(np.percentile(planar, 5)),
        'range_median': float(np.median(planar)),
        'ratios': ratios,
        'points': pts,
    }


def acquire_car_estimate(node, args, box, color_size, rear_image_side):
    """Runs while the robot is held still and the scanner node is still alive."""
    deadline = time.monotonic() + args.depth_timeout
    frames, stamps = [], set()
    while time.monotonic() < deadline and len(frames) < args.depth_frames:
        with node.lock:
            item, odom, info, hold_time = node.depth, node.odom, node.info, node.hold_time
        if item is not None:
            msg, received = item
            stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
            if received > hold_time + 1.0 and stamp not in stamps:
                stamps.add(stamp)
                frames.append(depth_to_meters(node.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')))
        time.sleep(.02)
    if not frames:
        raise RuntimeError(f'No fresh depth frames on {args.depth_topic}')
    if odom is None:
        raise RuntimeError('No odometry for the robot pose')
    if len(frames) > 1:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            depth = np.nanmedian(np.stack(frames), axis=0)
    else:
        depth = frames[0]
    dh, dw = depth.shape
    intr = intrinsics_for(info, args.hfov_deg, dw, dh)
    if intr is None:
        raise RuntimeError(f'No camera intrinsics: nothing on {args.camera_info_topic}. '
                           'Find the right topic with `ros2 topic list | grep camera_0` '
                           'or pass --hfov-deg.')
    est = estimate_car_from_depth(depth, scale_box(box, color_size, (dw, dh)), intr, rear_image_side)
    robot_pose = (odom.pose.pose.position.x, odom.pose.pose.position.y, yaw_of(odom))
    cr = est['car']
    cw = robot_to_world((cr.x, cr.y), robot_pose)
    car = Car(cw[0], cw[1], normalize_angle(cr.rear_yaw + robot_pose[2]), cr.length)
    print(f'DEPTH: {len(frames)} frame(s), {dw}x{dh}, fx={intr[0]:.1f}', flush=True)
    print(f'CAR DISTANCE FROM CAMERA: nearest {est["range_near"]:.2f} m, median {est["range_median"]:.2f} m '
          f'(horizontal range)', flush=True)
    print(f'Car in robot frame: centre=({cr.x:.2f} fwd, {cr.y:+.2f} left) m, '
          f'length={cr.length:.2f} m, rear direction={math.degrees(cr.rear_yaw):.1f} deg', flush=True)
    print(f'CFV rear = image {rear_image_side}; end image-slopes (right/forward)={est["ratios"]}', flush=True)
    return car, robot_pose, est



def rear_goal(car):
    
    ux, uy = math.cos(car.rear_yaw), math.sin(car.rear_yaw)
    bumper = (car.x + ux*car.length/2., car.y + uy*car.length/2.)
    goal = (bumper[0] + ux*REAR_CLEARANCE, bumper[1] + uy*REAR_CLEARANCE,
            normalize_angle(car.rear_yaw + math.pi))
    return goal, bumper


def inside_inflated_car(x, y, car, inflation=INFLATION):
    dx, dy = x - car.x, y - car.y
    c, s = math.cos(car.rear_yaw), math.sin(car.rear_yaw)
    local_x = c*dx + s*dy
    local_y = -s*dx + c*dy
    return abs(local_x) <= car.length/2. + inflation and abs(local_y) <= CAR_WIDTH/2. + inflation


def straight_path_clear(start, goal, car, inflation=INFLATION):
    sx, sy = start
    gx, gy = goal
    steps = max(1, math.ceil(math.hypot(gx - sx, gy - sy)/0.1))
    for i in range(steps + 1):
        t = i/steps
        if inside_inflated_car(sx + t*(gx - sx), sy + t*(gy - sy), car, inflation):
            return False
    return True


def plan_astar(start, goal, car):
    resolution = GRID_RESOLUTION
    sx, sy = start
    gx, gy = goal
    xmin = min(sx, gx, car.x) - MAP_MARGIN
    ymin = min(sy, gy, car.y) - MAP_MARGIN
    xmax = max(sx, gx, car.x) + MAP_MARGIN
    ymax = max(sy, gy, car.y) + MAP_MARGIN

    def to_grid(x, y):
        return (round((x - xmin)/resolution), round((y - ymin)/resolution))

    def to_world(cell):
        return (xmin + cell[0]*resolution, ymin + cell[1]*resolution)

    def blocked(cell):
        x, y = to_world(cell)
        return x < xmin or x > xmax or y < ymin or y > ymax or inside_inflated_car(x, y, car)

    start_node = to_grid(sx, sy)
    goal_node = to_grid(gx, gy)
    if blocked(start_node):
        raise RuntimeError('A* start is inside an obstacle')
    if blocked(goal_node):
        raise RuntimeError('A* goal is inside an obstacle')

    def heuristic(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    queue = [(0, start_node)]
    costs = {start_node: 0.0}
    parents = {}
    visited = set()
    while queue:
        _, current = heapq.heappop(queue)
        if current in visited:
            continue
        visited.add(current)
        if current == goal_node:
            break
        for dx, dy in moves:
            nxt = (current[0] + dx, current[1] + dy)
            if blocked(nxt) or not straight_path_clear(to_world(current), to_world(nxt), car):
                continue
            if dx != 0 and dy != 0:
                if blocked((current[0] + dx, current[1])) or blocked((current[0], current[1] + dy)):
                    continue
            new_cost = costs[current] + math.hypot(dx, dy)
            if new_cost < costs.get(nxt, float('inf')):
                costs[nxt] = new_cost
                parents[nxt] = current
                heapq.heappush(queue, (new_cost + heuristic(nxt, goal_node), nxt))
    if goal_node not in costs:
        raise RuntimeError('A* could not find a path')
    cells = []
    node = goal_node
    while node != start_node:
        cells.append(node)
        node = parents[node]
    cells.reverse()
    raw_path = [start] + [to_world(c) for c in cells] + [goal]
    path = []
    current = 0
    while current < len(raw_path) - 1:
        furthest = current + 1
        for j in range(len(raw_path) - 1, current, -1):
            if straight_path_clear(raw_path[current], raw_path[j], car):
                furthest = j
                break
        if not straight_path_clear(raw_path[current], raw_path[furthest], car):
            raise RuntimeError('Invalid grid-to-world connector; no clear path')
        path.append(raw_path[furthest])
        current = furthest
    return path


def box_corners(car, half_l, half_w):
    c, s = math.cos(car.rear_yaw), math.sin(car.rear_yaw)
    return [(car.x + c*a - s*b, car.y + s*a + c*b)
            for a, b in ((half_l, half_w), (half_l, -half_w), (-half_l, -half_w), (-half_l, half_w))]


def save_debug_plot(filename, robot_pose, car, goal, bumper, path, points_robot):
    pts = [robot_to_world(p, robot_pose) for p in points_robot[::4]]
    inflated = box_corners(car, car.length/2. + INFLATION, CAR_WIDTH/2. + INFLATION)
    body = box_corners(car, car.length/2., CAR_WIDTH/2.)
    allp = pts + inflated + path + [robot_pose[:2], goal[:2]]
    xmin, xmax = min(p[0] for p in allp) - 1., max(p[0] for p in allp) + 1.
    ymin, ymax = min(p[1] for p in allp) - 1., max(p[1] for p in allp) + 1.
    scale = min(80., 1600./max(xmax - xmin, ymax - ymin))
    img = np.full((int((ymax - ymin)*scale), int((xmax - xmin)*scale), 3), 255, np.uint8)

    def px(p):
        return (int((p[0] - xmin)*scale), int((ymax - p[1])*scale))

    def poly(points):
        return np.array([px(p) for p in points], np.int32)

    cv2.fillPoly(img, [poly(inflated)], (225, 225, 225))
    cv2.fillPoly(img, [poly(body)], (150, 150, 150))
    for p in pts:
        cv2.circle(img, px(p), 1, (200, 100, 0), -1)            # depth points (blue)
    cv2.circle(img, px(bumper), 6, (0, 0, 255), -1)            # rear bumper (red)
    chain = [robot_pose[:2]] + list(path)
    for a, b in zip(chain[:-1], chain[1:]):
        cv2.line(img, px(a), px(b), (0, 160, 0), 2)            # A* path (green)
    for pose, colour in ((robot_pose, (200, 0, 200)), (goal, (0, 140, 255))):
        tip = (pose[0] + .6*math.cos(pose[2]), pose[1] + .6*math.sin(pose[2]))
        cv2.arrowedLine(img, px(pose[:2]), px(tip), colour, 3, tipLength=.3)   # robot / goal heading
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(filename), img)
    print(f'Debug top-down plot written to {filename}', flush=True)


class JackalController(Node):

    def __init__(self, cmd_topic=CMD_TOPIC):
        super().__init__('jackal_rear_navigator')
        self.set_parameters([Parameter('use_sim_time', value=True)])
        self.publisher = self.create_publisher(TwistStamped, cmd_topic, 10)
        self.create_subscription(Odometry, ODOM_TOPIC, self.receive_odom, 10)
        self.lock = threading.Lock()
        self.linear = 0.0
        self.angular = 0.0
        self.last_update = time.monotonic()
        self.pose = None
        self.pose_time = 0.0
        self.timer = self.create_timer(0.1, self.publish_velocity)

    def receive_odom(self, msg):
        with self.lock:
            self.pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw_of(msg))
            self.pose_time = time.monotonic()

    def get_pose(self, max_age=.5):
        with self.lock:
            if self.pose is None or time.monotonic() - self.pose_time > max_age:
                return None
            return self.pose

    def set_velocity(self, linear, angular):
        with self.lock:
            self.linear = float(linear)
            self.angular = float(angular)
            self.last_update = time.monotonic()

    def publish_velocity(self):
        with self.lock:
            linear = self.linear
            angular = self.angular
            if time.monotonic() - self.last_update > 2.0:
                linear = 0.0
                angular = 0.0
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = linear
        msg.twist.angular.z = angular
        self.publisher.publish(msg)

    def stop(self):
        self.set_velocity(0.0, 0.0)
        if rclpy.ok():
            self.publish_velocity()


def report_final(x, y, yaw, yaw_error, car):
    ux, uy = math.cos(car.rear_yaw), math.sin(car.rear_yaw)
    behind = ux*(x - car.x) + uy*(y - car.y) - car.length/2.
    lateral = -uy*(x - car.x) + ux*(y - car.y)
    print(f'Final odom pose: x={x:.3f} y={y:.3f} yaw={yaw:.3f}')
    print(f'Jackal is {behind:.2f} m behind the rear bumper, {lateral:+.2f} m off the car '
          f'centreline, heading error {math.degrees(yaw_error):+.1f} deg - facing the car rear.')


def navigate(controller, path, final_yaw, car, backup=0.0):
    waypoint_index = 0
    backup_start = None
    state = 'TURN'
    start_time = time.monotonic()
    last_log = 0.0
    last_pose = None
    last_motion_time = start_time
    last_odom_ok = start_time
    print('Waypoints:', [(round(px, 2), round(py, 2)) for px, py in path])
    while rclpy.ok():
        now = time.monotonic()
        if now - start_time > NAV_TIMEOUT:
            raise RuntimeError('Navigation timed out')
        pose = controller.get_pose()
        if pose is None:
            controller.stop()
            if now - last_odom_ok > 2.0:
                raise RuntimeError('Odometry stale for 2 s; stopped')
            time.sleep(0.05)
            continue
        last_odom_ok = now
        x, y, yaw = pose
        if last_pose is None:
            last_pose = pose
        if math.hypot(x - last_pose[0], y - last_pose[1]) > 0.03 or abs(normalize_angle(yaw - last_pose[2])) > 0.04:
            last_pose = pose
            last_motion_time = now
        if now - last_motion_time > 20.0:
            raise RuntimeError('No Jackal movement detected for 20 seconds')
        if waypoint_index < len(path):
            tx, ty = path[waypoint_index]
            dx = tx - x
            dy = ty - y
            distance = math.hypot(dx, dy)
            if not straight_path_clear((x, y), (tx, ty), car, TRACK_INFLATION):
                raise RuntimeError('Tracking departed from clear route; stopped. Replan from current pose.')
            yaw_error = normalize_angle(math.atan2(dy, dx) - yaw)
            if distance < 0.08:
                controller.stop()
                print(f'Waypoint {waypoint_index + 1}/{len(path)} reached')
                waypoint_index += 1
                state = 'TURN'
                last_motion_time = now
                continue
            if state == 'TURN':
                if abs(yaw_error) < 0.12:
                    controller.stop()
                    state = 'DRIVE'
                    continue
                linear = 0.0
                angular = clamp(1.8*yaw_error, MAX_ANGULAR)
            else:
                if abs(yaw_error) > 0.45:
                    state = 'TURN'
                    controller.stop()
                    continue
                linear = min(MAX_LINEAR, 0.8*distance)*max(0.2, math.cos(yaw_error))
                angular = clamp(1.5*yaw_error, MAX_ANGULAR)
            controller.set_velocity(linear, angular)
        else:
            yaw_error = normalize_angle(final_yaw - yaw)
            if backup_start is None:
                state = 'FINAL_ALIGN'
                if abs(yaw_error) < YAW_TOL:
                    controller.stop()
                    if backup <= 0.0:
                        report_final(x, y, yaw, yaw_error, car)
                        return
                    backup_start = (x, y)
                    print(f'Aligned with the car rear; backing up {backup:.2f} m for a better view.')
                    continue
                controller.set_velocity(0.0, clamp(1.5*yaw_error, MAX_ANGULAR))
            else:
                state = 'BACKUP'
                remaining = backup - math.hypot(x - backup_start[0], y - backup_start[1])
                if remaining <= 0.02:
                    controller.stop()
                    report_final(x, y, yaw, yaw_error, car)
                    return
                # Straight reverse (away from the car), holding the facing direction.
                controller.set_velocity(-max(0.05, min(0.20, 0.8*remaining)),
                                        clamp(1.5*yaw_error, MAX_ANGULAR))
        if now - last_log > 1.5:
            print(f'[{state}] Jackal=({x:.2f},{y:.2f}), yaw={yaw:.2f}, waypoint={min(waypoint_index + 1, len(path))}/{len(path)}')
            last_log = now
        time.sleep(0.05)


def nav_main(execute, car, robot_pose, est, rear_image_side, debug_image, backup=0.0, cmd_topic=CMD_TOPIC):
    goal, bumper = rear_goal(car)
    rx, ry, ryaw = robot_pose
    gl = (-(math.sin(ryaw))*(goal[0] - rx) + math.cos(ryaw)*(goal[1] - ry))   # goal, +left of robot
    gf = (math.cos(ryaw)*(goal[0] - rx) + math.sin(ryaw)*(goal[1] - ry))
    print('DEPTH MODE: CFV mirror -> rear image side; car pose measured by the depth camera.')
    print(f'CFV rear on image {rear_image_side.upper()} -> goal is {abs(gl):.2f} m to the robot\'s '
          f'{"LEFT" if gl > 0 else "RIGHT"} and {gf:.2f} m ahead.')
    print('Robot pose (odom): ({:.3f}, {:.3f}, {:.3f})'.format(*robot_pose))
    print('Car (odom): centre=({:.3f}, {:.3f}) length={:.2f} rear_dir={:.1f} deg'.format(
        car.x, car.y, car.length, math.degrees(car.rear_yaw)))
    print('Rear bumper:', bumper)
    print('Goal (robot centre, facing the car):', goal)
    path = plan_astar(robot_pose[:2], goal[:2], car)
    for i, point in enumerate(path, 1):
        print('Waypoint {}: ({:.3f}, {:.3f})'.format(i, *point))
    if debug_image:
        save_debug_plot(debug_image, robot_pose, car, goal, bumper, path, est['points'])
    if not execute:
        print('Preview only. Check the debug plot / goal, then run with --execute.')
        return
    rclpy.init()
    controller = JackalController(cmd_topic)

    def spin():
        try:
            rclpy.spin(controller)
        except ExternalShutdownException:
            pass
    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            if (controller.publisher.get_subscription_count() and controller.get_pose() is not None
                    and controller.get_clock().now().nanoseconds > 0):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError('Missing controller subscriber, odometry or simulation clock. '
                               'Keep the clock bridge running.')
        navigate(controller, path, goal[2], car, backup)
    except KeyboardInterrupt:
        print('Interrupted; stopping.')
    finally:
        controller.stop()
        time.sleep(0.5)
        if rclpy.ok():
            rclpy.shutdown()
        thread.join(timeout=2)
        controller.destroy_node()


if __name__ == '__main__':
    scan_main()
