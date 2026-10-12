import cv2
import mediapipe as mp
import numpy as np
import json
from datetime import datetime
import os

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
LAND = mp_pose.PoseLandmark

# ---------------------------
# smoothing (تنعيم القيم)
# ---------------------------
prev_angles = {}

def smooth(key, new_val, alpha=0.7):
    if key in prev_angles:
        val = alpha * prev_angles[key] + (1 - alpha) * new_val
    else:
        val = new_val
    prev_angles[key] = val
    return val

# ---------------------------
# حساب الزاوية
# ---------------------------
def angle_between(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle = np.degrees(np.arccos(cos_angle))
    return angle

def landmark_to_xy(landmark, w, h):
    return (landmark.x * w, landmark.y * h, landmark.z * w)

NORMAL_RANGES = {
    "elbow": (10, 160),
    "knee": (10, 160),
    "shoulder": (10, 180),
    "hip": (10, 130),
    "ankle": (0, 50)
}

def check_joint(name, angle, other_angle=None):
    min_a, max_a = NORMAL_RANGES.get(name, (0, 180))
    issue = False
    reasons = []

    if angle < min_a or angle > max_a:
        issue = True
        reasons.append(f"Out of range [{min_a},{max_a}]")

    if other_angle is not None:
        if abs(angle - other_angle) > 20:
            issue = True
            reasons.append("Asymmetry > 20°")

    return issue, reasons

# ---------------------------
# MAIN
# ---------------------------
def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera error")
        return

    os.makedirs("reports", exist_ok=True)

    with mp_pose.Pose(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            report = {
                "time": datetime.now().isoformat(),
                "joints": {},
                "flags": []
            }

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                # فلترة النقاط الضعيفة
                if lm[LAND.LEFT_ELBOW.value].visibility < 0.5:
                    continue

                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                points = {}
                for name in LAND:
                    pt = lm[name.value]
                    points[name.name] = landmark_to_xy(pt, w, h)

                joints = {}

                # ELBOW
                joints['left_elbow'] = smooth('left_elbow',
                    angle_between(points['LEFT_SHOULDER'], points['LEFT_ELBOW'], points['LEFT_WRIST']))
                joints['right_elbow'] = smooth('right_elbow',
                    angle_between(points['RIGHT_SHOULDER'], points['RIGHT_ELBOW'], points['RIGHT_WRIST']))

                # KNEE
                joints['left_knee'] = smooth('left_knee',
                    angle_between(points['LEFT_HIP'], points['LEFT_KNEE'], points['LEFT_ANKLE']))
                joints['right_knee'] = smooth('right_knee',
                    angle_between(points['RIGHT_HIP'], points['RIGHT_KNEE'], points['RIGHT_ANKLE']))

                # SHOULDER
                joints['left_shoulder'] = smooth('left_shoulder',
                    angle_between(points['LEFT_HIP'], points['LEFT_SHOULDER'], points['LEFT_ELBOW']))
                joints['right_shoulder'] = smooth('right_shoulder',
                    angle_between(points['RIGHT_HIP'], points['RIGHT_SHOULDER'], points['RIGHT_ELBOW']))

                # HIP
                joints['left_hip'] = smooth('left_hip',
                    angle_between(points['LEFT_SHOULDER'], points['LEFT_HIP'], points['LEFT_KNEE']))
                joints['right_hip'] = smooth('right_hip',
                    angle_between(points['RIGHT_SHOULDER'], points['RIGHT_HIP'], points['RIGHT_KNEE']))

                # ANKLE (fix مهم)
                if 'LEFT_FOOT_INDEX' in points:
                    joints['left_ankle'] = smooth('left_ankle',
                        angle_between(points['LEFT_KNEE'], points['LEFT_ANKLE'], points['LEFT_FOOT_INDEX']))
                if 'RIGHT_FOOT_INDEX' in points:
                    joints['right_ankle'] = smooth('right_ankle',
                        angle_between(points['RIGHT_KNEE'], points['RIGHT_ANKLE'], points['RIGHT_FOOT_INDEX']))

                report['joints'] = {k: round(v, 2) for k, v in joints.items()}

                # CHECK
                pairs = [
                    ("elbow", "left_elbow", "right_elbow"),
                    ("knee", "left_knee", "right_knee"),
                    ("shoulder", "left_shoulder", "right_shoulder"),
                    ("hip", "left_hip", "right_hip"),
                    ("ankle", "left_ankle", "right_ankle"),
                ]

                for name, l, r in pairs:
                    if l in joints and r in joints:
                        issue_l, reason_l = check_joint(name, joints[l], joints[r])
                        issue_r, reason_r = check_joint(name, joints[r], joints[l])

                        if issue_l:
                            report['flags'].append({"joint": l, "reasons": reason_l})
                        if issue_r:
                            report['flags'].append({"joint": r, "reasons": reason_r})

                # عرض
                y = 20
                for k, v in report['joints'].items():
                    cv2.putText(frame, f"{k}: {v:.1f}", (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                    y += 18

                if report['flags']:
                    cv2.putText(frame, "Issues detected!", (10, h-20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            else:
                cv2.putText(frame, "No person", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            cv2.imshow("Pose Check", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            if key == ord('r'):
                name = f"reports/report_{datetime.now().strftime('%H%M%S')}.json"
                with open(name, "w") as f:
                    json.dump(report, f, indent=2)
                print("Saved:", name)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()