import os
import cv2
import numpy as np
from mmpose.apis import init_model, inference_topdown

# ---- 1. Your config and checkpoint filenames ----
body_cfg = r'D:\Projects\STELLA\pipeline\assets\models\body2d\rtmpose-l_8xb256-420e_aic-coco-384x288.py'
body_ckpt = r'D:\Projects\STELLA\pipeline\assets\models\body2d\rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-384x288-97d6cb0f_20230228.pth'
face_cfg = r'D:\Projects\STELLA\pipeline\assets\models\face\rtmpose-m_8xb256-120e_face6-256x256.py'
face_ckpt = r'D:\Projects\STELLA\pipeline\assets\models\face\rtmpose-m_simcc-face6_pt-in1k_120e-256x256-72a37400_20230529.pth'
hand_cfg = r'D:\Projects\STELLA\pipeline\assets\models\hand\rtmpose-m_8xb256-210e_hand5-256x256.py'
hand_ckpt = r'D:\Projects\STELLA\pipeline\assets\models\hand\rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth'
video_path = r"D:\Projects\STELLA\pipeline\assets\video_input\video1.mp4"
output_path = r"D:\Projects\STELLA\pipeline\outputs\keypoints_full.npz"

# ---- 2. Helper functions for cropping ----
def get_face_bbox(body_keypoints, rescale=1.5):
    face_points = body_keypoints[[0, 1, 2, 3, 4], :2]
    if np.all(face_points == 0):
        return None
    x_min, y_min = np.min(face_points, axis=0)
    x_max, y_max = np.max(face_points, axis=0)
    w, h = x_max - x_min, y_max - y_min
    if w <= 0 or h <= 0: return None
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    w, h = w * rescale, h * rescale
    return [int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2)]

def get_hand_bbox(body_keypoints, side='left', rescale=2.0):
    wrist_idx, elbow_idx = (9, 7) if side == 'left' else (10, 8)
    if np.all(body_keypoints[wrist_idx, :2] == 0) or np.all(body_keypoints[elbow_idx, :2] == 0):
        return None
    wrist_pt, elbow_pt = body_keypoints[wrist_idx, :2], body_keypoints[elbow_idx, :2]
    if np.array_equal(wrist_pt, elbow_pt): return None
    hand_length = np.linalg.norm(wrist_pt - elbow_pt) * rescale
    if hand_length <= 0: return None
    cx, cy = wrist_pt
    return [int(cx - hand_length/2), int(cy - hand_length/2), int(cx + hand_length/2), int(cy + hand_length/2)]

# --- Helper to extract keypoints from PoseDataSample ---
def extract_keypoints(pose_results, num_keypoints):
    if not pose_results or len(pose_results) == 0:
        return np.zeros((num_keypoints, 3))
    
    pred_instances = pose_results[0].pred_instances
    keypoints = pred_instances.keypoints[0]  # Shape (num_keypoints, 2)
    scores = pred_instances.keypoint_scores[0] # Shape (num_keypoints,)
    
    # Combine keypoints and scores into (num_keypoints, 3)
    return np.hstack((keypoints, scores[:, None]))

# ---- 3. Load all models ----
print("[INFO] Loading pose models...")
body_model = init_model(body_cfg, body_ckpt, device='cuda:0')
face_model = init_model(face_cfg, face_ckpt, device='cuda:0')
hand_model = init_model(hand_cfg, hand_ckpt, device='cuda:0')

# ---- 4. Open video and process ----
cap = cv2.VideoCapture(video_path)
if not cap.isOpened(): raise RuntimeError(f"Cannot open video: {video_path}")
all_results = []
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"[INFO] Processing video: {video_path} ({frame_count} frames)")

while True:
    ret, frame = cap.read()
    if not ret: break
    h, w, _ = frame.shape
    
    # **FIX 1: Use the correct input format [ [x,y,x,y] ]**
    body_bbox = [[0, 0, w, h]]
    pose_results = inference_topdown(body_model, frame, body_bbox, bbox_format='xyxy')
    # **FIX 2: Use helper to correctly parse the PoseDataSample output**
    body_kps = extract_keypoints(pose_results, 17)

    face_kps, lhand_kps, rhand_kps = np.zeros((68,3)), np.zeros((21,3)), np.zeros((21,3))

    if np.any(body_kps > 0):
        # FACE
        face_bbox = get_face_bbox(body_kps)
        if face_bbox:
            fx1, fy1, fx2, fy2 = np.clip(face_bbox, [0,0,0,0], [w-1,h-1,w-1,h-1])
            if fy2 > fy1 and fx2 > fx1:
                face_crop = frame[fy1:fy2, fx1:fx2]
                face_results = inference_topdown(face_model, face_crop, [[0,0,face_crop.shape[1],face_crop.shape[0]]], bbox_format='xyxy')
                face_kps = extract_keypoints(face_results, 68)
                face_kps[:,:2] += [fx1, fy1]

        # LEFT HAND
        lh_bbox = get_hand_bbox(body_kps, 'left')
        if lh_bbox:
            lx1, ly1, lx2, ly2 = np.clip(lh_bbox, [0,0,0,0], [w-1,h-1,w-1,h-1])
            if ly2 > ly1 and lx2 > lx1:
                lhand_crop = frame[ly1:ly2, lx1:lx2]
                lhand_results = inference_topdown(hand_model, lhand_crop, [[0,0,lhand_crop.shape[1],lhand_crop.shape[0]]], bbox_format='xyxy')
                lhand_kps = extract_keypoints(lhand_results, 21)
                lhand_kps[:,:2] += [lx1, ly1]

        # RIGHT HAND
        rh_bbox = get_hand_bbox(body_kps, 'right')
        if rh_bbox:
            rx1, ry1, rx2, ry2 = np.clip(rh_bbox, [0,0,0,0], [w-1,h-1,w-1,h-1])
            if ry2 > ry1 and rx2 > rx1:
                rhand_crop = frame[ry1:ry2, rx1:rx2]
                rhand_results = inference_topdown(hand_model, rhand_crop, [[0,0,rhand_crop.shape[1],rhand_crop.shape[0]]], bbox_format='xyxy')
                rhand_kps = extract_keypoints(rhand_results, 21)
                rhand_kps[:,:2] += [rx1, ry1]

    all_results.append({'body': body_kps, 'face': face_kps, 'left_hand': lhand_kps, 'right_hand': rhand_kps})
    if len(all_results) % 50 == 0: print(f"[INFO] Processed {len(all_results)}/{frame_count} frames")

cap.release()
print("[INFO] Processing finished.")

# ---- 5. Save results ----
output_dir = os.path.dirname(output_path)
if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)
np.savez_compressed(output_path, **{k: np.array([r[k] for r in all_results]) for k in all_results[0]})
print(f"[INFO] Keypoints saved to {output_path}")