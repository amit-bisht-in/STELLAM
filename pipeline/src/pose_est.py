import os
import cv2
import numpy as np
from mmpose.apis import init_model
from mmpose.apis.inference import inference_topdown

# ---- 1. Edit these to your config and checkpoint filenames ----
body_cfg = r'D:\Projects\STELLA\pipeline\assets\models\body2d\rtmpose-l_8xb256-420e_aic-coco-384x288.py'
body_ckpt = r'D:\Projects\STELLA\pipeline\assets\models\body2d\rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-384x288-97d6cb0f_20230228.pth'
face_cfg = r'D:\Projects\STELLA\pipeline\assets\models\face\rtmpose-m_8xb256-120e_face6-256x256.py'
face_ckpt = r'D:\Projects\STELLA\pipeline\assets\models\face\rtmpose-m_simcc-face6_pt-in1k_120e-256x256-72a37400_20230529.pth'
hand_cfg = r'D:\Projects\STELLA\pipeline\assets\models\hand\rtmpose-m_8xb256-210e_hand5-256x256.py'
hand_ckpt = r'D:\Projects\STELLA\pipeline\assets\models\hand\rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth'
video_path = r"assets/videos_input/your_input.mp4"
output_path = "assets/videos_input/keypoints_full.npz"

# ---- 2. Helper functions for cropping face and hands from body keypoints ----
def get_face_bbox(body_keypoints, rescale=1.5):
    # This function remains unchanged
    face_points = body_keypoints[[0, 1, 2, 3, 4], :2]
    x_min, y_min = np.min(face_points, axis=0)
    x_max, y_max = np.max(face_points, axis=0)
    w, h = x_max - x_min, y_max - y_min
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    w, h = w * rescale, h * rescale
    return [int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2)]

def get_hand_bbox(body_keypoints, side='left', rescale=2.0):
    # This function remains unchanged
    if side == 'left':
        wrist, elbow = 9, 7
    else:
        wrist, elbow = 10, 8
    wrist_pt = body_keypoints[wrist, :2]
    elbow_pt = body_keypoints[elbow, :2]
    hand_vec = wrist_pt - elbow_pt
    hand_length = np.linalg.norm(hand_vec) * rescale
    cx, cy = wrist_pt
    return [int(cx - hand_length/2), int(cy - hand_length/2),
            int(cx + hand_length/2), int(cy + hand_length/2)]

# ---- 3. Load all models (on GPU) ----
print("[INFO] Loading pose models...")
body_model = init_model(body_cfg, body_ckpt, device='cuda:0')
face_model = init_model(face_cfg, face_ckpt, device='cuda:0')
hand_model = init_model(hand_cfg, hand_ckpt, device='cuda:0')

# ---- 4. Open video or launch camera ----
is_live_camera = False
if os.path.exists(video_path):
    print(f"[INFO] Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Total frames: {frame_count}")
else:
    print(f"[INFO] Video not found. Launching webcam...")
    cap = cv2.VideoCapture(0)
    is_live_camera = True
    frame_count = float('inf')

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video source.")

all_results = []

while len(all_results) < frame_count:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    
    # BODY
    body_bbox = [0, 0, w, h]
    # **FIX 1: Changed bbox format**
    pose_results = inference_topdown(
        body_model, frame, np.array([body_bbox]), bbox_format='xyxy'
    )
    body_kps = pose_results[0].get('keypoints', np.zeros((17,3)))

    # FACE
    face_bbox = get_face_bbox(body_kps)
    fx1, fy1, fx2, fy2 = np.clip(face_bbox, [0,0,0,0], [w-1,h-1,w-1,h-1])
    if fy2 > fy1 and fx2 > fx1:
        face_crop = frame[fy1:fy2, fx1:fx2]
        face_bbox_in_crop = np.array([[0, 0, fx2 - fx1, fy2 - fy1]])
        # **FIX 2: Changed bbox format**
        face_result = inference_topdown(
            face_model, face_crop, face_bbox_in_crop, bbox_format='xyxy'
        )
        face_kps = face_result[0].get('keypoints', np.zeros((68,3)))
        if face_kps.any(): face_kps[:,:2] += [fx1, fy1]
    else:
        face_kps = np.zeros((68,3))

    # LEFT HAND
    lh_bbox = get_hand_bbox(body_kps, 'left')
    lx1, ly1, lx2, ly2 = np.clip(lh_bbox, [0,0,0,0], [w-1,h-1,w-1,h-1])
    if ly2 > ly1 and lx2 > lx1:
        lhand_crop = frame[ly1:ly2, lx1:lx2]
        lhand_bbox_in_crop = np.array([[0, 0, lx2 - lx1, ly2 - ly1]])
        # **FIX 3: Changed bbox format**
        left_hand_result = inference_topdown(
            hand_model, lhand_crop, lhand_bbox_in_crop, bbox_format='xyxy'
        )
        lhand_kps = left_hand_result[0].get('keypoints', np.zeros((21,3)))
        if lhand_kps.any(): lhand_kps[:,:2] += [lx1, ly1]
    else:
        lhand_kps = np.zeros((21,3))

    # RIGHT HAND
    rh_bbox = get_hand_bbox(body_kps, 'right')
    rx1, ry1, rx2, ry2 = np.clip(rh_bbox, [0,0,0,0], [w-1,h-1,w-1,h-1])
    if ry2 > ry1 and rx2 > rx1:
        rhand_crop = frame[ry1:ry2, rx1:rx2]
        rhand_bbox_in_crop = np.array([[0, 0, rx2 - rx1, ry2 - ry1]])
        # **FIX 4: Changed bbox format**
        right_hand_result = inference_topdown(
            hand_model, rhand_crop, rhand_bbox_in_crop, bbox_format='xyxy'
        )
        rhand_kps = right_hand_result[0].get('keypoints', np.zeros((21,3)))
        if rhand_kps.any(): rhand_kps[:,:2] += [rx1, ry1]
    else:
        rhand_kps = np.zeros((21,3))

    all_results.append({
        'body': body_kps,
        'face': face_kps,
        'left_hand': lhand_kps,
        'right_hand': rhand_kps,
    })

    if is_live_camera:
        cv2.imshow('Live Pose Estimation - Press Q to Quit', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if len(all_results) % 50 == 0:
        progress_msg = f"[INFO] Processed {len(all_results)}"
        if not is_live_camera:
            progress_msg += f"/{frame_count}"
        progress_msg += " frames"
        print(progress_msg)


cap.release()
cv2.destroyAllWindows()
print("[INFO] Processing finished.")

# ---- 5. Save results if any frames were processed ----
if all_results:
    print(f"[INFO] Saving {len(all_results)} frames of keypoints...")
    np.savez_compressed(
        output_path,
        body=np.array([r['body'] for r in all_results]),
        face=np.array([r['face'] for r in all_results]),
        left_hand=np.array([r['left_hand'] for r in all_results]),
        right_hand=np.array([r['right_hand'] for r in all_results]),
    )
    print(f"[INFO] Keypoints saved to {output_path}")
else:
    print("[INFO] No frames were processed. Nothing to save.")