from mmpose.apis.inference import init_model as init_pose_model # pyright: ignore[reportMissingImports]

# This script verifies various pose estimation models loading or not using MMPose.

# Body 2D Model
cfg = r'D:\Projects\STELLA\pipeline\assets\models\body2d\rtmpose-l_8xb256-420e_aic-coco-384x288.py'
ckpt = r'D:\Projects\STELLA\pipeline\assets\models\body2d\rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-384x288-97d6cb0f_20230228.pth'
model = init_pose_model(cfg, ckpt, device='cuda:0')
print("2D model loaded successfully.")


# Body 3D Model
cfg_body3d = r'D:\Projects\STELLA\pipeline\assets\models\body3d\video-pose-lift_tcn-243frm-supv_8xb128-160e_h36m.py'
ckpt_body3d = r'D:\Projects\STELLA\pipeline\assets\models\body3d\videopose_h36m_243frames_fullconv_supervised-880bea25_20210527.pth'
body3d_model = init_pose_model(cfg_body3d, ckpt_body3d, device='cuda:0')
print("✅ Body 3D model loaded successfully.")

# Face Pose Model
cfg_face = r'D:\Projects\STELLA\pipeline\assets\models\face\rtmpose-m_8xb256-120e_face6-256x256.py'
ckpt_face = r'D:\Projects\STELLA\pipeline\assets\models\face\rtmpose-m_simcc-face6_pt-in1k_120e-256x256-72a37400_20230529.pth'
face_model = init_pose_model(cfg_face, ckpt_face, device='cuda:0')
print("✅ Face model loaded successfully.")

# === Hand Pose Model ===
cfg_hand = r'D:\Projects\STELLA\pipeline\assets\models\hand\rtmpose-m_8xb256-210e_hand5-256x256.py'
ckpt_hand = r'D:\Projects\STELLA\pipeline\assets\models\hand\rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth'
hand_model = init_pose_model(cfg_hand, ckpt_hand, device='cuda:0')
print("✅ Hand model loaded successfully.")