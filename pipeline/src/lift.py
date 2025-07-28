import numpy as np
import warnings
from pathlib import Path
from tqdm import tqdm

from mmpose.apis import PoseLifter

# --- Constants and Configuration ---
ROOT_DIR = Path(r"D:\Projects\STELLA\pipeline")
MODEL_DIR = ROOT_DIR / "assets" / "models" / "body3d"
OUTPUT_DIR = ROOT_DIR / "outputs"

# --- Input/Output Paths ---
KEYPOINTS_2D_PATH = OUTPUT_DIR / "keypoints_full.npz"
KEYPOINTS_3D_PATH = OUTPUT_DIR / "keypoints_3d_body.npz"

# --- Model Configuration ---
CONFIG_FILE = MODEL_DIR / "video-pose-lift_tcn-243frm-supv_8xb128-160e_h36m.py"
CHECKPOINT_FILE = MODEL_DIR / "videopose_h36m_243frames_fullconv_supervised-880bea25_20210527.pth"

# --- Constants ---
PELVIS_INDEX = 0  # Index of the pelvis/root keypoint for centering (Human3.6M format)


def load_and_validate_2d_keypoints(path: Path) -> np.ndarray:
    """Loads and validates the 2D keypoints from an .npz file."""
    print(f"[INFO] Loading 2D keypoints from: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Could not find 2D keypoint file: {path}")

    with np.load(path) as data:
        if 'body' not in data:
            raise KeyError("Required 'body' array not found in .npz file.")
        keypoints_2d = data['body']
        if keypoints_2d.ndim != 3 or keypoints_2d.shape[1] != 17 or keypoints_2d.shape[2] < 2:
            raise ValueError(f"Body 2D keypoints shape is invalid: {keypoints_2d.shape}")
    return keypoints_2d


def preprocess_2d_keypoints(keypoints_2d: np.ndarray) -> list:
    """Handles missing frames and prepares the data for the PoseLifter API."""
    is_frame_bad = np.all(keypoints_2d[..., :2] == 0, axis=-1).all(axis=-1)
    
    if np.all(is_frame_bad):
        raise ValueError("All frames contain zero keypoints. Cannot perform 3D lifting.")

    bad_frame_indices = np.where(is_frame_bad)[0]
    if len(bad_frame_indices) > 0:
        warnings.warn(
            f"{len(bad_frame_indices)} frames have no 2D keypoints. Filling with the last valid pose."
        )
        first_valid_idx = np.where(~is_frame_bad)[0][0]
        
        print("[INFO] Forward-filling missing frames...")
        for i in tqdm(bad_frame_indices, desc="Processing bad frames"):
            fill_idx = first_valid_idx if i == 0 else i - 1
            keypoints_2d[i] = keypoints_2d[fill_idx]
            
    sample = {
        'keypoints': keypoints_2d[..., :2].astype(np.float32),
        'keypoint_scores': keypoints_2d[..., 2].astype(np.float32)
    }
    return [sample]


def main():
    """Main function to run the 3D pose lifting pipeline."""
    try:
        # 1. Load and prepare 2D data
        keypoints_2d_raw = load_and_validate_2d_keypoints(KEYPOINTS_2D_PATH)
        processed_data = preprocess_2d_keypoints(keypoints_2d_raw)

        # 2. Initialize the PoseLifter
        print("[INFO] Initializing PoseLifter for MMPose v1.3.0...")
        pose_lifter = PoseLifter(
            config=str(CONFIG_FILE),
            checkpoint=str(CHECKPOINT_FILE),
            device='cuda:0'
        )

        # 3. Perform 3D inference
        print("[INFO] Lifting 2D keypoints to 3D...")
        pose_3d_generator = pose_lifter(processed_data)
        
        pose_3d_results = list(pose_3d_generator)
        
        keypoints_3d = pose_3d_results[0][0].pred_instances.keypoints

        # 4. Post-process the 3D data
        print("[INFO] Post-processing 3D keypoints...")
        pelvis_position = keypoints_3d[:, PELVIS_INDEX:PELVIS_INDEX+1, :]
        keypoints_3d_centered = keypoints_3d - pelvis_position

        # 5. Save the results
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            KEYPOINTS_3D_PATH,
            body3d_world=keypoints_3d,
            body3d_centered=keypoints_3d_centered,
        )
        print("\n--- 🚀 Success! ---")
        print(f"3D body keypoints saved to: {KEYPOINTS_3D_PATH}")
        print(f"    Shape: {keypoints_3d.shape} (Frames, Keypoints, Dims)")

    except Exception as e:
        print(f"\n--- ❌ ERROR ---")
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()