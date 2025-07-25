import cv2
import os
import argparse

def get_video_capture(filename=None, source=0):
    """
    Opens a video file (if filename is given and exists), else opens the webcam.
    Returns a cv2.VideoCapture object.
    """
    if filename is not None and os.path.isfile(filename):
        cap = cv2.VideoCapture(filename)
        if cap.isOpened():
            print(f"[INFO] Opened video file: {filename}")
            return cap
        else:
            print(f"[WARN] Cannot open video file: {filename}. Falling back to webcam.")
    # Use webcam as fallback
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot access webcam (source={source}) either.")
    print(f"[INFO] Opened webcam on source {source}")
    return cap

def release_capture(capture):
    """Release the video resource and all windows."""
    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video input handler')
    parser.add_argument('--video', type=str, default=None, help='Path to input video file')
    args = parser.parse_args()

    cap = get_video_capture(filename=args.video)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video or cannot grab frame.")
            break
        cv2.imshow("Input Video", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    release_capture(cap)
