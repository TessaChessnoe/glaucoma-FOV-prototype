import cv2
import numpy as np
from glaucomaFilter import visionFilter

def cam_stream_init(w=640, h=480, cam_index = 0):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(w))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(h))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera #{cam_index}")
    return cap

def main():
    print("Starting program...")
    SEVERITY_LEVELS = {
        'MILD': 1,
        'MODERATE': 2,
        'SEVERE': 3, 
        'EXTREME': 4
    }
    resolution = [640, 480]
    # 0 is external webcam w DSHOW
    cap = cam_stream_init(resolution[0], resolution[1], 0)

    # Initialize filter & Pre-compute mask
    severity = SEVERITY_LEVELS['MILD']
    filter = visionFilter((resolution[0], resolution[1]), 'nasal-step', severity)
    mask = filter.build_mask(severity)
    # Normalize & reshape mask
    mask = (mask.astype('float32') / 255.0)
    mask = mask[..., np.newaxis]

    while True:
        ret, frame = cap.read()
        # Apply mask 
        frame = (frame.astype(np.float32) * mask).astype(np.uint8)
        # Create stereo vision channels
        left = frame
        right = cv2.flip(left.copy(), 1)
        stereo = np.concatenate([left, right], axis=1)
        # Display masked stereo video
        cv2.imshow('Glaucoma Simulation', stereo)
        # Close video feed with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return

if __name__ == "__main__":
    main()