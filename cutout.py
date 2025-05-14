import cv2
import numpy as np

def apply_glaucoma_mask(frame, w=300, blur=20, darkness=0.0):
    """Apply a severe glaucoma-style mask: only a small oval is visible, rest is dark."""
    rows, cols = frame.shape[:2]
 
    mask = np.zeros((rows, cols), dtype=np.uint8)
    center_x, center_y = cols // 2, rows // 2

    # Ensure w:h is 3:2 ratio, approx FOV of human eye
    major_axis = w // 2
    minor_axis = int(2/3 * major_axis)
    # Create black eliptical mask
    cv2.ellipse(mask, (center_x, center_y), (major_axis, minor_axis), 0, 0, 360, 255, -1)

    # Blur the mask to create a smooth transition
    blurred_mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=blur, sigmaY=blur)

    # Normalize mask between 0 and 1
    blurred_mask = blurred_mask.astype(np.float32) / 255.0

    # Now correctly blend: center is normal, outside fades to black
    frame_blurred = (frame.astype(np.float32) * (darkness + blurred_mask[..., np.newaxis])).astype(np.uint8)

    return frame_blurred

def main():
    # Create const pairs for different glaucoma levels
    MILD_GLAUCOMA = (425, 50)
    MODERATE_GLAUCOMA = (250, 100)
    SEVERE_GLAUCOMA = (150, 150)
    EXTREME_GLAUCOMA = (80, 200)

    GLAUCOMA_LEVELS = {
        'MILD': MILD_GLAUCOMA,
        'MODERATE': MODERATE_GLAUCOMA,
        'SEVERE': SEVERE_GLAUCOMA, 
        'EXTREME': EXTREME_GLAUCOMA
    }

    level = 'MILD'
    
    # Open webcam: usually 0 is integrated, 1 for usb cam
    cap = cv2.VideoCapture(1)


    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame horizontally for a mirror effect (optional)
        frame = cv2.flip(frame, 1)
        
        # Apply glaucoma mask using pre-set lvl OR custom w and blur
        width, blur = GLAUCOMA_LEVELS[level]
        masked_frame = apply_glaucoma_mask(frame, width, blur)

        cv2.imshow('Glaucoma Simulation', masked_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
