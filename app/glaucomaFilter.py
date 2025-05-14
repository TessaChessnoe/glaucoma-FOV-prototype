import cv2
import numpy as np

class visionFilter:
    def __init__(self, size, disease, severity):
        self.disease = disease
        self.severity = severity
        self.w, self.h = size

    def build_mask(self, severity):
        w,h = self.w, self.h
        
        # Create eye cutouts to mimic human sight
        mask = np.zeros((h,w), dtype=np.uint8)
        center = (w//2, h//2)
        major_axis = int(11*w/22)
        # Maintain approx 2:3 ratio of w:h for real eye
        minor_axis = int(major_axis * 2 / 3)
        axes = major_axis, minor_axis
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

        if self.disease == 'glaucoma':
            # Decrease FOV by 19% each severity level
            severity_factor = 0.19
            #mask = np.zeros((h,w), dtype=np.uint8)
            center = (w//2, h//2)
            # Severity controls tunnel width 
            major_axis = int(2*w/3 - (severity * severity_factor * w))
            # Maintain approx 2:3 ratio of w:h for real eye
            minor_axis = int(major_axis * 2 / 3)
            axes = major_axis, minor_axis
            # Draw an entire (0-360) filled (-1) ellipse where vision remains in center (on/255)
            return cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        elif self.disease == 'nasal-step':
            #mask = np.ones((h, w), dtype=np.uint8) * 255
            nose_height = 125
            nose_width = 200
            step_width = 135
            # Determines center of nose (full blind spot)
            center = (w, 7*h//8)

            # Step triangle: partial vision loss 50%
            step_p1 = [w, center[1] - nose_height//2 - step_width] # Top vertex
            step_p2 = [w - nose_width//2 - step_width, center[1] + nose_height//2 + step_width] # Left vertex
            step_p3 = [w, center[1] + nose_height//2 + step_width] # Bottom vertex
            step_triangle = np.array([step_p1, step_p2, step_p3])
            # Apply triangular cutout to mask
            cv2.fillPoly(mask, [step_triangle], 128)

            # Nose triangle: full vision loss 0%
            nose_p1 = [w, center[1] - nose_height//2] # Top vertex
            nose_p2 = [w - nose_width//2, center[1] + nose_height // 2] # Left vertex
            nose_p3 = [w, center[1] + nose_height//2] # Bottom vertex
            nose_triangle = np.array([nose_p1, nose_p2, nose_p3])
            # Apply triangular cutout to mask
            cv2.fillPoly(mask, [nose_triangle], 0)
            return mask
        elif self.disease == 'scotoma':
            severity_factor = 1.05
            mask = np.zeros((h,w), dtype=np.uint8)
            # Placeholder pos (upper third)
            center = (w//3, h//3)
            # hardcoded for small scotoma
            radius = w//15
            return cv2.circle(mask, center, radius, 255, -1)
        elif self.disease == 'depression':
            mask = np.ones((h, w), dtype=np.uint8) * 255
            outer_axes = (int(self.severity * w/6), int(self.severity * h/6))
            inner_axes = (int(self.severity * w/10), int(self.severity * h/10))
            center = (w//2, h//2)
            # Outer ellipse region: set to partial loss (e.g. 128)
            cv2.ellipse(mask, center, outer_axes, 0, 0, 360, 128, -1)
            # Inner ellipse region: set to full loss (0)
            cv2.ellipse(mask, center, inner_axes, 0, 0, 360,   0, -1)
            return mask
        else:
            raise Exception("Invalid argument: Please specify a vision loss type listed in documentation. ")

    