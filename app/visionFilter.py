import cv2
import numpy as np

class visionFilter:
    def __init__(self, size, preset, severity):
        self.preset = preset
        self.severity = severity
        self.w, self.h = size

    def build_mask(self, severity):
        w,h = self.w, self.h
        severity = self.severity
        # Create eye cutout to mimic human sight
        mask = np.zeros((h, w), dtype=np.uint8)  # Full visibility base
        fov_axes = (int(w*0.9), int(h*0.6))
        cv2.ellipse(mask, (w//2, h//2), fov_axes, 0, 0, 360, 255, 40)
        mask = 255 - mask # Invert to create visibility inside mask
        if self.preset == 'normal':
            pass
        elif self.preset == 'glaucoma':
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
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        elif self.preset == 'arcuate':
            # Arcuate nasal step (arc-shaped defect)
            angle_start = 180+22.5  # Left side of nasal superior quadrant
            angle_end = 315 # Right side of nasal superior quadrant
            center = (w//2, h//3) # Arc at top-center
            axes = (w//2, h//6)
            # Scale arc thickness according to resolution
            arc_thickness = int(40 + 0.045 * w * severity) # 4.5% is empirical val found through testing
            cv2.ellipse(mask, center, axes, 
                   0, angle_start, angle_end, 0, arc_thickness)
        elif self.preset == 'scotoma':
            severity_factor = 1.05
            mask = np.zeros((h,w), dtype=np.uint8)
            # Placeholder pos (upper third)
            center = (w//3, h//3)
            # hardcoded for small scotoma
            radius = w//15
            cv2.circle(mask, center, radius, 255, -1)
        else:
            raise Exception("Invalid argument: Please specify a vision loss type listed in documentation. ")
        return mask

    