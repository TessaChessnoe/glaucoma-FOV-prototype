import cv2

def list_cameras(max_index=10, backends=None):
    """
    Scans camera indexes from 0 to max_index-1, trying each backend.
    Returns a list of (index, backend_name) tuples that successfully opened.
    """
    if backends is None:
        backends = {
            'ANY':    cv2.CAP_ANY,
            'DSHOW':  cv2.CAP_DSHOW,
            'MSMF':   cv2.CAP_MSMF
        }

    found = []
    for i in range(max_index):
        for name, api in backends.items():
            cap = cv2.VideoCapture(i, api)
            if cap.isOpened():
                found.append((i, name))
                cap.release()
                break  # don’t try other backends once this index works
            cap.release()
    return found

if __name__ == "__main__":
    cams = list_cameras(max_index=8)
    if cams:
        print("Detected cameras:")
        for idx, backend in cams:
            print(f"  • Index {idx} (via {backend})")
    else:
        print("No cameras found in 0–7.")