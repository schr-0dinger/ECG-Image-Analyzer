import cv2

__all__ = ['ROISelector']

class ROISelector:
    def __init__(self, image, window_name="Select ROI"):
        self.image = image.copy()
        self.clone = image.copy()
        self.window_name = window_name
        self.start_point = None
        self.current_point = None # Track current mouse position for real-time preview
        self.end_point = None
        self.drawing = False

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = (x, y)
            self.current_point = (x, y)
            self.drawing = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.end_point = (x, y)
            self.drawing = False

    def get_roi(self):
        while True:
            display_image = self.clone.copy()

            # Draw the rectangle preview while dragging or after completion
            if self.start_point and self.current_point:
                cv2.rectangle(display_image, self.start_point, self.current_point, (0, 255, 0), 2)

            cv2.putText(display_image, "CLICK + DRAG box, then press ENTER to confirm. ESC to cancel.",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow(self.window_name, display_image)
            key = cv2.waitKey(1) & 0xFF

            if key == 13:  # ENTER key
                if self.start_point and self.end_point:
                    x0, y0 = self.start_point
                    x1, y1 = self.end_point
                    w, h = abs(x1 - x0), abs(y1 - y0)
                    # Only return if the box has actual area
                    if w > 0 and h > 0:
                        cv2.destroyWindow(self.window_name)
                        return (min(x0, x1), min(y0, y1), w, h)
                
                # If invalid selection, show a reminder instead of returning None
                print("Warning: Please drag to select a valid area before pressing ENTER.")

            elif key == 27:  # ESC key
                cv2.destroyWindow(self.window_name)
                return None