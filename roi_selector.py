import cv2

class ROISelector:
    def __init__(self, image, window_name="Select ROI"):
        self.image = image.copy()
        self.clone = image.copy()
        self.window_name = window_name
        self.start_point = None
        self.end_point = None
        self.drawing = False
        self.roi_done = False

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = (x, y)
            self.drawing = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.end_point = (x, y)
            self.drawing = False
            self.roi_done = True

    def get_roi(self):
        while True:
            display_image = self.clone.copy()

            if self.start_point and self.end_point:
                cv2.rectangle(display_image, self.start_point, self.end_point, (0, 255, 0), 2)

            cv2.putText(display_image, "Drag and press ENTER to confirm. ESC to cancel.",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow(self.window_name, display_image)
            key = cv2.waitKey(1) & 0xFF

            if key == 13:  # ENTER key
                if self.start_point and self.end_point:
                    x0, y0 = self.start_point
                    x1, y1 = self.end_point
                    return (min(x0, x1), min(y0, y1), abs(x1-x0), abs(y1-y0))
                else:
                    return None

            elif key == 27:  # ESC
                return None

