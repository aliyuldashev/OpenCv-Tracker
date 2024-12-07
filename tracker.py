import cv2 ,time
# import numpy as np

class ObjectTracker:
    def __init__(self):
        tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
        tracker_type = tracker_types[-1]

        if tracker_type == 'BOOSTING':
            tracker = cv2.legacy.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create() 
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create() 
        if tracker_type == 'TLD':
            tracker = cv2.legacy.TrackerTLD_create() 
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.legacy.TrackerMedianFlow_create() 
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.legacy.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()
        self.tracker = tracker
        self.initial_frame = None
        self.template = None
        self.bbox = None
        self.tracking = False
        
    def select_roi(self, frame):
        bbox = cv2.selectROI("Select Object", frame, False)
        cv2.destroyWindow("Select Object")
        self.initial_frame = frame
        self.template = frame[int(bbox[1]):int(bbox[1]+bbox[3]), 
                            int(bbox[0]):int(bbox[0]+bbox[2])]
        return bbox
    
    def redetect_object(self, frame):
        result = cv2.matchTemplate(frame, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        if max_val > 0.7:  # Threshold for match confidence
            w, h = self.template.shape[1], self.template.shape[0]
            bbox = (max_loc[0], max_loc[1], w, h)
            self.tracker.init(frame, bbox)
            self.tracking = True
            return True, bbox
        return False, None

def main(source:str):
    # Change the path here for a video file
     # Example video file
    
    #  input source
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        raise RuntimeError("Failed to open video source.") 
    
    tracker = ObjectTracker()
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video frame")
        raise RuntimeError("Failed to ead video frame.")
        
    bbox = tracker.select_roi(frame)
    tracker.tracker.init(frame, bbox)
    tracker.tracking = True
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if tracker.tracking:
            success, bbox = tracker.tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Tracking", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                tracker.tracking = False
        
        else:  # Lost tracking
            success, bbox = tracker.redetect_object(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Re-detected", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            else:
                cv2.putText(frame, "Searching...", (100, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    first = "assets/same_instance.mp4"
    second = "assets/small_instance.mp4"
    user = input("Choose option to process \n1 for same_instance \n2 for small_instance to choose the source:\ninput here: ")
    if user == '1':
        start = time.time()
        main(first)
        end = time.time()
        print(f"\nTotal processing time: {end - start:.2f} seconds")
    elif user == '2':
        start = time.time()
        main(second)
        end = time.time()
        print(f"\nTotal processing time: {end - start:.2f} seconds")
else:
        print("Invalid input. Exiting...")

