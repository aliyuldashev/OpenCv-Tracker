# Object Tracking with OpenCV CSRT 🎯

[![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)](https://opencv.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## 👨🏻‍🎓 Javokhir Yuldoshev 12214760 

[![Video Thumbnail](https://img.youtube.com/vi/VIDEO_ID/0.jpg)](https://drive.google.com/file/d/1823YlenNyStZTw__1S37nhv4DmB0CLGf/view?usp=sharing)

A robust object tracking implementation using OpenCV's CSRT tracker with automatic redetection capabilities.

## Table of Contents

- [Tracker Module Comparison](#tracker-module-comparison)
- [Why CSRT?](#why-csrt)
- [Implementation Details](#implementation-details)
- [Features Explained](#features-explained)
- [Usage Guide](#usage-guide)
- [Showing demo](#performance-optimization)

## Tracker Module Comparison

OpenCV provides several tracking algorithms, each with unique characteristics:

### Speed-Focused Trackers ⚡

1. **MOSSE** (~450-700 FPS)
   - Fastest tracker
   - Minimal CPU usage
   - Limited accuracy

2. **KCF** (~150-250 FPS)
   - Good speed/accuracy balance
   - Efficient CPU usage
   - Handles partial occlusions

3. **MEDIANFLOW** (~100-120 FPS)
   - Good for predictable motion
   - Built-in failure detection
   - Limited in fast movements

### Accuracy-Focused Trackers 🎯

1. **CSRT** (~25-50 FPS)
   - Highest accuracy
   - Superior boundary adherence
   - Handles scale variations

2. **GOTURN** (~50-100 FPS with GPU)
   - Deep learning based
   - Requires GPU
   - Good for appearance changes

3. **TLD** (~20-25 FPS)
   - Long-term tracking
   - Recovery capabilities
   - Higher CPU usage

### Legacy Trackers 🔄

1. **BOOSTING** (~10-15 FPS)
   - Traditional approach
   - Online AdaBoost
   - Higher computational cost

2. **MIL** (~15-20 FPS)
   - Multiple instance learning
   - Better than BOOSTING
   - Slower processing

## Why CSRT ? 🤔

Our implementation uses CSRT (Channel and Spatial Reliability Tracker) for several key reasons:

1. **Superior Accuracy** 📊
   - Best-in-class boundary adherence
   - Precise object tracking
   - Robust scale adaptation

2. **Feature Rich** ✨
   - Multi-channel processing
   - Spatial reliability mapping
   - Advanced channel weighting

3. **Reliability** 💪
   - Handles partial occlusions
   - Deals with scale changes
   - Robust in various lighting conditions

4. **Real-world Performance** 🌟
   - Balanced speed/accuracy trade-off
   - Suitable for most applications
   - Reasonable resource usage

## Implementation Details 🛠️

### Core Components

```python
class ObjectTracker:
    def __init__(self):
        # Initialize with CSRT tracker
        self.tracker = cv2.TrackerCSRT_create()
        self.tracking = False
        self.template = None
```

### Key Features

1. **Automatic Tracker Selection**

```python
tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 
                'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[-1]  # CSRT selected
```

2. **ROI Selection**

```python
def select_roi(self, frame):
   bbox = cv2.selectROI("Select Object", frame, False)
   cv2.destroyWindow("Select Object")
   self.initial_frame = frame
   self.template = frame[int(bbox[1]):int(bbox[1]+bbox[3]), 
                        int(bbox[0]):int(bbox[0]+bbox[2])]
```

3. **Object Redetection**

```python
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
```

## Features Explained 🔍

### 1. Automatic Recovery System

- Implements template matching for redetection
- Maintains tracking continuity
- Handles temporary occlusions

### 2. Visual Feedback

```python
# Status indicators
cv2.putText(frame, "Tracking", (x, y - 10),
           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
# Lost object indication
cv2.putText(frame, "Searching...", (100, 50),
           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
```

### 3. State Management

- Tracks object state (tracking/lost)
- Seamless state transitions
- Robust failure recovery

### 4. Template-Based Redetection

- Stores initial object appearance
- Uses correlation-based matching
- Configurable confidence threshold

## Usage Guide 📚

### Basic Usage

```python
# Initialize tracker
tracker = ObjectTracker()

# Select ROI and initialize
bbox = tracker.select_roi(frame)
tracker.tracker.init(frame, bbox)

# Main tracking loop
while True:
    success, bbox = tracker.tracker.update(frame)
    if success:
        # Draw tracking results
        # ...
    else:
        # Attempt redetection
        # ...
```

### Advanced Configuration

```python
# Adjust redetection threshold
def redetect_object(self, frame, threshold=0.7):
    # Higher threshold = more strict matching
    if max_val > threshold:
        return True, bbox
    return False, None
```

## Performance Optimization 🚀

### 1. Frame Processing

- Consider frame skipping for higher FPS
- Implement ROI-based processing
- Use resolution scaling when needed

### 2. Redetection Strategy

```python
# Optimize redetection frequency
if not tracking and frame_count % redetection_interval == 0:
    success, bbox = tracker.redetect_object(frame)
```

### 3. Resource Management

- Implement frame buffer limits
- Clear template cache when needed
- Monitor memory usage

## Contributing 🤝

Feel free to contribute to this project by:

1. Forking the repository
2. Creating feature branches
3. Submitting pull requests

## License 📄

This project is licensed under the BSD 3-Clause License.

---

## Acknowledgments 🙏

- OpenCV community
- Computer Vision researchers
- Project contributors
