# Object Tracking with OpenCV CSRT 🎯

[![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)](https://opencv.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## 👨🏻‍🎓 Javokhir Yuldoshev 12214760 

https://github.com/user-attachments/assets/84a9f0d3-fef5-4e9c-a43e-c8077f3c0b01

>[Reminder]
> You can find code and implementation explanation video by clicking this [link](https://drive.google.com/file/d/1823YlenNyStZTw__1S37nhv4DmB0CLGf/view?usp=sharing) or clicking bellow `video button`

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

### 4. Automatic Recovery System

- Implements template matching for redetection
- Maintains tracking continuity
- Handles temporary occlusions

### 5. Visual Feedback

```python
# Status indicators
cv2.putText(frame, "Tracking", (x, y - 10),
           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
# Lost object indication
cv2.putText(frame, "Searching...", (100, 50),
           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
```

### 6. State Management

- Tracks object state (tracking/lost)
- Seamless state transitions
- Robust failure recovery

### 7. Template-Based Redetection

- Stores initial object appearance
- Uses correlation-based matching
- Configurable confidence threshold

## Usage Guide 📚

### 1. Clone repository

>[Note] 
> You have A~C options to clone repository

### A. Cloning with HTTPS

Clone with HTTPS with `git` command

```bash
git clone https://github.com/aliyuldashev/OpenCv-Tracker.git
```

### B. Cloning with SSH

Clone with SSH with `git` command in terminal

```bash
git clone git@github.com:aliyuldashev/OpenCv-Tracker.git
```

### C. Download Archive file of repository

Download Archive file of repository in github by this [link](https://github.com/aliyuldashev/OpenCv-Tracker/archive/refs/heads/main.zip). Then Unarchive it with tool which you have.

### 2. Installing requirements

To run code first install need requirements by running this commands in terminal open in this directory

```bash
pip install -r requirements.txt
```

### 3. Running code

To following commands in that terminal

```bash
python -m tracker.py
```

## Contributing 🤝

Feel free to contribute to this project by:

1. Forking the repository
2. Creating feature branches
3. Submitting pull requests

---

## Acknowledgments 🙏

- OpenCV community
- Computer Vision researchers
- Project contributors
