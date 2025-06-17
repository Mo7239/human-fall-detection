# 🧍‍♂️ Human Fall Detection using YOLO

This project uses the YOLO (You Only Look Once) object detection model to detect whether a person in a video has fallen or is standing.

---

## 📽️ Overview

- The system loads a custom YOLO model (`yolo11m.pt`) trained to detect people.
- It processes a given input video, detects people in each frame, and checks if they are **standing** or **fallen** based on the shape of the bounding box.
- Results are saved in a new video file with bounding boxes and labels drawn on each person detected.

---

## 🧠 How Fall is Detected?

We determine if a person has fallen based on the **aspect ratio** of the bounding box:
- If width > height → considered **fallen**
- Else → considered **standing**

---

## 📁 Folder Structure

```
human-fall-detection/
│
├── main.py                      # Main script
├── fall_detection_output.mp4    # Example output video
├── README.md                    # This file
├── yolo11m.pt                   # Your custom YOLO model (not pushed to GitHub)
├── results/                     # Folder to save output videos
└── videos and images/
     └── fall.mp4                # Input video to process
```

---

## 🧪 Requirements

- Python 3.8+
- OpenCV
- [Ultralytics YOLO](https://docs.ultralytics.com/)

Install requirements:

```bash
pip install opencv-python ultralytics
```

---

## ▶️ How to Run

1. Clone this repo:
```bash
git clone https://github.com/Mo7239/human-fall-detection.git
cd human-fall-detection
```

2. Adjust paths in `main.py` if necessary:
```python
model_path = 'path/to/yolo11m.pt'
video_path = 'path/to/input_video.mp4'
output_dir = 'path/to/save/output'
```

3. Run the script:
```bash
python main.py
```

4. Output will be saved in the `results/` folder with bounding boxes and labels.

---

## 📦 Output Example

The output video contains:
- 🔴 Red box: Fallen person → `Person Fall Detected`
- 🟢 Green box: Standing person → `Standing Person`

---

## 📌 Notes

- Press `q` to stop the video early while it's running.
- The model only works on videos where people are clearly visible and not occluded.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

---

## 📧 Contact

For any questions: [LinkedIn](https://www.linkedin.com/in/mohamed-wasef-789743233/)
