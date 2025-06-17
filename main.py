import os
import cv2
from ultralytics import YOLO

# model loading and classes init
model_path = r'D:\pycharm\human fall detection\yolo11m.pt'
model = YOLO(model_path)
cls = model.names
#-----------------------------------------------
# video caputure
video_path = r'D:\pycharm\human fall detection\videos and images\fall.mp4'
cap = cv2.VideoCapture(video_path)
#-----------------------------------------------
# check for video is found or not
if not cap.isOpened():
    print('video not found')
    exit()
#-----------------------------------------------
# dimensions of frame
target_width = 1024
target_height = 600
#-----------------------------------------------
# determining the colors used in bounding boxes
green = (0,255,0)
red = (0,0,255)
#-----------------------------------------------
# to get the total frame counts
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
frame_count = 0
#-----------------------------------------------
# video saving
output_dir = r'D:\pycharm\human fall detection\results'
orginal_video_name = os.path.basename(video_path)
out_video_name = f'detected__{orginal_video_name}'
out_path = os.path.join(output_dir,out_video_name)
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_path,fourcc,fps,(target_width,target_height))


#reading video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Update progress
    frame_count +=1
    progress = (frame_count/total_frames)*100
    print(f"\rProcessing: {progress:.1f}% ({frame_count}/{total_frames})", end="")

    # resizing
    frame = cv2.resize(frame,(target_width,target_height))

    # make the detection with yolo model
    results = model(frame)[0]
    # capture of results data of the yolo prediction
    for result in results.boxes.data:
        x1,y1,x2,y2,conf,cls_id = result.cpu().numpy()
        # convert the numerical data to class name '0---->person'
        cls_id = int(cls_id)
        cls_id = cls[cls_id]
        # check that we have the person class only and the confidence greater than 50%
        if cls_id == 'person' and conf > 0.50 :
            B_height = y2-y1
            B_width = x2-x1
            is_fallen = B_width > B_height

            color = red if is_fallen else green
            label = 'Person Fall Detected' if is_fallen else 'Standing Person'

            cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),color,2)

            (text_width,text_height),baseline = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.5,2)
            rect_start = (int(x1), int(y1) - text_height - baseline - 5)
            rect_end = (int(x1) + text_width + 5, int(y1))
            cv2.rectangle(frame, rect_start, rect_end, color, -1)
            cv2.putText(frame, label, (int(x1) + 2, int(y1) - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    out.write(frame)
    cv2.imshow('fall detection frame',frame)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()
print(f'video saved in {out_path}')