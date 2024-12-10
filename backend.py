from fastapi import FastAPI, File, UploadFile
import cv2
import pandas as pd
from ultralytics import YOLO

app = FastAPI()

# Enable CORS
from fastapi.middleware.cors import CORSMiddleware

orig_origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=orig_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the YOLO models
model1 = YOLO('trained_yolov8_model.pt')
model2 = YOLO('yolov8n.pt')

@app.post("/process_video/")
async def process_video(file: UploadFile = File(...)):
    video_path = f'./{file.filename}'
    
    # Save the uploaded video
    with open(video_path, 'wb') as buffer:
        buffer.write(file.file.read())
    
    frame_counts_model1 = []
    frame_counts_model2 = []
    
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process with model 1
        results1 = model1.predict(frame)
        counts_model1 = len(results1[0].boxes.cls)  # Count bounding boxes
        frame_counts_model1.append(counts_model1)
        
        # Process with model 2
        results2 = model2.predict(frame)
        counts_model2 = len(results2[0].boxes.cls)  # Count bounding boxes
        frame_counts_model2.append(counts_model2)

    cap.release()
    
    # Calculate average bounding boxes
    avg_count_model1 = sum(frame_counts_model1) / len(frame_counts_model1) if frame_counts_model1 else 0
    avg_count_model2 = sum(frame_counts_model2) / len(frame_counts_model2) if frame_counts_model2 else 0
    
    avg_count_model1_percentage = (avg_count_model1/30) * 100
    avg_count_model2_percentage = (avg_count_model1/30) * 100
    if avg_count_model1_percentage > 100 :
        avg_count_model1_percentage= 100
    if avg_count_model2_percentage > 100:
        avg_count_model2_percentage = 100
    return {
        "model1_average_count": avg_count_model1_percentage,
        "model2_average_count": avg_count_model2_percentage
    }

