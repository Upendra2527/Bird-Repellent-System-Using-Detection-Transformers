import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
from PIL import Image
import cv2
import os
import pygame
import matplotlib.pyplot as plt
import numpy as np

# Define the paths
checkpoint_path = "C:\\Users\\hp\\Desktop\\CVAT BIRDS ANNOTATIONS\\detr_model_checkpoint.ckpt"
model_weights_path = "C:\\Users\\hp\\Desktop\\CVAT BIRDS ANNOTATIONS\\detr_model_weights"
audio_path = "C:\\Users\\hp\\Desktop\\CVAT BIRDS ANNOTATIONS\\distress_call.wav"

# Initialize Pygame mixer
pygame.mixer.init()

def load_model():
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    return model, processor

model, processor = load_model()

def predict_frame(frame, threshold=0.9):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]
    
    return results

def visualize_frame(frame, results):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(frame_rgb, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
        label_text = f'Bird: {score:.2f}'
        cv2.putText(frame_rgb, label_text, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return frame_rgb

def detect_birds_laptop():
    cap = cv2.VideoCapture(0)

    plt.ion()  # Turn on interactive mode for real-time updates
    fig, ax = plt.subplots(figsize=(8, 6))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = predict_frame(frame, threshold=0.3)

        # If any birds detected, play audio
        if len(results['scores']) > 0:
            print("Bird detected! Playing sound.")
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

        output_frame_rgb = visualize_frame(frame, results)

        # Display frame using matplotlib
        ax.clear()
        ax.imshow(output_frame_rgb)
        ax.set_title("Bird Detection")
        ax.axis('off')
        plt.pause(0.001)

        # Optional: Add 'q' keypress to quit (but matplotlib doesn't support it directly)
        # Use a workaround like keyboard module if needed

    cap.release()
    plt.ioff()
    plt.close()
    cv2.destroyAllWindows()

detect_birds_laptop()
