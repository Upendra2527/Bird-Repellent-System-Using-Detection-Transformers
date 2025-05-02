import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
from PIL import Image
import cv2
import pygame  # For playing audio
from picamera2 import Picamera2

# Define the paths to the saved model weights and checkpoint
checkpoint_path = "detr_model_checkpoint.ckpt"
model_weights_path = "detr_model_weights"

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

def detect_birds_raspberry_pi(audio_file):
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)
    
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={'size': (640, 480)}))
    picam2.start()
    
    while True:
        frame = picam2.capture_array()
        results = predict_frame(frame, threshold=0.3)
        
        if len(results['scores']) > 0:
            print("Bird detected! Playing distress call.")
            if not pygame.mixer.music.get_busy():
                pygame.mixer.music.play()
        
        cv2.imshow("Bird Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()

# Example usage
detect_birds_raspberry_pi("bird_sound.mp3")
