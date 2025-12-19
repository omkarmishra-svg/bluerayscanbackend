import cv2
import numpy as np
import os
import uuid
from app.config import settings

class GradCAM:
    def __init__(self):
        pass
    
    def generate_mock_heatmap(self, image_path: str, label: str) -> str:
        """
        Generates a convincing simulated heatmap for fallback/demo purposes.
        If FAKE: Highlights random facial regions (simulating detection).
        If REAL: diffuse highlight.
        """
        try:
            # 1. Load image
            img = cv2.imread(image_path)
            if img is None:
                return ""
            
            # 2. Generate a heatmap mask
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
            
            # Simulate "Attention" blobs
            h, w = img.shape[:2]
            
            if label == "FAKE":
                 # Create 1-3 random "hotspots" (simulating detected artifacts)
                for _ in range(np.random.randint(1, 3)):
                    center_x = np.random.randint(int(w*0.3), int(w*0.7))
                    center_y = np.random.randint(int(h*0.3), int(h*0.7))
                    sigma = min(h, w) * 0.15
                    
                    x = np.arange(0, w, 1, float)
                    y = np.arange(0, h, 1, float)
                    x, y = np.meshgrid(x, y)
                    
                    blob = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
                    mask += blob
            else:
                 # Real: Diffuse attention, less focused
                 mask.fill(0.2)
            
            # Normalize
            mask = np.maximum(mask, 0)
            mask = np.minimum(mask, 1)
            mask = np.uint8(255 * mask)
            
            # 3. Apply Color Map
            heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            
            # 4. Superimpose
            superimposed_img = heatmap * 0.4 + img
            
            # 5. Save
            unique_filename = f"heatmap_{uuid.uuid4().hex}.jpg"
            save_path = os.path.join(settings.UPLOAD_DIR, unique_filename)
            cv2.imwrite(save_path, superimposed_img)
            
            return unique_filename
        except Exception as e:
            print(f"Heatmap gen failed: {e}")
            return ""

gradcam = GradCAM()
