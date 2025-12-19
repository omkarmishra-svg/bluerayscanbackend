import requests
from PIL import Image
import io
import json

img = Image.new('RGB', (100, 100), color = 'red')
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='PNG')
img_byte_arr.seek(0)

url = "http://127.0.0.1:8000/api/scan"
files = {'file': ('test_image.png', img_byte_arr, 'image/png')}

try:
    response = requests.post(url, files=files)
    data = response.json()
    print("Analysis Result:")
    print(json.dumps(data, indent=2))
except Exception as e:
    print(e)
