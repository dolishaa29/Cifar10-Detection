# predict_client.py
import requests
import sys

def predict(image_path, url="http://localhost:5000/predict"):
    with open(image_path, "rb") as f:
        files = {"file": (image_path, f, "image/jpeg")}
        resp = requests.post(url, files=files)
        print(resp.status_code, resp.text)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict_client.py path/to/image.jpg")
    else:
        predict(sys.argv[1])
