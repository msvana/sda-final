import pickle
import os

from flask import Flask, request

from utils import load_image

os.makedirs("./uploads/", exist_ok=True)

app = Flask(__name__)
f = open("randomforest.pickle", "rb")
model_pipeline = pickle.load(f)
f.close()


@app.post("/")
def index():
    if "file" not in request.files:
        return "SEND_FILE", 400
    
    file = request.files["file"]
    file.save(f"./uploads/{file.filename}")
    
    test_image = load_image(f"./uploads/{file.filename}")
    test_image_pca = model_pipeline["pca"].transform([test_image])
    animal = model_pipeline["model"].predict(test_image_pca)[0]
    return animal

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")