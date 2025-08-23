import pickle

from flask import Flask

from utils import load_image


app = Flask(__name__)
f = open("randomforest.pickle", "rb")
model_pipeline = pickle.load(f)
f.close()


@app.get("/")
def index():
    test_image = load_image("images/dog/OIF-e2bexWrojgtQnAPPcUfOWQ.jpeg")
    test_image_pca = model_pipeline["pca"].transform([test_image])
    animal = model_pipeline["model"].predict(test_image_pca)[0]
    return animal

if __name__ == "__main__":
    app.run(debug=True)