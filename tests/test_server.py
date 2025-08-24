import sys
sys.path.append("./src")

from server import app

def test_dog_answer():
    test_file = "images/dog/OIF-e2bexWrojgtQnAPPcUfOWQ.jpeg"
    test_filename = test_file.split("/")[-1]
    test_client = app.test_client()

    f = open(test_file, "rb")
    files = {"file": (f, test_filename)}
    response = test_client.post("/", data=files)
    f.close()

    assert response.text == "dog"

def test_cat_answer():
    test_file = "images/cat/3.jpeg"
    test_filename = test_file.split("/")[-1]
    test_client = app.test_client()

    f = open(test_file, "rb")
    files = {"file": (f, test_filename)}
    response = test_client.post("/", data=files)
    f.close()

    assert response.text == "cat"