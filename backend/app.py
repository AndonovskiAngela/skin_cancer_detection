import numpy as np
from flasgger import Swagger
from flask import Flask, request
from PIL import Image
from io import BytesIO
from flask_cors import CORS
from train import train_model, predict


def create_app():
    app = Flask(__name__)
    CORS(app)
    Swagger(app)
    train_model()

    return app


app = create_app()


def imageToArray(image):
    img_data = image.read()
    img = Image.open(BytesIO(img_data)).convert('RGB')
    image_array = np.asarray(img)
    image_array = image_array / 255.0

    image_array = image_array.reshape(224, 224, 3)

    return image_array


@app.route("/test", methods=["POST"])
def function_for_api():
    img = request.files['file']
    CLASSES = np.array(['benign', 'malignant'])
    image_array = imageToArray(img)
    image_array = np.expand_dims(image_array, axis=0)
    preds = predict(image_array)
    preds_single = CLASSES[np.argmax(preds, axis=-1)]
    confidence_score = np.max(preds, axis=-1)
    confidence_score_percentage = round(float(confidence_score[0]) * 100, 2)
    return {"class": preds_single[0], "confidence": confidence_score_percentage}


if __name__ == '__main__':
    app.run(debug=False)
