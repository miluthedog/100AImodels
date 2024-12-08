import cv2 as cv
import numpy as np
from keras import models
import sys
sys.stdout.reconfigure(encoding='utf-8')

model = models.load_model('catClassification/meomeo.keras')
cam = cv.VideoCapture(0)

while True:
    ret, frame = cam.read()

    frameResized = cv.resize(frame, (128, 128)) / 255.0
    frameResized = np.expand_dims(frameResized, axis=0)

    prediction = model.predict(frameResized)
    confidence = prediction[0][0] * 100
    label = "meo meo" if prediction[0] > 0.5 else "???"

    cv.putText(frame, f"{label} ({confidence:.2f}%)", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv.imshow('Meo meo finder', frame)

    pressEsc = cv.waitKey(1) & 0xff
    if pressEsc == 27:
        break

cam.release()
cv.destroyAllWindows()
