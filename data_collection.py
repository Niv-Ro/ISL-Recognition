import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time

cap = cv2.VideoCapture(1)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300
counter = 0
folder = "Data/ש"

while True:
    success, img = cap.read()
    if not success:
        continue

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        x1 = max(x - offset, 0)
        y1 = max(y - offset, 0)
        x2 = min(x + w + offset, img.shape[1])
        y2 = min(y + h + offset, img.shape[0])

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size != 0:
            cropH, cropW = imgCrop.shape[:2]
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            aspect_ratio = cropH / cropW

            if aspect_ratio > 1:
                # גובה גדול יותר – נצמד לגובה 300
                scale = imgSize / cropH
                newW = int(cropW * scale)
                newH = imgSize
                imgResize = cv2.resize(imgCrop, (newW, newH))
                wGap = (imgSize - newW) // 2
                imgWhite[:, wGap:wGap + newW] = imgResize
            else:
                # רוחב גדול יותר – נצמד לרוחב 300
                scale = imgSize / cropW
                newH = int(cropH * scale)
                newW = imgSize
                imgResize = cv2.resize(imgCrop, (newW, newH))
                hGap = (imgSize - newH) // 2
                imgWhite[hGap:hGap + newH, :] = imgResize

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)


    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)