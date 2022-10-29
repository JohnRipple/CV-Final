
import cv2 as cv
import numpy as np


IMAGE_NAME = 'cat-lyrics.png'


def main():
    music = cv.imread(IMAGE_NAME)
    music = cv.cvtColor(music, cv.COLOR_BGR2GRAY)
    thresh = cv.adaptiveThreshold(music, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 15, 2 )

    # Find the horizontal Lines
    horzSize = int(thresh.shape[1] / 30)
    horzStruct = cv.getStructuringElement(cv.MORPH_RECT, (horzSize, 1))
    horz = cv.erode(thresh, horzStruct)
    horz = cv.dilate(horz, horzStruct)

    # Find vertical lines
    vertSize = int(thresh.shape[0] / 30)
    vertStruct = cv.getStructuringElement(cv.MORPH_RECT, (1, vertSize))
    vert = cv.erode(thresh, vertStruct)
    vert = cv.dilate(vert, vertStruct)

    # Remove horizontal Lines and add vertical lines
    out = thresh - horz + vert
    out = cv.bitwise_not(out)
    out = cv.blur(out, (2,3))

    num_labels, labels, stats, centroid = cv.connectedComponentsWithStats(cv.bitwise_not(out))
    out = cv.cvtColor(out, cv.COLOR_GRAY2BGR)
    for i in range(num_labels):
        x = stats[i, cv.CC_STAT_LEFT]
        y = stats[i, cv.CC_STAT_TOP]
        w = stats[i, cv.CC_STAT_WIDTH]
        h = stats[i, cv.CC_STAT_HEIGHT]
        if h/w > 2 and h/w < 5:
            cv.imshow('Individual Notes', out[y:y + h, x:x + w])
            cv.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv.waitKey(30)

    cv.imshow("Threshold", out)
    cv.waitKey(0)


if __name__ == '__main__':
    main()

cv.destroyAllWindows()