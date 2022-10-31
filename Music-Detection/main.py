
import cv2 as cv
import numpy as np


# IMAGE_NAME = 'cat-lyrics.png'
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
    out = thresh - horz - vert
    out = cv.bitwise_not(out)
    out = cv.blur(out, (2,3))
    out_display = cv.cvtColor(out, cv.COLOR_GRAY2BGR)   # Copy the image so it isn't affected by dilation/erosion

    # Get rid of small noise
    kernel = np.ones((int(out.shape[0] / 200), int(out.shape[0] / 200)))
    out = cv.dilate(out, kernel)
    out = cv.erode(out, kernel)

    # Get an image with only horizontal and vertical bars
    bars = horz + vert
    num_labels, labels, stats, centroid = cv.connectedComponentsWithStats(cv.bitwise_not(out))
    num_labels_bar, labels_bar, stats_bar, centroid_bar = cv.connectedComponentsWithStats(bars) # Find the horizontal bar locations
    out = cv.cvtColor(out, cv.COLOR_GRAY2BGR)
    bars = cv.cvtColor(cv.bitwise_not(bars), cv.COLOR_GRAY2BGR)
    for i in range(num_labels_bar):
        x = stats_bar[i, cv.CC_STAT_LEFT]
        y = stats_bar[i, cv.CC_STAT_TOP]
        w = stats_bar[i, cv.CC_STAT_WIDTH]
        h = stats_bar[i, cv.CC_STAT_HEIGHT]

        # Get only the horizontal music bars
        if x + y > (bars.shape[0]+bars.shape[1]) / 30:
            # Add a buffer in case notes go outside the bars
            buffer = 10
            y -= int(h / buffer)
            h += int(h / (buffer/4))
            cv.rectangle(out_display, (x, y), (x + w, y + h), (0, 0, 255), 1)
            for j in range(num_labels):
                x_note = stats[j, cv.CC_STAT_LEFT]
                y_note = stats[j, cv.CC_STAT_TOP]
                w_note = stats[j, cv.CC_STAT_WIDTH]
                h_note = stats[j, cv.CC_STAT_HEIGHT]

                # Find if the notes are insde the music bars boudning box
                if (x <= x_note and y <= y_note):
                    # Check if bottom right corner is in the outer bounding box
                    if x + w >= x_note + w_note and y + h >= y_note + h_note:
                        #if h_note/w_note >= 2 and h_note/w_note < 5:
                        # Filter connected components based on height and width ratio
                        if h_note / w_note >= 0.5and h_note / w_note < 5:
                            # print(h_note/w_note)
                            cv.imshow('Individual Notes', out[y_note:y_note + h_note, x_note:x_note + w_note])
                            cv.rectangle(out_display, (x_note, y_note), (x_note + w_note, y_note + h_note), (0, 0, 255), 1)
                            cv.waitKey(30)

    cv.imshow("Threshold", out_display)
    cv.waitKey(0)


if __name__ == '__main__':
    main()

cv.destroyAllWindows()