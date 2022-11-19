
import cv2 as cv
import numpy as np
import winsound


IMAGE_NAME = 'cat-lyrics.png'
# IMAGE_NAME = 'god.png'

# Takes in the horizontal lines picture and returns the y position of the top and bottom of each staff
def find_staff_box(horizontal_lines):
    staff = []
    dist = []
    top_bar = 0
    num_labels, labels, stats, centroid = cv.connectedComponentsWithStats(horizontal_lines) # Get all horizontal lines connected components

    # Find the distance between connected components
    for i in range(num_labels - 1):
        y = stats[i, cv.CC_STAT_TOP]
        y_next = stats[i+1, cv.CC_STAT_TOP]
        dist.append(y_next-y)
    dist.sort()
    bar_dist = dist[int(len(dist)/2)]   # Find median distance between connected components, most likely the distance between lines in a staff
    last = False
    for i in range(num_labels - 1):
        x = stats[i, cv.CC_STAT_LEFT]
        y = stats[i, cv.CC_STAT_TOP]
        y_next = stats[i + 1, cv.CC_STAT_TOP]
        x_next = stats[i + 1, cv.CC_STAT_LEFT]
        # shift = 10
        if not last:
            top_bar = y
        if (y_next - y) < bar_dist*1.2 and (x_next - x) < 3:
            last = True
        elif last:
            staff.append((top_bar, y))
            last = False
    return staff


def get_Note_Freq(note_cord, top_staff, bottom_staff):
    # Takes the y coordinates of the note, bottom of staff, and top of staff and converts note into the frequency
    # of the note, assuming it's not sharped or flatted
    # https://pages.mtu.edu/~suits/notefreqs.html
    line_dis = (bottom_staff - top_staff) / 8
    note_pos = round((bottom_staff - note_cord) / line_dis) + 4  # gets an index based around A3
    octave = int(note_pos / 8)  # finds the octave
    oct_start = 220 * 2**octave
    oct_end = 2 * oct_start
    hz_per_note = (oct_end - oct_start) / 12
    # Because the frequency of notes is dependent upon even split of 12 notes from oct_start to end, we only want the
    # ones that aren't sharped or flatted
    note_to_interval = [0, 2, 3, 5, 7, 8, 10]
    hz = hz_per_note * note_to_interval[note_pos % 7] + oct_start

    return hz

def play_song(frequencies):
    # Takes an in order list of frequencies and plays the song
    for freq in frequencies:
        #the note duration in ms
        sec = 1000
        winsound.Beep(freq, sec)

def main():
    music = cv.imread(IMAGE_NAME)
    music = cv.cvtColor(music, cv.COLOR_BGR2GRAY)
    thresh = cv.adaptiveThreshold(music, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 15, 2 )

    # Find the horizontal Lines
    horzSize = int(thresh.shape[1] / 30)
    horzStruct = cv.getStructuringElement(cv.MORPH_RECT, (horzSize, 1))
    horz = cv.erode(thresh, horzStruct)
    horz = cv.dilate(horz, horzStruct)
    staff_positions = find_staff_box(horz)

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
    num_labels_bar, labels_bar, stats_bar, centroid_bar = cv.connectedComponentsWithStats(bars)  # Find the horizontal bar locations
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
                if x <= x_note and y <= y_note:
                    # Check if bottom right corner is in the outer bounding box
                    if x + w >= x_note + w_note and y + h >= y_note + h_note:
                        # if h_note/w_note >= 2 and h_note/w_note < 5:
                        # Filter connected components based on height and width ratio
                        if h_note / w_note >= 0.5and h_note / w_note < 5:
                            # print(h_note/w_note)
                            cv.imshow('Individual Notes', out[y_note:y_note + h_note, x_note:x_note + w_note])
                            cv.putText(out_display, str(j), (x_note, y_note-10), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
                            cv.rectangle(out_display, (x_note, y_note), (x_note + w_note, y_note + h_note), (0, 0, 255), 1)
                            for staff in staff_positions:
                                if staff[0] > y and staff[1] < y + h:
                                    freq = get_Note_Freq(int(y_note + h_note/2), staff[0], staff[1])
                                    print(freq)
                            cv.waitKey(30)

    cv.imshow("Threshold", out_display)
    cv.waitKey(0)


if __name__ == '__main__':
    main()

cv.destroyAllWindows()