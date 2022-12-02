import cv2
import cv2 as cv
import numpy as np
import pyaudio
import playAudio
from pysinewave import SineWave
import time

IMAGE_NAME = 'cat-lyrics.png'
# IMAGE_NAME = 'god.png'
# IMAGE_NAME = 'rhody.png'
# IMAGE_NAME = 'cabbage.png'
#this reshapes the output image of sheetmusic
cv2.namedWindow("Threshold", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Threshold", 400,500)


class Note:
    # Holds all the information about a note including
    def __init__(self, x, y, w, h, frequency, length=0.5):
        self.x = x  # x coordinate (top left)
        self.y = y  # y coordinate  (top left)
        self.w = w  # note width
        self.h = h  # note height
        self.frequency = frequency  # note frequency
        self.length = length    # note length in seconds

    def __str__(self):
        return f"X: {self.x} Y: {self.y} Frequency: {self.frequency}"


def find_staff_box(horizontal_lines):
    # Takes in the horizontal lines picture and returns the y position of the top and bottom of each staff
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
    octave = int(note_pos / 7)  # finds the octave
    oct_start = 220 * 2**octave
    oct_end = 2 * oct_start
    hz_per_note = (oct_end - oct_start) / 12
    # Because the frequency of notes is dependent upon even split of 12 notes from oct_start to end, we only want the
    # ones that aren't sharped or flatted
    note_to_interval = [0, 1, 2, 4, 6, 7, 9]
    hz = hz_per_note * note_to_interval[note_pos % 7] + oct_start

    return hz

def organize(staff_positions, notes):
    # notes is a python list of all notes of format [x pos, y pos, freq]
    # staff positions is a list of the staff positions
    #
    # the function returns multidimensional list of freq for each note organized by staff
    # e.g freq[0] returns all the notes in the first staff

    #list for the note freq
    frequencies = []
    # list of notes organized by staff
    # format: list[i][(x of note, y of note, frequency of note)]
    # i is the staff
    organize_notes = []
    #keep track of the place in the organize_notes list
    place = 0

    for staff in staff_positions:
        organize_notes.append([])

        #top and bottom of staff positions
        top = staff[0] + 10
        bottom = staff[1] + 10

        for note in notes:
            #note coordinate
            x_note = note.x
            y_note = note.y
            #note freq
            freq = note.frequency

            #see if note is in the current staff
            if top <= y_note <= bottom:
                # add note to current staff
                # organize_notes[place].append([x_note,y_note,freq])
                organize_notes[place].append(note)

        place = place + 1

    #all the notes placed in the proper staff, now organize each staff
    for staff in organize_notes:
        temp_staff = staff
        staff = sorted(temp_staff, key=lambda x: x.x)
        print(staff)
        for note in staff:
            # frequencies.append(note[2])
            frequencies.append(note)

    #return the list of frequencies
    return frequencies


def play_song(frequencies):
    # Takes an in order list of frequencies and plays the song
    sample_rate = 44100
    p = pyaudio.PyAudio()   # Creates audio object
    # creates the audio stream
    stream = p.open(format=p.get_format_from_width(1),  # 8bit
                    channels=1,  # mono
                    rate=sample_rate,
                    output=True)
    for freq in frequencies:
        #the note duration in s
        sec = 0.5
        print(freq.frequency, freq.length)
        # Plays the frequency using a sine wave
        show_note(freq.frequency)
        playAudio.sine_tone(
            stream,
            frequency=int(freq.frequency),  # Hz, waves per second A4
            duration=freq.length,  # seconds to play sound
            volume=0.5,  # 0..1 how loud it is
            sample_rate=sample_rate # number of samples per second
        )
    # Closes the steam and terminates the audio object
    stream.stop_stream()
    stream.close()
    p.terminate()


def merge_notes(notes, thresh):
    # Takes array of Notes class
    # Checks if there is an object right next to a note and merges them
    # Merges dots and notes and creates a dotted half note.
    popping_list = []
    kernel = np.ones((3,3), np.uint8)
    for i in range(len(notes) ):
        # Gets the bulb of the note from the original thresholded image
        notes_img = thresh[notes[i].y: notes[i].y + notes[i].h, notes[i].x: notes[i].x + notes[i].w]

        # Gets the bulb and stem
        long_notes_img = thresh[notes[i].y - notes[i].h * 2: notes[i].y + notes[i].h, notes[i].x: notes[i].x + notes[i].w]
        num_labels, labels, stats, centroid = cv.connectedComponentsWithStats(long_notes_img)   # Sees if there is a stem or if its a whole note
        notes_img = cv.morphologyEx(notes_img, cv.MORPH_OPEN, kernel)   # Filter the image to make it more white (finds half notes)
        notes_img = cv.bitwise_not(notes_img)

        white_pixels = np.sum(notes_img == 255)
        black_pixels = np.sum(notes_img == 0)
        # print(white_pixels / (black_pixels + white_pixels), num_labels)

        # Checks for ratio of white to black pixels to find whole, half, or dotted half notes
        if white_pixels / (black_pixels + white_pixels) > 0.6:
            # Every note with a stem had 2 connected components while whole notes had 4+
            if num_labels > 2:
                notes[i].length = notes[i].length * 4
            else:
                notes[i].length = notes[i].length * 2

        # Check if the next possible note is within a note length of the current note and close to the y position
        if i < len(notes) - 1:
            if (notes[i].x + notes[i].w) < notes[i + 1].x < (notes[i].x + 2 * notes[i].w):
                if (notes[i].y + 0.5 * notes[i].h) > notes[i + 1].y > (notes[i].y - 0.5 * notes[i].h):
                    notes[i].w = notes[i + 1].w + notes[i+1].x - notes[i].x
                    notes[i].length = notes[i].length * 1.5
                    popping_list.append(i + 1)
                    i += 1
    for i in sorted(popping_list, reverse=True):
        notes.pop(i)

    return notes



def show_note(note):
    # displays the note to the keyboard.jpg image, so you can play along
    board = cv.imread("keyboard.jpg")
    length = board.shape[1] / 21
    octave = int(np.log2(note / 220))
    note_pos = note - 220 * 2 ** octave
    note_pos = round(note_pos / (220*2**octave / 12))
    y = 171
    note_to_interval = [0, 1, 2, 4, 6, 7, 9]
    note_pos = note_to_interval.index(note_pos)
    # undoes all the math from get_note_frequency to get the octave and position of the note
    print(octave, note_pos)
    if octave < -1 or octave == -1 and note_pos < 2 or octave > 2 or octave == 2 and note_pos > 1:
        return

    x = int(length * (octave * 7 + note_pos) + 208)  # calculates where the dot should be based on the image, octave, and pos
    cv.circle(board, (x,y), 7, (255,255,0), -1)
    cv.imshow("board", board)
    cv.waitKey(1)  # necessarry so that the image displays while running


def identify_orb(notes):
    # notes is ([x_note, y_note, freq])
    # takes identified frequencies and determines which ones are valid
    # func is identifying some things as notes that are not notes like the 4,4
    detector = cv2.ORB_create(nfeatures=500,  # default = 500
                              nlevels=8,  # default = 8
                              firstLevel=0,  # default = 0
                              patchSize=31,  # default = 31
                              edgeThreshold=31)  # default = 31
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    cat_img = cv2.imread("cat-lyrics.png")
    full_note = cv2.imread("full_note.png")
    half_note = cv2.imread("half_note.png")
    gray_full = cv.cvtColor(full_note, cv.COLOR_BGR2GRAY)
    # gray_cat = cv.cvtColor(cat_img, cv.COLOR_BGR2GRAY)
    gray_half = cv.cvtColor(half_note, cv.COLOR_BGR2GRAY)
    for note in notes:
        x_pos = note.x
        y_pos = note.y
        det_note = cat_img[y_pos:note.y + note.h, x_pos:note.x + note.w]
        cv2.imshow("Detected note", det_note)
        # cv2.waitKey(0)
        gray_det = cv.cvtColor(det_note, cv.COLOR_BGR2GRAY)
        kp_train, desc_train = detector.detectAndCompute(gray_det, mask=None)
        kp_query, desc_query = detector.detectAndCompute(gray_full, mask=None)
        matches = matcher.match(desc_query, desc_train)
        print("matches full: " + str(matches))       # no matches showing up for full note
        final_img = cv2.drawMatches(full_note, kp_query,
                                    det_note, kp_train, matches[:20], None)
        # Show the final image
        cv2.imshow("Match of full note", final_img)
        # cv2.waitKey(0)

        # half note
        kp_query, desc_query = detector.detectAndCompute(gray_half, mask=None)
        matches = matcher.match(desc_query, desc_train)
        print("matches half: " + str(matches))
        final_img = cv2.drawMatches(half_note, kp_query,
                                    det_note, kp_train, matches[:20], None)
        # Show the final image
        cv2.imshow("Matches of half note", final_img)
        cv2.waitKey(0)


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

    # list to store all note coordinates
    notes = []

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
    k = 0
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

                # Find if the notes are inside the music bars bounding box
                if x <= x_note and y <= y_note:
                    # Check if bottom right corner is in the outer bounding box
                    if x + w >= x_note + w_note and y + h >= y_note + h_note:
                        # Filter connected components based on height and width ratio
                        if 0.5 <= h_note / w_note < 5:
                            if x_note > x + w * 0.05:
                                if k == 0 and x_note + w_note < x + w * 0.1:
                                    cv.rectangle(out_display, (x, y), (int(x + w * 0.1), y + h), (0, 255, 0), 1)
                                    k = 1
                                else:
                                    cv.imshow('Individual Notes', out[y_note:y_note + h_note, x_note:x_note + w_note])
                                    for staff in staff_positions:
                                        if staff[0] > y and staff[1] < y + h:
                                            freq = get_Note_Freq(int(y_note + h_note/2), staff[0], staff[1])

                                            notes.append(Note(x_note, y_note, w_note, h_note, freq))
                            else:
                                cv.rectangle(out_display, (x, y), (int(x + w * 0.05), y + h), (255, 0, 255), 1)
                            cv.waitKey(30)

    frequencies = organize(staff_positions, notes)
    frequencies = merge_notes(frequencies, thresh)
    for note in frequencies:
        cv.rectangle(out_display, (note.x, note.y), (note.x + note.w, note.y + note.h), (0, 0, 255), 1)
    cv.imshow("Threshold", out_display)
    cv.waitKey(0)
    # identify_orb(frequencies)
    play_song(frequencies)
    print(len(notes))

if __name__ == '__main__':
    main()

cv.destroyAllWindows()