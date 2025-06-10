import cv2
import sklearn
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from util import get_parking_spots_bboxes, empty_or_not

def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

mask ="D:\Opencv\projects\ParkingSpaceCounter\code\mask_1920_1080.png"
video_path ="D:\Opencv\projects\ParkingSpaceCounter\code\parking_1920_1080.mp4"

mask = cv2.imread(mask, 0)
cap = cv2.VideoCapture(video_path)

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)

# Initialize a list to store the status of each spot
spot_status = [None for _ in spots]
diffs = [None for _ in spots]

# Initialize the previous frame
prev_frame = None
3


frame_nmr = 0
ret = True
step = 30  # Process every 30th frame

while ret:
    ret, frame = cap.read()
    if not ret:  # Break if no frame is readq
        break

    # Process every 30th frame
    if frame_nmr % step == 0 and prev_frame is not None:
        for spot_index, spot in enumerate(spots):
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            diffs[spot_index] = calc_diff(spot_crop, prev_frame[y1:y1 + h, x1:x1 + w, :])
        print([diffs[j] for j in np.argsort(diffs)][::-1])

    # Determine the spot status for each spot
    if frame_nmr % step == 0:
        if prev_frame is None:
            arr_ = range(len(spots))
        else:
            arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]
        
        # Update spot status in the correct list
        for spot_indx in arr_:
            spot = spots[spot_indx]
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            status = empty_or_not(spot_crop)
            spot_status[spot_indx] = status  # Assign status correctly

    # Update the previous frame
    if frame_nmr % step == 0:
        prev_frame = frame.copy()

    # Draw rectangles based on spot status
    for spot_indx, spot in enumerate(spots):
        status = spot_status[spot_indx]
        x1, y1, w, h = spots[spot_indx]

        if status:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)  # Green for available
        else:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)  # Red for unavailable

    # Display available spots text
    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
    cv2.putText(frame, 'Available spots: {} / {}'.format(str(sum(spot_status)), str(len(spot_status))),
                (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the frame
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_nmr += 1

cap.release()
cv2.destroyAllWindows()
