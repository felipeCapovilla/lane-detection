import cv2
import numpy as np

video = cv2.VideoCapture("v01.MP4")  #Path to the video.

while video.isOpened():
    status, frame = video.read()
    
    if not status:  
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame_blurred = cv2.GaussianBlur(frame, (7, 7), 0)

    threshold_low = 10
    threshold_high = 200
    frame_canny = cv2.Canny(frame_blurred, threshold_low, threshold_high)


    #CAUTION: the interest region depends on each situation. Open your frame with matplotlib to view the interest region that you want.
    interest_region  = np.array([[(200,550),(200,100),(1000,100),(1000,550)]],dtype=np.int32) 
    mask = np.zeros_like(gray_frame)
    cv2.fillPoly(mask, interest_region, 255)
    masked_frame = cv2.bitwise_and(frame_canny, mask)

    p = 2
    angle = np.pi / 180
    threshold = 60
    min_pixels_lane = 170
    max_lane_gap = 40
    lines = cv2.HoughLinesP(masked_frame, p, angle, threshold, np.array([]), minLineLength=min_pixels_lane, maxLineGap=max_lane_gap)
    line_image = np.zeros((masked_frame.shape[0], masked_frame.shape[1], 3), dtype=np.uint8)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), [0, 0, 255], 5)

    alpha = 0.5
    beta = 3
    gamma = 0

    output_frame = cv2.addWeighted(frame, alpha, line_image, beta, gamma)

    cv2.imshow("Processed Video", output_frame)

    if cv2.waitKey(30) & 0xFF == ord('q'): 
        break

video.release()
cv2.destroyAllWindows()
