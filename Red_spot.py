# import cv2
# import imutils

# redLower = (15, 164, 162)
# redUpper = (45, 232, 255)
# # redLower = (0, 120, 70)
# # redUpper = (10, 255, 255)

# camera = cv2.VideoCapture(0)

# while True:
#     (grabbed , frame)= camera.read()

#     frame = imutils.resize(frame, width=600)
#     blurred = cv2.GaussianBlur(frame, (11,11),0)
#     hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

#     mask = cv2.inRange(hsv,redLower, redUpper)
#     mask = cv2.erode(mask, None, iterations=2)
#     mask = cv2.dilate(mask, None, iterations=2)

#     cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
#     centre = None
#     if len(cnts)>0:
#         c = max(cnts, key=cv2.contourArea)
#         ((x,y),radius) = cv2.minEnclosingCircle(c)
#         M = cv2.moments(c)
#         centre = (int(M['m10']/M['m00']),int(M['m01']/M['m00']))
#         if radius>10:
#            cv2.circle(frame, (int(x),int(y)),int(radius),(0,255,255),2)
#            cv2.circle(frame, centre, 5,(0,0,255), -1)
#            if radius > 250:
#               print("stop")
#            else:
#               if(centre[0]<150):
#                  print("left")
#               elif(centre[0]>450):
#                 print("right")
#               elif(radius<250):
#                  print("front") 
#               else:
#                  print("Stop")
#     cv2.imshow("frame",frame)
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord("q"):
#        break

# camera.release()
# cv2.destroyAllWindows()                       


import cv2

camera = cv2.VideoCapture(0)

# Read the first frame
ret, prev_frame = camera.read()
prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_frame = cv2.GaussianBlur(prev_frame, (21, 21), 0)

while True:
    # Read current frame
    ret, frame = camera.read()
    if not ret:
        break

    # Convert to grayscale and blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Find difference between current frame and previous frame
    frame_delta = cv2.absdiff(prev_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in the threshold image
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        if cv2.contourArea(c) < 500:  # ignore small motion
            continue

        # Compute the bounding box and center
        (x, y, w, h) = cv2.boundingRect(c)
        center = (x + w // 2, y + h // 2)

        # Draw red circle on the moving object
        cv2.circle(frame, center, 20, (0, 0, 255), 2)

        # Optional: direction logic
        if center[0] < 150:
            print("Left")
        elif center[0] > 450:
            print("Right")
        else:
            print("Front")

    # Show frame
    cv2.imshow("Motion Detection", frame)

    # Update previous frame
    prev_frame = gray.copy()

    # Break if 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
