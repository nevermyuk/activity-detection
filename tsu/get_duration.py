import cv2

cap = cv2.VideoCapture("../data/TSU/TSU_Videos_mp4/P11T15C01.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV v2.x used "CV_CAP_PROP_FPS"
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("fps = " + str(fps))
print("number of frames = " + str(frame_count))


cap.release()
