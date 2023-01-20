from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from mylib.mailer import Mailer
from mylib import config, thread
import time, schedule, csv
import numpy as np
import argparse, imutils
import time, dlib, cv2, datetime
from itertools import zip_longest

t0 = time.time()

def run():

	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--prototxt", required=False,
		help="path to Caffe 'deploy' prototxt file")
	ap.add_argument("-m", "--model", required=True,
		help="path to Caffe pre-trained model")
	ap.add_argument("-i", "--input", type=str,
		help="path to optional input video file")
	ap.add_argument("-o", "--output", type=str,
		help="path to optional output video file")
	# 탐지 최소 확률 0.5
	ap.add_argument("-c", "--confidence", type=float, default=0.5,
		help="minimum probability to filter weak detections")
	ap.add_argument("-s", "--skip-frames", type=int, default=30,
		help="# of skip frames between detections")
	args = vars(ap.parse_args())

	# initialize the list of class labels MobileNet SSD was trained to
	# detect
	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]

	# 직렬화 된 모델을 가져온다.
	net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

	# ip카메라의 주소를 통해 영상을 가져오는 구문
	if not args.get("input", False):
		print("[INFO] Starting the live stream..")
		vs = VideoStream(config.url).start()
		time.sleep(2.0)

	# ip카메라가 없을때 로컬 비디오 사용
	else:
		print("[INFO] Starting the video..")
		vs = cv2.VideoCapture(args["input"])

	# 비디오 초기화
	writer = None

	# 프레임 크기 초기화
	W = None
	H = None

	# 추적기를 초기화
	# dlib 추적기에 인덱스 붙이기
	# 인덱스를 trackableObjects에 매핑
	ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
	trackers = []
	trackableObjects = {}

	# 이동한 보행자에 따라 프레임 수 초기화
	totalFrames = 0
	totalDown = 0
	totalUp = 0
	x = []
	empty=[]
	empty1=[]

	# 프레임 당 추적 시작
	fps = FPS().start()

	if config.Thread:
		vs = thread.ThreadingClass(config.url)

	# loop over frames from the video stream
	while True:
		#다음 프레임을 읽음
		frame = vs.read()
		frame = frame[1] if args.get("input", False) else frame

		# 영상이 더이상 없다면 끝내기
		if args["input"] is not None and frame is None:
			break

		# 픽셀 사이즈를 정하고 프레임을 씌움
		frame = imutils.resize(frame, width = 500)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		if W is None or H is None:
			(H, W) = frame.shape[:2]

		# 비디오 저장
		if args["output"] is not None and writer is None:
			fourcc = cv2.VideoWriter_fourcc(*"mp4v")
			writer = cv2.VideoWriter(args["output"], fourcc, 30,(W, H), True)

		# 경계를 그리며 현재 상태를 초기화
		# 검출 or 추적까지 대기
		status = "대기"
		rects = []

		# 추적기가 물체를 감지함
		if totalFrames % args["skip_frames"] == 0:
			# 현재 상태를 탐지으로 설정
			status = "탐지중"
			trackers = []

			# 프레임을 blob로변환
			# 객체 인식 잘안되면 scalefactor 사이즈좀 키워볼까
			blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
			net.setInput(blob)
			detections = net.forward()

			# 탐지하는것을 반복
			for i in np.arange(0, detections.shape[2]):
				# 신뢰도 추출
				confidence = detections[0, 0, i, 2]

				# 신뢰도 보다 낮으면 거르기
				if confidence > args["confidence"]:
					# 감지 목록에서 인덱스 추출
					idx = int(detections[0, 0, i, 1])

					# 사람이 아니면 무시
					if CLASSES[idx] != "person":
						continue

					# x,y 좌표 계산
					box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
					(startX, startY, endX, endY) = box.astype("int")


					# dlib 직사각형 모양 경계 그려주기
					# tracker
					tracker = dlib.correlation_tracker()
					rect = dlib.rectangle(startX, startY, endX, endY)
					tracker.start_track(rgb, rect)

					# 트래커 목록에 트래커를 추가
					trackers.append(tracker)

		# 탐지 후 추적
		else:
			# loop over the trackers
			for tracker in trackers:
				# 현재 상태를 추적으로 설정
				status = "추적중"

				# update the tracker and grab the updated position
				tracker.update(rgb)
				pos = tracker.get_position()

				# unpack the position object
				startX = int(pos.left())
				startY = int(pos.top())
				endX = int(pos.right())
				endY = int(pos.bottom())

				# add the bounding box coordinates to the rectangles list
				rects.append((startX, startY, endX, endY))

		# draw a horizontal line in the center of the frame -- once an
		# object crosses this line we will determine whether they were
		# moving 'up' or 'down'
		cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 0), 3)
		cv2.putText(frame, "-Prediction border - Entrance-", (10, H - ((i * 20) + 200)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

		# use the centroid tracker to associate the (1) old object
		# centroids with (2) the newly computed object centroids
		objects = ct.update(rects)

		# loop over the tracked objects
		for (objectID, centroid) in objects.items():
			# check to see if a trackable object exists for the current
			# object ID
			to = trackableObjects.get(objectID, None)

			# if there is no existing trackable object, create one
			if to is None:
				to = TrackableObject(objectID, centroid)

			# otherwise, there is a trackable object so we can utilize it
			# to determine direction
			else:
				# the difference between the y-coordinate of the *current*
				# centroid and the mean of *previous* centroids will tell
				# us in which direction the object is moving (negative for
				# 'up' and positive for 'down')
				y = [c[1] for c in to.centroids]
				direction = centroid[1] - np.mean(y)
				to.centroids.append(centroid)

				# check to see if the object has been counted or not
				if not to.counted:
					# if the direction is negative (indicating the object
					# is moving up) AND the centroid is above the center
					# line, count the object
					if direction < 0 and centroid[1] < H // 2:
						totalUp += 1
						empty.append(totalUp)
						to.counted = True

					# if the direction is positive (indicating the object
					# is moving down) AND the centroid is below the
					# center line, count the object
					elif direction > 0 and centroid[1] > H // 2:
						totalDown += 1
						empty1.append(totalDown)
						#print(empty1[-1])
						# if the people limit exceeds over threshold, send an email alert
						if sum(x) >= config.Threshold:
							cv2.putText(frame, "-ALERT: People limit exceeded-", (10, frame.shape[0] - 80),
								cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
							if config.ALERT:
								print("[INFO] Sending email alert..")
								Mailer().send(config.MAIL)
								print("[INFO] Alert sent")

						to.counted = True
						
					x = []
					# compute the sum of total people inside
					x.append(len(empty1)-len(empty))
					#print("Total people inside:", x)


			# store the trackable object in our dictionary
			trackableObjects[objectID] = to

			# draw both the ID of the object and the centroid of the
			# object on the output frame
			text = "ID {}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

		# construct a tuple of information we will be displaying on the
		info = [
		("Exit", totalUp),
		("Enter", totalDown),
		("Status", status),
		]

		info2 = [
		("Total people inside", x),
		]

                # Display the output
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

		for (i, (k, v)) in enumerate(info2):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (265, H - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

		# Initiate a simple log to save data at end of the day
		if config.Log:
			datetimee = [datetime.datetime.now()]
			d = [datetimee, empty1, empty, x]
			export_data = zip_longest(*d, fillvalue = '')

			with open('Log.csv', 'w', newline='') as myfile:
				wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
				wr.writerow(("End Time", "In", "Out", "Total Inside"))
				wr.writerows(export_data)
				
		# check to see if we should write the frame to disk
		if writer is not None:
			writer.write(frame)

		# show the output frame
		cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

		# increment the total number of frames processed thus far and
		# then update the FPS counter
		totalFrames += 1
		fps.update()

		if config.Timer:
			# Automatic timer to stop the live stream. Set to 8 hours (28800s).
			t1 = time.time()
			num_seconds=(t1-t0)
			if num_seconds > 28800:
				break

	# stop the timer and display FPS information
	fps.stop()
	print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


	# # if we are not using a video file, stop the camera video stream
	# if not args.get("input", False):
	# 	vs.stop()
	#
	# # otherwise, release the video file pointer
	# else:
	# 	vs.release()
	
	# issue 15
	if config.Thread:
		vs.release()

	# close any open windows
	cv2.destroyAllWindows()


##learn more about different schedules here: https://pypi.org/project/schedule/
if config.Scheduler:
	##Runs for every 1 second
	#schedule.every(1).seconds.do(run)
	##Runs at every day (09:00 am). You can change it.
	schedule.every().day.at("09:00").do(run)

	while 1:
		schedule.run_pending()

else:
	run()
