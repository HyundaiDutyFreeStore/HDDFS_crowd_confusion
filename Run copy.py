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

	# 학습된 데이터를 가져온다.
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

	# tracker를 초기화
	# dlib tracker에 인덱스 붙이기
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
		status = "Waiting"
		rects = []

		# tracker가 물체를 감지함
		if totalFrames % args["skip_frames"] == 0:
			# 현재 상태를 탐지으로 설정
			status = "Detecting"
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
				status = "Tracking"

				# tracker 업데이트 및 위치 업데이트
				tracker.update(rgb)
				pos = tracker.get_position()

				# 객체 위치 저장
				startX = int(pos.left())
				startY = int(pos.top())
				endX = int(pos.right())
				endY = int(pos.bottom())

				# 추적 중인 객체들에 직사각형 모양 경계 그려주기
				rects.append((startX, startY, endX, endY))

		# 프레임 중앙에 수평선을 그리고 객체가 이 선을 넘으면 in out 판단
		cv2.line(frame, (W // 2, 0), (W // 2, H), (0, 0, 0), 3)
		cv2.putText(frame, "-Prediction border - Entrance-", (10, H - ((i * 20) + 200)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

		# tracker로 이전 객체 연결
		# 새로 탐색된 객체도 연결
		objects = ct.update(rects)

		# loop over the tracked objects
		for (objectID, centroid) in objects.items():
			# 현재 추적 가능한 개체가 있는지 확인
			to = trackableObjects.get(objectID, None)

			# 기존 추적 가능한 개체가 없는 경우 새로 만듬
			if to is None:
				to = TrackableObject(objectID, centroid)

			# 추적가능한 객체가 있다면 방향 결정
			else:
				# 현재 좌표를 이용하여 물체가 움직이는 방향 판단
				y = [c[1] for c in to.centroids]
				direction = centroid[1] - np.mean(y)
				to.centroids.append(centroid)

				# 객체의 숫자가 카운트 되었는지 확인
				if not to.counted:
					# 방향이 아래이고 중심선을 지나면 up의 숫자를 ++
					if direction < 0 and centroid[1] < H // 2:
						totalUp += 1
						empty.append(totalUp)
						to.counted = True

					# 방향이 위고 중심선을 지나면 down의 숫자를 ++
					elif direction > 0 and centroid[1] > H // 2:
						totalDown += 1
						empty1.append(totalDown)
						to.counted = True

			# 추적 가능한 개체를 저장
			trackableObjects[objectID] = to

			# 출력 프레임에 개체의 ID와 개체의 중심을 모두 그림
			text = "ID {}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

		# 창에 표시할 정보들 입력
		info = [
		("Exit", totalUp),
		("Enter", totalDown),
		("Status", status),
		]

        # 출력표시
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

		# log.csv 파일로 저장
		if config.Log:
			d = [empty1]
			export_data = zip_longest(*d, fillvalue = '')

			with open('loginside.csv', 'w', newline='') as myfile:
				wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
				#wr.writerow(("In", "Out"))
				wr.writerows(export_data)
				# log.csv 파일로 저장

		if config.Log:
			d = [empty]
			export_data = zip_longest(*d, fillvalue = '')

			with open('logoutside.csv', 'w', newline='') as myfile:
				wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
				#wr.writerow(("In", "Out"))
				wr.writerows(export_data)
				
		# 프레임 저장
		if writer is not None:
			writer.write(frame)

		# 프레임 윈도우에 표시
		cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
		key = cv2.waitKey(1) & 0xFF

		# q 누르면 꺼지게 설정
		if key == ord("q"):
			break

		# 총 프레임수 1 올리고 fps업데이트
		totalFrames += 1
		fps.update()

		if config.Timer:
			# 라이브 스트리밍 자동 중지 시간 설정(8시간)
			t1 = time.time()
			num_seconds=(t1-t0)
			if num_seconds > 28800:
				break

	# 종료하면서 실행시간 표시
	fps.stop()
	print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	
	if config.Thread:
		vs.release()

	cv2.destroyAllWindows()

if config.Scheduler:
	#1초마다 실행

	while 1:
		schedule.run_pending()

else:
	run()