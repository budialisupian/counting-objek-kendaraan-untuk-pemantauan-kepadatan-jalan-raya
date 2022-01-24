# USAGE

# import package yang diperlukan
import numpy as np
from firebase import firebase
import imutils
import time
import cv2
import os
# import psycopg2
import datetime

savename = []
car = 0
bus = 0
truck = 0
motorbike = 0
car_ = []
bus_ = []
truck_ = []
motorbike_ = []

frames = 0

# try:
#     connection = psycopg2.connect(user = "postgres",
#                                   password = "12345678",
#                                   host = "127.0.0.1",
#                                   port = "5432",
#                                   database = "detection_vehicle")
#
#     cursor = connection.cursor()
# 	# print ( connection.get_dsn_parameters(),"\n")
#
#     # Print PostgreSQL version
#     cursor.execute("SELECT version();")
#     record = cursor.fetchone()
#     print("You are connected to - ", record,"\n")
#
# except (Exception, psycopg2.Error) as error :
#     print ("Error while connecting to PostgreSQL", error)

# membuat koneksi ke database firebase
firebase = firebase.FirebaseApplication("https://vehicle-1d2dd.firebaseio.com/", None)


def adjust_gamma(image, gamma=1.0):
    # buat nilai pencarian untuk memetakan nilai piksel 0 - 255
    # Kemudian menyesuaikan nilai gamma
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # menerapkan koreksi gamma menggunakan tabel pemcarian
    return cv2.LUT(image, table)


# memuatkan label kelas coco dari file training yolo
labelsPath = os.path.join("yolov-coco", "coco.names")
LABELS = open(labelsPath).read().strip().split("\n")

# inisialisasi daftar warna yang munkin mewakili label kelas yang digunakan
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 4),
                           dtype="uint8")

# memuat sortcut ke bobot yolo dan mengkonfigurasi model hasil training
weightsPath = os.path.join("yolov-coco", "yolov3.weights")
configPath = os.path.join("yolov-coco", "yolov3.cfg")

# memuat objek yolo yang dilatih dari class coco
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# menentukan file video yang akan diproses
# dan menentukan dimens frame dari video
vs = cv2.VideoCapture("fix_sample_2.mp4")
writer = None
(W, H) = (None, None)

font = cv2.FONT_HERSHEY_SIMPLEX

# mencoba menentukan jumlah total frame di file video
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))

# menampilkan tulisan apabila terjadi kesalahan dalam menentukan total frame pada video
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1

# melakukan looping dari frame video
while True:

    checker = True
    # initialize waktu
    date_time = datetime.datetime.now()

    total_vehicle = []
    # baca frame selanjutnya dari file video
    (grabbed, frame) = vs.read()
    # kondisi jika frame tidak diambil makan selesai
    if not grabbed:
        break

    if W is None or H is None:
        (H, W) = frame.shape[:2]

        print(str(H), str(W))

    frame = adjust_gamma(frame, gamma=1.5)
    # membuat blob (binary large object atau melakukan thresholding)
    # jika lulu atau berhasil terdeteksi dari objek yolo, beri kotak pembatas
    # dan tentukan probabilitasnya
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # inisialisasi kotak pembatas yang terdeteksi, nilai akurasi
    # dan id kelas masing-masing
    boxes = []
    confidences = []
    classIDs = []
    center = []

    # melakukan looping setiap layer output
    for output in layerOutputs:
        # melakukan looping setiap deteksi
        for detection in output:
            # ekstrak id kelas dan nilai akurasi
            # dari objek yang terdeteksi
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # menentukan nilai minimal probabilitas untuk objek
            if confidence > 0.5:
                # menskalakan kembali koordinat kotak pembatas terhadap
                # ukuran gambar, ingat bahwa YOLO
                # benar-benar mengembalikan koordinat pusat (x, y) dari
                # kotak pembatas diikuti dengan lebar kotak dan
                # tinggi
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # gunakan koordinat pusat (x, y) untuk menurunkan puncak
                # dan pojok kiri kotak pembatas
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # perbarui daftar koordinat kotak pembatas kami,
                # niali akurasi, dan ID class
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # menerapkan penekanan non-maxima untuk tumpang tindih
    # kotak pembatas
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,
                            0.3)

    # memastikan setidaknya ada satu yang tedeteksi atau > 0
    if len(idxs) > 0:
        # mempertahankan indeks
        for i in idxs.flatten():
            if LABELS[classIDs[i]] == "car" or LABELS[classIDs[i]] == "bus" or LABELS[classIDs[i]] == "truck" or LABELS[
                classIDs[i]] == "motorbike":
                # ekstrak koordinat kotak pembatas
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # menggambar kotak persegi panjang pembatas dan label pada frame
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                                           confidences[i])
                cv2.putText(frame, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # cv2.line(frame, (100,200), (350,200), [0, 255, 0], 2)
                total_vehicle.append(LABELS[classIDs[i]])

                if LABELS[classIDs[i]] == "car" and y + h > 650 and y + h < 750:
                    car_.append(y + h)
                    print("car_coordinate", y + h)
                    if LABELS[classIDs[i]] == "car" and y + h > 720 and len(car_) > 5:
                        car = car + 1
                        # postgres_insert_query = """ INSERT INTO detection_vehicle (date_time, vehicle_type, confidence, weight) VALUES (%s,%s,%s,%s)"""
                        # record_to_insert = (str(date_time), "car", confidences[i], 1590)
                        # cursor.execute(postgres_insert_query, record_to_insert)
                        # connection.commit()
                        data = {
                            "date_time": str(date_time),
                            "vehicle_type": "car",
                            "confidence": str(confidences[i]),
                            "weight": str(1590),
                            "total_car": str(car)
                        }

                        result = firebase.post("/vehicle-1d2dd/Costumer", data)
                        print(result)
                        print("finish car")
                        car_ = []

                if LABELS[classIDs[i]] == "bus" and y + h > 650 and y + h < 750:
                    bus_.append(y + h)
                    print("bus_coordinate", y + h)
                    if LABELS[classIDs[i]] == "bus" and y + h > 720 and len(bus_) > 5:
                        bus = bus + 1
                        # postgres_insert_query = """ INSERT INTO detection_vehicle (date_time, vehicle_type, confidence, weight) VALUES (%s,%s,%s,%s)"""
                        # record_to_insert = (str(date_time), "bus", confidences[i], 8060)
                        # cursor.execute(postgres_insert_query, record_to_insert)
                        # connection.commit()
                        data = {
                            "date_time": str(date_time),
                            "vehicle_type": "bus",
                            "confidence": str(confidences[i]),
                            "weight": str(8060),
                            "total_bus": str(bus)
                        }

                        result = firebase.post("/vehicle-1d2dd/Costumer", data)
                        print(result)
                        bus_ = []

                if LABELS[classIDs[i]] == "truck" and y + h > 650 and y + h < 750:
                    truck_.append(y + h)
                    print("truck_coordinate", y + h)
                    if LABELS[classIDs[i]] == "truck" and y + h > 720 and len(truck_) > 5:
                        truck = truck + 1
                        # postgres_insert_query = """ INSERT INTO detection_vehicle (date_time, vehicle_type, confidence, weight) VALUES (%s,%s,%s,%s)"""
                        # record_to_insert = (str(date_time), "truck", confidences[i], 2560)
                        # cursor.execute(postgres_insert_query, record_to_insert)
                        # connection.commit()
                        data = {
                            "date_time": str(date_time),
                            "vehicle_type": "truck",
                            "confidence": str(confidences[i]),
                            "weight": str(2560),
                            "total_truck": str(truck)
                        }

                        result = firebase.post("/vehicle-1d2dd/Costumer", data)
                        print(result)
                        truck_ = []

                if LABELS[classIDs[i]] == "motorbike" and y + h > 650 and y + h < 750:
                    motorbike_.append(y + h)
                    print("motor_coordinate", y + h)
                    if LABELS[classIDs[i]] == "motorbike" and y + h > 720 and len(motorbike_) > 5:
                        motorbike = motorbike + 1
                        # postgres_insert_query = """ INSERT INTO detection_vehicle (date_time, vehicle_type, confidence, weight) VALUES (%s,%s,%s,%s)"""
                        # record_to_insert = (str(date_time), "motorbike", confidences[i], 170)
                        # cursor.execute(postgres_insert_query, record_to_insert)
                        # connection.commit()
                        data = {
                            "date_time": str(date_time),
                            "vehicle_type": "motorbike",
                            "confidence": str(confidences[i]),
                            "weight": str(170),
                            "total_motorbike": str(motorbike)
                        }

                        result = firebase.post("/vehicle-1d2dd/Costumer", data)
                        print(result)
                        motorbike_ = []
                        print("finish motor")

            ## for case counting vehicle in parking
            # total_detect = len(total_vehicle)
            # car = len([p for p in total_vehicle if p == 'car'])
            # truck = len([p for p in total_vehicle if p == 'truck'])
            # bus = len([p for p in total_vehicle if p == 'bus'])
            # motorbike = len([p for p in total_vehicle if p == 'motorbike'])
            # bicycle = len([p for p in total_vehicle if p == 'bicycle'])

            cv2.rectangle(frame, (1200, 20), (1400, 120), (180, 132, 109), -1)
            cv2.putText(frame, 'Car: ' + str(car), (1200, 40), font, 0.5, (0, 0xFF, 0xFF), 1, cv2.FONT_HERSHEY_SIMPLEX)
            cv2.putText(frame, 'Truck: ' + str(truck), (1200, 60), font, 0.5, (0, 0xFF, 0xFF), 1, cv2.FONT_HERSHEY_SIMPLEX)
            cv2.putText(frame, 'Bus: ' + str(bus), (1200, 80), font, 0.5, (0, 0xFF, 0xFF), 1, cv2.FONT_HERSHEY_SIMPLEX)
            cv2.putText(frame, 'motorbike: ' + str(motorbike), (1200, 100), font, 0.5, (0, 0xFF, 0xFF), 1, cv2.FONT_HERSHEY_SIMPLEX)
            cv2.line(frame, (750, 700), (1700, 650), [0, 255, 0], 2)

    else:
        cv2.rectangle(frame, (1200, 20), (1400, 120), (180, 132, 109), -1)
        cv2.putText(frame, 'Car: ' + str(car), (1200, 40), font, 0.5, (0, 0xFF, 0xFF), 1, cv2.FONT_HERSHEY_SIMPLEX)
        cv2.putText(frame, 'Truck: ' + str(truck), (1200, 60), font, 0.5, (0, 0xFF, 0xFF), 1, cv2.FONT_HERSHEY_SIMPLEX)
        cv2.putText(frame, 'Bus: ' + str(bus), (1200, 80), font, 0.5, (0, 0xFF, 0xFF), 1, cv2.FONT_HERSHEY_SIMPLEX)
        cv2.putText(frame, 'motorbike: ' + str(motorbike), (1200, 100), font, 0.5, (0, 0xFF, 0xFF), 1, cv2.FONT_HERSHEY_SIMPLEX)
        cv2.line(frame, (750, 700), (1700, 650), [0, 255, 0], 2)

    # if you want to see window realtime video
    # cv2.imshow('vehicle counting',frame)
    # key = cv2.waitKey(1) & 0xFF

    # # if the `q` key was pressed, break from the loop
    # if key == ord("q"):
    # 	break

    # if you want to write output video
    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter("output_yolo.mp4", fourcc, 30,
                                 (frame.shape[1], frame.shape[0]), True)

        # some information on processing single frame
        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(
                elap * total))

    # write the output frame to disk
    writer.write(frame)

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()