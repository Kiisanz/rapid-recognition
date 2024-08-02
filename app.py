import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import time

# Path ke file bobot dan konfigurasi YOLO
weights_path = 'yolov3.weights'
config_path = 'yolov3.cfg'
names_path = 'coco.names'

# Menggunakan YOLO untuk deteksi objek
net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Muat nama kelas
with open(names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Fungsi untuk mengonversi piksel ke meter (berdasarkan kalibrasi)
def pixel_to_meter(pixels):
    # Misalnya, 1 piksel = 0.05 meter (harus disesuaikan berdasarkan kalibrasi kamera)
    return pixels * 0.05

# Kalman Filter Class
class KalmanFilter:
    def __init__(self):
        self.dt = 1.0  # Waktu antar frame
        self.A = np.array([[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]])  # State transition matrix
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  # Measurement matrix
        self.P = np.eye(4)  # Covariance matrix
        self.R = np.eye(2)  # Measurement noise
        self.Q = np.eye(4)  # Process noise
        self.x = np.zeros((4, 1))  # Initial state

    def predict(self):
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x[:2]

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.H.shape[1])
        self.P = np.dot(I - np.dot(K, self.H), self.P)

# Fungsi untuk menghitung jarak Euclidean
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

# Memuat video
cap = cv2.VideoCapture('video.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

# Menyimpan posisi sebelumnya
trackers = []
track_id = 0

# Ukuran baru frame untuk 360p (16:9)
resize_width = 640
resize_height = 360

# Fungsi untuk mengonversi kecepatan dari piksel per detik ke meter per detik
def pxs_to_mps(speed_pxs, pixel_to_meter):
    return speed_pxs * pixel_to_meter

# Loop utama
prev_positions = {}
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, (resize_width, resize_height))

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Menginisialisasi daftar deteksi
    detections = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 2:  # Hanya mobil
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                detections.append((x, y, x + w, y + h, confidence))

    # Memprediksi posisi baru menggunakan Kalman Filter
    predictions = []
    for tr in trackers:
        prediction = tr.predict()
        predictions.append(prediction.ravel())

    # Asosiasi deteksi dengan prediksi menggunakan Hungarian Algorithm
    col_ind = []  # Initialize col_ind
    if len(predictions) > 0 and len(detections) > 0:
        cost_matrix = np.zeros((len(predictions), len(detections)))
        for i, pred in enumerate(predictions):
            for j, det in enumerate(detections):
                center_det = np.array([(det[0] + det[2]) / 2, (det[1] + det[3]) / 2])
                cost_matrix[i, j] = euclidean_distance(pred, center_det)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        new_trackers = []
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < 100:  # Threshold untuk asosiasi
                center_det = np.array([(detections[j][0] + detections[j][2]) / 2, (detections[j][1] + detections[j][3]) / 2])
                trackers[i].update(center_det.reshape(-1, 1))
                new_trackers.append(trackers[i])

                x1, y1, x2, y2, _ = detections[j]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f'ID {i}', (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

                # Hitung kecepatan
                if i in prev_positions:
                    prev_x, prev_y, prev_time = prev_positions[i]
                    current_x, current_y = center_det
                    delta_time = time.time() - prev_time
                    speed_pxs = euclidean_distance(np.array([prev_x, prev_y]), np.array([current_x, current_y])) / delta_time
                    speed_mps = pxs_to_mps(speed_pxs, 0.05)  # Gantilah 0.05 dengan nilai kalibrasi piksel ke meter
                    cv2.putText(frame, f'Speed: {speed_mps:.2f} m/s', (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                # Simpan posisi dan waktu sekarang
                prev_positions[i] = (center_det[0], center_det[1], time.time())
        trackers = new_trackers

    # Menambahkan deteksi yang tidak terasosiasi sebagai pelacak baru
    for i in range(len(detections)):
        if i not in col_ind:
            kf = KalmanFilter()
            center_det = np.array([(detections[i][0] + detections[i][2]) / 2, (detections[i][1] + detections[i][3]) / 2])
            kf.x[:2] = center_det.reshape(-1, 1)
            trackers.append(kf)

    # Menampilkan frame
    cv2.imshow('Frame', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
