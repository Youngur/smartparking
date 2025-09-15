import cv2
import numpy as np

# -------------------------------
# Пути к YOLOv3 файлам
# -------------------------------
cfg_file = r"data/yolov3.cfg"
weights_file = r"data/yolov3.weights"
names_file = r"data/coco.names"

# -------------------------------
# Загрузка модели YOLOv3
# -------------------------------
net = cv2.dnn.readNet(weights_file, cfg_file)
with open(names_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# -------------------------------
# Загрузка изображения парковки
# -------------------------------
img_path = r"data/parking_lot.jpg"
img = cv2.imread(img_path)
if img is None:
    print("Ошибка: картинка не найдена!")
    exit()

height, width = img.shape[:2]

# -------------------------------
# Подготовка изображения для YOLO
# -------------------------------
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
outs = net.forward(output_layers)

# -------------------------------
# Получение bounding boxes машин
# -------------------------------
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.3 and classes[class_id] == "car":
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w/2)
            y = int(center_y - h/2)
            boxes.append([x, y, w, h])
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)

# -------------------------------
# Настройка парковочных мест (пример)
# -------------------------------
parking_spaces = [
    (50, 200, 150, 300),
    (170, 200, 270, 300),
    (290, 200, 390, 300),
    (410, 200, 510, 300)
]

# -------------------------------
# Проверка свободных мест
# -------------------------------
free_spaces = []
for i, (x1, y1, x2, y2) in enumerate(parking_spaces):
    occupied = False
    for bx, by, bw, bh in boxes:
        if not (bx+bw < x1 or bx > x2 or by+bh < y1 or by > y2):
            occupied = True
            break
    if not occupied:
        free_spaces.append(i+1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    else:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

print("Свободные места:", free_spaces)
if free_spaces:
    print(f"Езжайте на место №{free_spaces[0]}")
else:
    print("Все места заняты!")

cv2.imshow("Smart Parking", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
