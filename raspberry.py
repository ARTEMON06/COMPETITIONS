import math
import smbus
import numpy
import datetime
from cv2 import cv2
from tensorflow.python.keras.models import load_model


#  НАСТРОЙКИ
bus = smbus.SMBus(1)
SLAVE_ADDRESS = 0x8
LINE_WIDTH_NORMAL = (120, 170)
MINSIZE_W = 40
MINSIZE_H = 55
MINSIZE_W_S = 65
MINSIZE_H_S = 65
camera_height = 480
camera_width = 640
settings = (100, 70, 140)
IS_RED = False
LOW_SPEED_FLAG = False
STOP_FLAG = False
time_pedestrian = datetime.datetime.now()
time_stop = datetime.datetime.now()
LIGHT_FLAG = True
light_counter = 0

I2C = {
    'wheel': 45,
    'speed': 0
}

#  Названия знаков
SIGN_CLASSIFICATION = ("Кирпич", "Пешеход", "Стоп", "Внимание", "Прямо", "Право", "Лево")
MODEL = load_model("22_1.h5")

# Получение видео
camera = cv2.VideoCapture(0)

#  Класс создания окна настроек (ползунки для цветокоррекции)
class Settings:
    def __init__(self, window_name, start_settings):
        self.window_name = window_name
        cv2.namedWindow(window_name)  # создаем окно настроек

        cv2.createTrackbar('minb', self.window_name, start_settings[0], 255, lambda x: None)
        cv2.createTrackbar('ming', self.window_name, start_settings[1], 255, lambda x: None)
        cv2.createTrackbar('minr', self.window_name, start_settings[2], 255, lambda x: None)
        cv2.createTrackbar('maxb', self.window_name, 255, 255, lambda x: None)
        cv2.createTrackbar('maxg', self.window_name, 255, 255, lambda x: None)
        cv2.createTrackbar('maxr', self.window_name, 255, 255, lambda x: None)

    def get_settings(self):
        minb = cv2.getTrackbarPos('minb', self.window_name)
        ming = cv2.getTrackbarPos('ming', self.window_name)
        minr = cv2.getTrackbarPos('minr', self.window_name)

        maxb = cv2.getTrackbarPos('maxb', self.window_name)
        maxg = cv2.getTrackbarPos('maxg', self.window_name)
        maxr = cv2.getTrackbarPos('maxr', self.window_name)

        return (minb, ming, minr), (maxb, maxg, maxr)

    def show_settings_and_mask(self, mask):
        cv2.imshow(self.window_name, mask)


# Функция определения знака
def prediction(frame, x, y, image_copy1):
    image = numpy.expand_dims(frame, axis=0)
    image = numpy.array(image)
    pred = MODEL.predict(image)
    kk = numpy.argmax(pred, axis=1)
    res = "none"
    accuracy = 0
    if pred[0][kk[0]] > 0.90:
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (x, y-20)
        fontScale = 1
        fontColor = (0, 255, 0)
        lineType = 1

        cv2.putText(frame, SIGN_CLASSIFICATION[kk[0]],
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        # print(f"{SIGN_CLASSIFICATION[kk[0]]} - {pred[0][kk[0]]}")  # Первое - результат, второе - точность [0;1]

        accuracy = pred[0][kk[0]]
        res = str(SIGN_CLASSIFICATION[kk[0]])
    return res, accuracy


# Отрисовка точек линии дороги (для корректировки значений)
def draw_dot(frame, text, position):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (position - 35, camera_height - 33)
    fontScale = 0.7
    fontColor = (0, 0, 255)
    lineType = 1

    cv2.putText(frame, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

    start_point = (position - 2, camera_height - 18)
    end_point = (position + 2, camera_height - 6)
    color = (0, 0, 255)
    thickness = -1
    cv2.rectangle(frame, start_point, end_point, color, thickness)


# Детектирование линии
def line_1(frame):
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    setting = cv2.getTrackbarPos('setting', "line")
    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, setting, 255, cv2.THRESH_BINARY)
    cv2.imshow("line", blackAndWhiteImage)

    crop_img = blackAndWhiteImage[camera_height-10:camera_height-9, 0:camera_width]
    mass = []
    for i in range(camera_width):
        mass.append(1) if crop_img[0, i] == 0 else mass.append(0)

    left = 0
    right = 0

    for i in range(len(mass)):
        if mass[i] == 1:
            right = i
    for i in range(len(mass)):
        if mass[len(mass) - i - 1] == 1:
            left = i

    line_center = right + (640 - right - left) // 4

    # print(f"Line left: {camera_width-left}     Line right: {right}     Car position: {line_center}")

    # Отрисовка точек линии
    draw_dot(frame, 'right', right)
    draw_dot(frame, 'left', camera_width-left)
    draw_dot(frame, 'car', line_center)

    return line_center, frame


# Трансформация изображений со знаком
def order_points(pts):
    rect = numpy.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[numpy.argmin(s)]
    rect[2] = pts[numpy.argmax(s)]
    diff = numpy.diff(pts, axis=1)
    rect[1] = pts[numpy.argmin(diff)]
    rect[3] = pts[numpy.argmax(diff)]
    return rect


# Исправление перспективы знака по точкам для лучшего распознавания
def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = numpy.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = numpy.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = numpy.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = numpy.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = numpy.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


# Определение цвета светофора по точкам
def spot_light(c, image_copy1):
    x, y, w, h = cv2.boundingRect(c)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = numpy.int0(box)
    peri = cv2.arcLength(box, True)
    approx = cv2.approxPolyDP(box, 0.02 * peri, True)

    res = 'nothing'

    if len(approx) == 4:
        paper = four_point_transform(frame, approx.reshape(4, 2))
        paper = cv2.resize(paper, (48, 48))

        counter_red = 0
        counter_green = 0

        for i in range(10, 30):
            for j in range(10, 30):
                pixel = paper[i][j]
                if pixel[2] > pixel[1] and pixel[2] > pixel[0]:
                    counter_red += pixel[2]
                if pixel[1] > pixel[0] and pixel[1] > pixel[2]:
                    counter_green += pixel[1]

        # Сравнение подсчитанных значений
        if counter_green > counter_red:
            res = 'GREEN'
        else:
            res = 'RED'


        cv2.rectangle(image_copy1, (x, y), (x + w, y + h), (255, 255, 255), 3)
        # print(f"{counter_red}, {counter_green}")

    return res


# Нахождение контура светофора
def light_1(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hsv = cv2.blur(hsv, (5, 5))

    prop = settings_light.get_settings()

    mask = cv2.inRange(hsv, *prop)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=4)

    cv2.imshow("settings light", mask)

    contours1, hierarchy1 = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    color = 'nothing'

    # Если контуры найдены, то определяем цвет светофора
    if contours1:
        c = max(contours1, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        if w > MINSIZE_W and h > MINSIZE_H:
            color = spot_light(c, frame)
    #     cv2.imshow('Simple approximation', image_copy1)
    # cv2.imshow('settings light', mask)

    return color, frame


# Нахождение контуров знака
def sign_1(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.blur(hsv, (5, 5))

    prop = settings_sign.get_settings()
    mask = cv2.inRange(hsv, *prop)

    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=5)

    cv2.imshow("settings sign", mask)

    contours1, hierarchy1 = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result = 'none'

    accuracy = 0
    # Если что-то нашли, то обрабатываем
    if contours1:
        c = max(contours1, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        # Если знак подходящего размера, регулировка размера определяет дальность срабатывания
        if w > MINSIZE_W_S and h > MINSIZE_H_S:
            x, y, w, h = cv2.boundingRect(c)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = numpy.int0(box)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            peri = cv2.arcLength(box, True)
            approx = cv2.approxPolyDP(box, 0.02 * peri, True)

            # Если похоже на контуры знака, трансформируем, обрезаем и передаем нейронке
            if len(approx) == 4:
                docCnt = approx
                paper = four_point_transform(frame, docCnt.reshape(4, 2))
                # cv2.imshow('paper', paper)
                result, accuracy = prediction(cv2.resize(paper, (48, 48)), x, y, frame)

    return result, accuracy, frame

# Создаем настройки
cv2.namedWindow("line")
cv2.createTrackbar('setting', "line", 200, 255, lambda x: None)
settings_light = Settings('settings light', (20, 50, 135))
settings_sign = Settings('settings sign', (100, 70, 160))

# Бесконечный цикл...
while True:
    ret, frame = camera.read()

    copy_frame = frame.copy()

    # Получение данных
    result_line, line = line_1(copy_frame)
    color_light, light = light_1(copy_frame)
    sign_result, accuracy, sign = sign_1(copy_frame)

    cv2.imshow("RESULT", copy_frame)

    I2C['wheel'] = math.trunc((result_line/640) * 90)  # Преобразование диапазонов

    # Логика определения движения

    # Принудительное отклонение руля при знаках
    if sign_result == 'Лево':
        I2C['wheel'] += 10

    if sign_result == 'Право':
        I2C['wheel'] -= 10

    # Увидели пешеходный знак, включили нужный режим
    if sign_result == 'Пешеход' and not LOW_SPEED_FLAG:
        LOW_SPEED_FLAG = True
        LIGHT_FLAG = True
        time_pedestrian = datetime.datetime.now()

    # Выключили через время проезда пешеходного перехода
    if (datetime.datetime.now() - time_pedestrian).total_seconds() >= 13:
        LOW_SPEED_FLAG = False

    # Увидели знак стоп, остановились
    if sign_result == 'Стоп' and not STOP_FLAG and accuracy > 0.98:
        STOP_FLAG = True
        LIGHT_FLAG = True
        time_stop = datetime.datetime.now()

    # Через 3 секунды после знака стоп продолжаем движение
    if (datetime.datetime.now() - time_stop).total_seconds() >= 3:
        STOP_FLAG = False

    # Красный сигнал -> режим остановки
    if color_light == 'RED' and accuracy < 9.8:
        IS_RED = True
        LIGHT_FLAG = True

    # Зеленый сигнал -> отключение режима остановки, подсчет зеленого кол-ва зеленых светофоров
    if color_light == 'GREEN' and accuracy < 9.8:
        IS_RED = False
        if LIGHT_FLAG:
            light_counter += 1
            LIGHT_FLAG = False

    # Условия скорости
    if LOW_SPEED_FLAG:  # Если пешеходный
        I2C['speed'] = 80
    elif STOP_FLAG or IS_RED or light_counter >= 4:  # Если стоп, красный светофор или финиш (4ый зеленый светофор)
        I2C['speed'] = 0
    else:  # обычная скорость
        I2C['speed'] = 120

    result_line = f'{I2C["wheel"] + 100}{I2C["speed"] + 100}*'  # команда для передачи по I2C
    for item in str(result_line):
        bus.write_byte(SLAVE_ADDRESS, ord(item))

    # print(f"GREEN: {light_counter}, I2C: {result_line}, LOW: {LOW_SPEED_FLAG}, STOP: {STOP_FLAG}, "
    #       f"COLOR: {color_light}, SIGN: {sign_result}, {accuracy}")

    # Кнопка закрытия и выхода
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
camera.release()

# made by Tema and Denis
