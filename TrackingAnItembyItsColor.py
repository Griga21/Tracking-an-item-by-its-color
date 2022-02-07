import imutils
import cv2

# Color detection in HSV/ Определение цвета в HSV
colorLower = (35, 100, 100)
colorUpper = (55, 255, 255)

#Connecting the camera / Подключаем камеру
camera = cv2.VideoCapture(0)

# keep looping/ запускаем цикл
while True:
    # grab the current frame / захват текущего кадра
    (grabbed, frame) = camera.read()

    # if the 'q' key is pressed, stop the loop / если нажата клавиша "q", остановите цикл
    if key == ord("q"):
        break

    frame = imutils.resize(frame, width=900)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    #создайте маску для цвета "зеленый",
    # затем выполните серию расширений и эрозий,
    # чтобы удалить все мелкие капли, оставшиеся в маске
    mask = cv2.inRange(hsv, colorLower, colorUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    # найдите контуры в маске
    # и инициализируйте текущий (x, y) центр шара
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    # only proceed if at least one contour was found
    # действуйте только в том случае, если был найден хотя бы один контур
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and centroid
        # найдите самый большой контур в маске,
        # затем используйте его для вычисления минимальной окружности и центра тяжести
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        # действуйте только в том случае, если радиус соответствует минимальному размеру
        if radius > 10:
            # draw the circle and centroid on the frame,
            # нарисуйте круг и центр тяжести на рамке,
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF



# turn off the camera and close all open windows /отключение камеры и закройте все открытые окна
camera.release()
cv2.destroyAllWindows()