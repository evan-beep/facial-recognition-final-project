import cv2
import mediapipe as mp
from imutils import face_utils
import dlib

CIRCLE_RADIUS = 50
CIRCLE_DISTANCE = 140
CIRCLE_TOUCH_COLOR = (0, 0, 255)
FINGERTIP_RADIUS = 0
ANIMATION_LENGTH = 120
IMG_SQUARE_SIDE = 1.4*CIRCLE_RADIUS
PROJECT_FOLDER = '/Users/EvanChen/project/'

current_menu = 'EYES'
menus = {'EYES': {'menu_count': 3}}

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(255, 128, 0), thickness=6)
handConStyle = mpDraw.DrawingSpec(color=(176, 224, 230), thickness=6)

LANDMARKS = {}


def addfilter(image, x_offset, y_offset, imgurl):
    # Get faces into webcam's image
    # For each detected face, find the landmark.

    # Show the image
    filter_image = cv2.imread(
        imgurl, cv2.IMREAD_COLOR)
    filter_image_resized = filter_image  # cv2.resize(filter_image, (200, 700))
    x_end = x_offset + filter_image_resized.shape[1]
    y_end = y_offset + filter_image_resized.shape[0]
    roi = image[y_offset:y_end, x_offset:x_end]

    #image[y_offset:y_end, x_offset:x_end] = bg
    filter_image_gray = cv2.cvtColor(
        filter_image_resized, cv2.COLOR_RGB2GRAY)
    ret, mask = cv2.threshold(
        filter_image_gray, 120, 255, cv2.THRESH_BINARY)
    bg = cv2.bitwise_or(roi, roi, mask=mask)
    mask_inv = cv2.bitwise_not(filter_image_gray)
    fg = cv2.bitwise_and(
        filter_image_resized, filter_image_resized, mask=mask_inv)
    final_roi = cv2.add(bg, fg)
    image[y_offset:y_end, x_offset:x_end] = final_roi
    return image


def drawIcon(icon, center, image):
    new_img = image.copy()
    icon = cv2.imread(icon)

    new_img[center[1]-35:center[1]-35+70,
            center[0]-35:center[0]+35] = icon
    return new_img


def DrawFace(img):
    img = img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    print(len(rects))
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        for n, (x, y) in enumerate(shape):
            img = cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
            LANDMARKS[n+1] = (x, y)
    return img


def DrawMenu(img, frame):
    finger_pos = [0, 0]
    init_pos_y = 75
    image = img.copy()

    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hand_need_result = hands.process(imgRGB)
    if hand_need_result.multi_hand_landmarks:
        for handLms in hand_need_result.multi_hand_landmarks:

            for i, lm in enumerate(handLms.landmark):
                xPos = int(lm.x * imgWidth)
                yPos = int(lm.y * imgHeight)

                if i == 8 and xPos > 0 and yPos > 0:
                    finger_pos = [xPos, yPos]
                    cv2.circle(image, (xPos, yPos), 15,
                               (128, 42, 42), cv2.FILLED)

    if current_menu == 'EYES':
        if(frame < 30):
            for i in range(3):
                imagelink = PROJECT_FOLDER + 'facial/icons/icon_' + \
                    current_menu+'_'+str(i)+'_'+'off.png'
                image = cv2.circle(
                    image, [1150, init_pos_y+i*CIRCLE_DISTANCE-90*i+frame*3*i], CIRCLE_RADIUS+5, (200, 200, 200), -1)
                image = cv2.circle(
                    image, [1150, init_pos_y+i*CIRCLE_DISTANCE-90*i+frame*3*i], CIRCLE_RADIUS, (255, 255, 255), -1)
                image = drawIcon(
                    imagelink, [1150, init_pos_y+i*CIRCLE_DISTANCE-90*i+frame*3*i], image)
            return image
        else:
            for i in range(3):

                if FingerTouch([1150, init_pos_y+i*CIRCLE_DISTANCE], finger_pos):
                    imagelink = PROJECT_FOLDER + 'facial/icons/icon_' + \
                        current_menu+'_'+str(i)+'_'+'off.png'
                    image = cv2.circle(
                        image, [1150, init_pos_y+i*CIRCLE_DISTANCE], CIRCLE_RADIUS+5, (200, 200, 200), -1)

                    image = cv2.circle(
                        image, [1150, init_pos_y+i*CIRCLE_DISTANCE], CIRCLE_RADIUS, (255, 255, 255), -1)
                    image = drawIcon(
                        imagelink, [1150, init_pos_y+i*CIRCLE_DISTANCE], image)
                else:
                    imagelink = PROJECT_FOLDER + 'facial/icons/icon_' + \
                        current_menu+'_'+str(i)+'_'+'on.png'
                    image = cv2.circle(
                        image, [1150, init_pos_y+i*CIRCLE_DISTANCE], CIRCLE_RADIUS+5, (0, 0, 200), -1)

                    image = cv2.circle(
                        image, [1150, init_pos_y+i*CIRCLE_DISTANCE], CIRCLE_RADIUS, CIRCLE_TOUCH_COLOR, -1)
                    image = drawIcon(
                        imagelink, [1150, init_pos_y+i*CIRCLE_DISTANCE], image)
            return image
    else:
        return image


def FingerTouch(circleMid, fingerpos):
    if CIRCLE_RADIUS+FINGERTIP_RADIUS < ((circleMid[0] - fingerpos[0])**2 + (circleMid[1] - fingerpos[1])**2)**0.5:
        return True
    else:
        return False


cap = cv2.VideoCapture(1)
fy = 0
while True:
    ret, image = cap.read()
    if not ret:
        break

    image = cv2.flip(image, 1)

    imgHeight = image.shape[0]
    imgWidth = image.shape[1]
    DrawFace(image)

    # Code放這下面，上面別動

    image = DrawMenu(image, fy)
    # image = addfilter(
    #    image, 0, 0, "/Users/EvanChen/project/facial/glass/glass2.jpg")
    cv2.imshow("Output", image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    if(fy < 1000):
        fy += 2
    # Q
    if k == 113:
        fy = 0
        current_menu = 'EYES'

cv2.destroyAllWindows()
cap.release()
