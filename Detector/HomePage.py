# import the necessary packages
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
import cv2
import dlib
import numpy as np
from scipy.spatial.distance import euclidean


def getFaceAttributeVector(image):
    predictorPath = "Assets/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictorPath)
    dets = detector(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for k, d in enumerate(dets):
        shape = predictor(image, d)

    faceCoord = np.empty([68, 2], dtype=int)

    for b in range(68):
        faceCoord[b][0] = shape.part(b).x
        faceCoord[b][1] = shape.part(b).y

    return faceCoord


def pixelReader(img, startHorizontal, startVertical, height):
    blackColour = []
    for j in range(-int(height * 1.5), 0):
        for i in range(startVertical - height, startVertical + int(height * 1.5)):
            blackLowerRange = [80, 50, 50]
            pixel = startHorizontal + j
            colorCI = img[int(pixel), i]
            if ((colorCI[0] <= blackLowerRange[0] and colorCI[1] <= blackLowerRange[1] and colorCI[2] <=
                 blackLowerRange[2])):
                blackColour.append([int(pixel), i])
    return blackColour


def getEyeCoordinates(image, faceCoord):
    leftEye = image[int(faceCoord[19][1]):int(faceCoord[42][1]), int(faceCoord[36][0]):int(faceCoord[39][0])]
    rightEye = image[int(faceCoord[19][1]):int(faceCoord[42][1]), int(faceCoord[42][0]):int(faceCoord[45][0])]

    eyeLCoordinate = [int(faceCoord[37][0] + int((faceCoord[38][0] - faceCoord[37][0]) / 2)),
                      int(faceCoord[38][1] + int((faceCoord[40][1] - faceCoord[38][1]) / 2))]
    eyeRCoordinate = [int(faceCoord[43][0] + int((faceCoord[44][0] - faceCoord[43][0]) / 2)),
                      int(faceCoord[43][1] + int((faceCoord[47][1] - faceCoord[43][1]) / 2))]

    leftBlackPixel = pixelReader(image, eyeLCoordinate[1], eyeLCoordinate[0],
                                 int((faceCoord[38][0] - faceCoord[37][0]) / 2))
    rightBlackPixel = pixelReader(image, eyeRCoordinate[1], eyeRCoordinate[0],
                                  int((faceCoord[44][0] - faceCoord[43][0]) / 2))
    return leftEye, rightEye, leftBlackPixel, rightBlackPixel


def getPupilPoint(img, blackCoordinates, eyeTopPointX, eyeBottomPointY):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=5,
                               param1=250, param2=10, minRadius=1, maxRadius=-1)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :1]:
            pupilPoint = [int(eyeTopPointX[0]) + i[0], int(eyeBottomPointY[1]) + i[1]]
    else:
        a = 0
        for j, k in blackCoordinates:
            if a == int(len(blackCoordinates) / 2):
                pupilPoint = [k, j]
            a += 1
    return pupilPoint


def openfilename():
    global path, panelA
    # label1.config(text='New Image')
    path = filedialog.askopenfilename(title='Select an image')

    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    if panelA is None:
        panelA = Label(image=image)
        panelA.image = image
        panelA.pack(side="left", padx=5, pady=5)
    else:
        panelA.configure(image=image)
        panelA.image = image

    return path


def center_Eye():
    global panelA, panelB
    # label1.config(text='Center Point of Eye')

    image = cv2.imread(path)
    img = image
    faceVector = getFaceAttributeVector(img)

    leftEye, rightEye, eyeLeftBlackPixels, eyeRightBlackPixels = getEyeCoordinates(img, faceVector)

    leftEyeCoord, eyeBrowCoord = faceVector[36], faceVector[19]
    leftPupilPoint = getPupilPoint(leftEye, eyeLeftBlackPixels, leftEyeCoord, eyeBrowCoord)

    rightEyeCoord = faceVector[42]
    rightPupilPoint = getPupilPoint(rightEye, eyeRightBlackPixels, rightEyeCoord, eyeBrowCoord)

    val1 = list(leftPupilPoint)
    val2 = list(rightPupilPoint)
    closest_dst = euclidean(val1, val2)
    limited_float = "{:.2f}".format(closest_dst)

    label.config(text='Distance between the points : ' + str(limited_float))

    cv2.circle(img, tuple(leftPupilPoint), 5, (0, 0, 255), -1)
    cv2.circle(img, tuple(rightPupilPoint), 5, (0, 0, 255), -1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mark = Image.fromarray(img)
    mark = ImageTk.PhotoImage(mark)

    if panelB is None:
        panelB = Label(image=mark)
        panelB.image = mark
        panelB.pack(side="right", padx=5, pady=5)
    else:
        panelB.configure(image=mark)
        panelB.image = mark


def facial_Landmark():
    global panelA, panelC
    # label1.config(text='Facial Landmark Detection')

    image = cv2.imread(path)
    img = image

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("Assets/shape_predictor_68_face_landmarks.dat")
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        landmarks = predictor(image=gray, box=face)

        x = landmarks.part(27).x
        y = landmarks.part(27).y
        cv2.circle(img, (x, y), 3, (0, 0, 255), thickness=-1)

        for n in range(0, 27):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(img, (x, y), 3, (0, 0, 255), thickness=-1)

        for n in range(36, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(img, (x, y), 3, (0, 0, 255), thickness=-1)

    label.config(text='')

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mark = Image.fromarray(img)
    mark = ImageTk.PhotoImage(mark)

    if panelC is None:
        panelC = Label(image=mark)
        panelC.image = mark
        panelC.pack(side="left", padx=5, pady=5)
    else:
        panelC.configure(image=mark)
        panelC.image = mark


root = Tk()
root.title("Glance-AI")
root.geometry("1485x800")
root.resizable(width=False, height=False)
panelA = None
panelB = None
panelC = None

label2 = Label(root, text="Facial Features for Identifying and Mapping Closed Geometry",
               fg="#000000",
               font=('Courier', 20),
               justify=CENTER)
label2.place(x=290, y=10)

choose_btn = Button(root, text='Select Image', command=openfilename,
                    bg='#000000', fg='#FFFFFF',
                    relief="flat",
                    font=('Courier', 10),
                    activebackground='#4D4D4D', activeforeground='#FFFFFF', padx=5, pady=5)
choose_btn.place(x=200, y=675)

landmark_btn = Button(root, text='Find Detection', command=facial_Landmark,
                      bg='#000000', fg='#FFFFFF',
                      relief="flat",
                      font=('Courier', 10),
                      activebackground='#4D4D4D', activeforeground='#FFFFFF', padx=5, pady=5)
landmark_btn.place(x=675, y=675)

# label1 = Label(root, text="",
#                fg="#000000",
#                font=('Courier', 17),
#                justify=CENTER)
# label1.place(x=600, y=60)

label = Label(root, text="",
              fg="#000000",
              font=('Courier', 14),
              justify=CENTER)
label.place(x=1040, y=720)

eye_btn = Button(root, text='Find Center', command=center_Eye,
                 bg='#000000', fg='#FFFFFF',
                 relief="flat",
                 font=('Courier', 10),
                 activebackground='#4D4D4D', activeforeground='#FFFFFF', padx=5, pady=5)
eye_btn.place(x=1200, y=675)

root.mainloop()
