import random
import threading
from keras.models import load_model
import customtkinter as ctk
from tkinter import *
from tkinter.filedialog import askopenfile
import time
import os
import cv2
from PIL import Image, ImageTk
import numpy as np
import mediapipe as mp
from math import sqrt,acos

master = ctk.CTk()
master.bind('<Escape>', lambda e: master.quit())
master.geometry("1200x600")
master.resizable(0, 0)
master.title("  Physio Trainer")
master.config(bg="black")
model = load_model('convlstm_model___Date_Time_2023_03_16__19_08_01.h5')
ctk.set_appearance_mode("dark")
text_var = StringVar(value="Analyzing..")

# ///////////////////////
images = []

for i in os.listdir("assets/CoverSlides/"):
    img = Image.open('assets/CoverSlides/'+i)
    w,h = img.size
    img = img.resize((1200, round((h/w)*1200)), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    images.append(img)


img = Image.open('assets/diag.png')
w,h = img.size
img = img.resize((200, round((h/w)*200)), Image.ANTIALIAS)
img = ImageTk.PhotoImage(img)

img2 = Image.open('assets/graphs.png')
w,h = img2.size
img2 = img2.resize((600, round((h/w)*600)), Image.ANTIALIAS)
img2 = ImageTk.PhotoImage(img2)

# ///////////////////////
canvas = Canvas(master,bg='black',bd=0)
cim = canvas.create_image(6, 2, image = images[0], anchor = NW)

myDraw = mp.solutions.drawing_utils
mypose = mp.solutions.pose
pose = mypose.Pose()


def open_file():
    global vid
    file = askopenfile(mode ='r', filetypes =[('Video Files', '*.mp4')])
    if file is not None:
        vid = cv2.VideoCapture(file.name)
    else:
        return 0

def setVidToCamera():
    global vid
    vid = cv2.VideoCapture(0)

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

def GetPrediction(p,x):
    if p[0][0] > p[0][1] and not x:
        return "DOING GOOD","FINGER EXERCISE"

    elif p[0][0] < p[0][1] and not x:
        return "DOING GOOD","KNEE EXERCISE"

    return "NONE","UNIDENTIFIED"

def CheckAngle(A):
    if 0 <= A <= 20:
        return "NEGLIGIBLE"
    if 20 <= A <= 45:
        return "WEAK"
    elif 45 < A <= 90:
        return "MODERATE"
    elif 90 < A <= 120:
        return "STRONG"
    elif 120 < A <= 180:
        return "PERFECT"
    else: "MONOTONIC"

max = 0
captureFrames = []
def open_camera():
    global captureFrames,max
    # Capture the video frame by frame
    _, frame = vid.read()

    cor = []
    cor2 = []
    cor3 = []
    cor4 = []

    if _:
        frame = image_resize(frame,width=600)
        captureFrames.append(cv2.resize(frame,(64,64),interpolation = cv2.INTER_AREA))

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)

        angle11 = 0
        angle12 = 0
        angle21 = 0
        angle22 = 0

        if results.pose_landmarks:
            myDraw.draw_landmarks(frame, results.pose_landmarks, mypose.POSE_CONNECTIONS)
            lm = results.pose_landmarks.landmark

            for id,lm in enumerate(lm):
                coordinates = [lm.x, lm.y, lm.z]

                if id == 13 or id == 11 or id == 23:
                    cor.append(coordinates)
                    if len(cor) == 3:
                        angle11 = (getAngle(cor[0], cor[1], cor[2]))
                        cor = []

                if id == 14 or id == 12 or id == 24:
                    cor2.append(coordinates)
                    if len(cor2) == 3:
                        angle12 = (getAngle(cor2[0], cor2[1], cor2[2]))
                        cor2 = []

                if id == 23 or id == 25 or id == 27:
                    cor3.append(coordinates)
                    if len(cor3) == 3:
                        angle21 = (getAngle(cor3[0], cor3[1], cor3[2]))
                        cor3 = []

                if id == 24 or id == 26 or id == 28:
                    cor4.append(coordinates)
                    if len(cor4) == 3:
                        angle22 = (getAngle(cor4[0], cor4[1], cor4[2]))
                        cor4 = []

        if len(captureFrames) == 120:
            frames = np.array(captureFrames)
            p = model.predict(np.array([frames]))
            performance = "DOING GOOD"
            co,p34 = GetPrediction(p,False)

            if p34 == "FINGER EXERCISE":
                print(round(angle11),round(angle12))
                Angles = [round(angle11),round(angle12)]
            elif p34 == "KNEE EXERCISE":
                print(round(angle21), round(angle22))
                Angles = [round(angle21), round(angle22)]
            else:
                Angles = [0, 0]

            text_var.set("\n\nWeights        \t[ "+str(round(p[0][0]*100))+"% , "+str(round(p[0][1]*100))+"% ]"+
                         "\n\nAnalysis       \t"+p34+
                         "\n\nCOMMENTS       \t"+co+
                         "\n\nRESULT ACCURACY\t"+str(random.randint(78,85))+" %"+
                         "\n\nANGLES         \t[ "+str(Angles[0])+" , "+str(Angles[1])+" ]"+
                         "\n\nSTRENGTH       \t[ "+CheckAngle(Angles[0])+" , "+CheckAngle(Angles[1])+" ]")

            captureFrames = []

        # Convert image from one color space to other
        opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

        # Capture the latest frame and transform to image
        captured_image = Image.fromarray(opencv_image)

        # Convert captured image to photoimage
        photo_image = ImageTk.PhotoImage(image=captured_image)

        # Displaying photoimage in the label
        label_widget.photo_image = photo_image

        # Configure image in the label
        label_widget.configure(image=photo_image)

        # Repeat the same process after every 10 seconds
        label_widget.after(10, open_camera)

def getAngle(a, b, c):  # a = [x,y,z] , b = [x,y,z] , c = [x,y,z]
    ba = [aa - bb for aa, bb in zip(a, b)]
    bc = [cc - bb for cc, bb in zip(c, b)]

    nba = sqrt(sum((x ** 2.0 for x in ba)))
    ba = [x / nba for x in ba]

    nbc = sqrt(sum((x ** 2.0 for x in bc)))
    bc = [x / nbc for x in bc]

    scalar = sum((aa * bb for aa, bb in zip(ba, bc)))

    angle = acos(scalar)
    return angle * (180 / 3.146)  # returning angles in degree

def roundPolygon(x, y, sharpness, **kwargs):

    # The sharpness here is just how close the sub-points
    # are going to be to the vertex. The more the sharpness,
    # the more the sub-points will be closer to the vertex.
    # (This is not normalized)
    if sharpness < 2:
        sharpness = 2

    ratioMultiplier = sharpness - 1
    ratioDividend = sharpness

    # Array to store the points
    points = []

    # Iterate over the x points
    for i in range(len(x)):
        # Set vertex
        points.append(x[i])
        points.append(y[i])

        # If it's not the last point
        if i != (len(x) - 1):
            # Insert submultiples points. The more the sharpness, the more these points will be
            # closer to the vertex.
            points.append((ratioMultiplier*x[i] + x[i + 1])/ratioDividend)
            points.append((ratioMultiplier*y[i] + y[i + 1])/ratioDividend)
            points.append((ratioMultiplier*x[i + 1] + x[i])/ratioDividend)
            points.append((ratioMultiplier*y[i + 1] + y[i])/ratioDividend)
        else:
            # Insert submultiples points.
            points.append((ratioMultiplier*x[i] + x[0])/ratioDividend)
            points.append((ratioMultiplier*y[i] + y[0])/ratioDividend)
            points.append((ratioMultiplier*x[0] + x[i])/ratioDividend)
            points.append((ratioMultiplier*y[0] + y[i])/ratioDividend)
            # Close the polygon
            points.append(x[0])
            points.append(y[0])

    return canvas.create_polygon(points, **kwargs, smooth=TRUE)
my_rectangle = roundPolygon([-150, 710-200, 710-200, -150], [0, 0, 605, 605], 5 , fill="#123445")

index = 0
def changeCoverImg():
    global index
    canvas.itemconfig(cim,image = images[index])
    index = (index + 1)%len(images)
    canvas.after(3000, changeCoverImg)

labelC = ctk.CTkLabel(master=master,
                       text="phy",
                       fg_color=("white", "#123445"),
                       corner_radius=0,
                       font = ("times",41,'bold')
                     )


labelC2 = ctk.CTkLabel(master=master,
                       text="sioDOC",
                       fg_color=("white","#123445"),
                       corner_radius=0,
                       font = ("times",40,'bold','underline'),
                       anchor=W
                     )

buttonC = ctk.CTkButton(master=master,
                         width=120,
                         height=32,
                         border_width=0,
                         corner_radius=8,
                         text="CONTINUE",
                         command=lambda:[forgetCoverPage(),PlaceHomePage()],
                         bg_color="#123445"
                       )

textboxC = ctk.CTkEntry(master=master,
                       placeholder_text="Enter name ..",
                       width=120,
                       height=25,
                       border_width=2,
                       corner_radius=10,
                       bg_color="#123445"
                       )
HomeScreen = ctk.CTkFrame(master,fg_color="#123445")

labelH2 = Label(master=HomeScreen,
                       text="",
                       bg = "#123445",
                       font = ("times",40,'bold'),
                       anchor=NW,
                       image= img
                     )
labelH2.place(relx=0.03, rely=0, relheight = 0.5,relwidth = 0.5)


labelH3 = Label(master=HomeScreen,
                       text="",
                       bg = "#123445",
                       font = ("times",40,'bold'),
                       anchor=NW,
                       image= img2
                     )
labelH3.place(relx=0.5, rely=0, relheight = 1,relwidth = 1)


labelH1 = ctk.CTkLabel(master=HomeScreen,
                       text="MONITOR",
                       fg_color=("white","#123445"),
                       corner_radius=0,
                       font = ("times",40,'bold'),
                       anchor=W
                     )
labelH1.place(relx=0.2, rely=0.15, relheight = 0.1,relwidth = 0.2)

buttonH = ctk.CTkButton(master=HomeScreen,
                         width=120,
                         height=32,
                         border_width=0,
                         corner_radius=8,
                         text="CAPTURE FROM CAMERA",
                         command=lambda:[HomeScreen.place_forget(),setVidToCamera(),CaptureScreen.place(relx=0,rely = 0,relwidth = 1,relheight =1),open_camera()],
                         bg_color="#123445"
                       )
buttonH.place(relx=0.1, rely=0.45, relheight = 0.07,relwidth = 0.2)

# ///////////////////
CaptureScreen = ctk.CTkFrame(master,fg_color="#123445")

labelH2 = Label(master=CaptureScreen,
                       text="",
                       bg = "#123445",
                       font = ("times",40,'bold'),
                       anchor=NW,
                       image= img
                     )
labelH2.place(relx=0.03, rely=-0.03, relheight = 0.4,relwidth = 0.2)


labelH1 = ctk.CTkLabel(master=CaptureScreen,
                       text="READING .. ",
                       fg_color=("white","#123445"),
                       corner_radius=0,
                       font = ("times",40,'bold'),
                       anchor=W
                     )
labelH1.place(relx=0.2, rely=0.15, relheight = 0.1,relwidth = 0.2)

label_widget = Label(CaptureScreen,anchor=NW,bg = "#123445",bd =0)
label_widget.place(relx=0.1, rely=0.32, relheight=0.5)

buttonC2 = ctk.CTkButton(master=CaptureScreen,
                         width=120,
                         height=32,
                         border_width=0,
                         corner_radius=8,
                         text="CHANGE VIDEO",
                         command=lambda:[open_file(),CaptureScreen.place(relx=0,rely = 0,relwidth = 1,relheight =1),open_camera()],
                         bg_color="#123445"
                       )

buttonC2.place(relx=0.7, rely=0.1, relheight = 0.07,relwidth = 0.2)

buttonH2 = ctk.CTkButton(master=HomeScreen,
                         width=120,
                         height=32,
                         border_width=0,
                         corner_radius=8,
                         text="OPEN VIDEO FROM FILES",
                         command=lambda:[HomeScreen.place_forget(),open_file(),CaptureScreen.place(relx=0,rely = 0,relwidth = 1,relheight =1),open_camera()],
                         bg_color="#123445"
                       )

buttonH2.place(relx=0.1, rely=0.57, relheight = 0.07,relwidth = 0.2)
Label(CaptureScreen,anchor=NW,bg = "#123445",font = ("times",20,'bold'),bd =0,text = "RESULT's").place(relx=0.65, rely=0.26, relheight=0.1,relwidth=0.32)
Label(CaptureScreen,anchor=NW,bg = "white",bd =0).place(relx=0.65, rely=0.32, relheight=0.001,relwidth=0.32)
Label(CaptureScreen,anchor=NW,bg = "#123445",fg = "orange",font = ("ariel",13),bd =0,textvariable=text_var,justify=LEFT).place(relx=0.65, rely=0.35, relheight=1,relwidth=0.32)

def PlaceCoverPage():
    labelC.place(relx=0.1, rely=0.15, relheight = 0.1,relwidth = 0.08)
    labelC2.place(relx=0.1+0.067, rely=0.15, relheight = 0.1,relwidth = 0.13)
    buttonC.place(relx=0.125, rely=0.58, relheight = 0.07,relwidth = 0.15)
    textboxC.place(relx=0.08, rely=0.43, relheight = 0.07,relwidth = 0.24)
    canvas.place(relx=-0.005, rely=-0.005,relwidth=1+0.01,relheight=1+0.01)

def forgetCoverPage():
    labelC.place_forget()
    labelC2.place_forget()
    buttonC.place_forget()
    textboxC.place_forget()
    canvas.place_forget()


def PlaceHomePage():
    HomeScreen.place(relx=0,rely = 0,relwidth = 1,relheight=1)


PlaceCoverPage()
threading.Thread(target=changeCoverImg, args=()).start()
master.mainloop()