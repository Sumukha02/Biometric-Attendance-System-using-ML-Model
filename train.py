import cv2
import os
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
from tkinter import messagebox
import ctypes
import smtplib
import dlib
import tkinter as tk
from imutils import face_utils

# Initialize the main window
window = tk.Tk()
window.title("Face Recognition Biometric Attendance System")

# Get screen width and height
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

# Set window geometry and allow resizing
window.geometry('1920x1080')
window.resizable(width=True, height=True)
# window.configure(background='white')

# Disable DPI scaling (Windows)
if ctypes.windll.shcore.SetProcessDpiAwareness(1):
    ctypes.windll.user32.SetProcessDPIAware()

# Load the background image
background_image = Image.open("GUI_image.jpg")
background_photo = ImageTk.PhotoImage(background_image)

# Create a Label widget with the background image
background_label = tk.Label(window, image=background_photo)
background_label.pack(fill="both", expand=True)

# Header label
message = tk.Label(window, text="Face Recognition Biometric Attendance System", bg="#007ACC", fg="black", width=80,
                   height=1, font=('times', 30, 'bold'))
message.place(x=5, y=20)

# fullscreen
# window.attributes('-fullscreen', True)

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

# INPUT FRAME 1
# Create a Frame for the input fields
input_frame_1 = tk.Frame(window, width=800, height=250)
input_frame_1.place(x=35, y=100)

# Load the background image for the input frame
background_image_input_frame_1 = Image.open("GUI_image_2.jpg")  # Replace with your image path
background_photo_input_frame_1 = ImageTk.PhotoImage(background_image_input_frame_1)

# Resize the background image to match the size of the input frame
background_image_input_frame_1 = background_image_input_frame_1.resize((800, 370), Image.ANTIALIAS)
background_photo_input_frame_1 = ImageTk.PhotoImage(background_image_input_frame_1)

message = tk.Label(input_frame_1, text="Enrollment Process", bg="#007ACC", fg="black", width=25,
                   height=1, font=('times', 30, 'bold'))
message.pack(fill="both", expand=True)

# Create a Label widget with the background image for input frame
background_label_input_frame = tk.Label(input_frame_1, image=background_photo_input_frame_1)
background_label_input_frame.pack(fill="both", expand=True)

# INPUT FRAME 2
# Create a Frame for the input fields
input_frame_2 = tk.Frame(window, width=800, height=500)
input_frame_2.place(x=35, y=550)

# Load the background image for the input frame
background_image_input_frame_2 = Image.open("GUI_image_2.jpg")  # Replace with your image path
background_photo_input_frame_2 = ImageTk.PhotoImage(background_image_input_frame_2)

# Resize the background image to match the size of the input frame
background_image_input_frame_2 = background_image_input_frame_2.resize((800, 250), Image.ANTIALIAS)
background_photo_input_frame_2 = ImageTk.PhotoImage(background_image_input_frame_2)

message = tk.Label(input_frame_2, text="Training", bg="#007ACC", fg="black", width=25,
                   height=1, font=('times', 30, 'bold'))
message.pack(fill="both", expand=True)

# Create a Label widget with the background image for input frame
background_label_input_frame_2 = tk.Label(input_frame_2, image=background_photo_input_frame_2)
background_label_input_frame_2.pack(fill="both", expand=True)

# INPUT FRAME 3
# Create a Frame for the input fields
input_frame_3 = tk.Frame(window, width=800, height=500)
input_frame_3.place(x=900, y=100)

# Load the background image for the input frame
background_image_input_frame_3 = Image.open("GUI_image_2.jpg")  # Replace with your image path
background_photo_input_frame_3 = ImageTk.PhotoImage(background_image_input_frame_3)

# Resize the background image to match the size of the input frame
background_image_input_frame_3 = background_image_input_frame_3.resize((800, 400), Image.ANTIALIAS)
background_photo_input_frame_3 = ImageTk.PhotoImage(background_image_input_frame_3)

message = tk.Label(input_frame_3, text="Attendance", bg="#007ACC", fg="black", width=25,
                   height=1, font=('times', 30, 'bold'))
message.pack(fill="both", expand=True)

# Create a Label widget with the background image for input frame
background_label_input_frame_3 = tk.Label(input_frame_3, image=background_photo_input_frame_3)
background_label_input_frame_3.pack(fill="both", expand=True)

lbl = tk.Label(input_frame_1, text="Enter Student Enrollment ID", width=23, height=2, fg="White", bg="#007ACC",
               font=('times', 15, ' bold '))
lbl.place(x=40, y=85)
txt = tk.Entry(input_frame_1, borderwidth=5, width=25, bg="yellow", fg="red", font=('times', 15, ' bold '))
txt.place(x=350, y=95)

lbl2 = tk.Label(input_frame_1, text="Enter Student Name", width=23, fg="White", bg="#007ACC", height=2,
                font=('times', 15, ' bold '))
lbl2.place(x=40, y=220)
txt2 = tk.Entry(input_frame_1, borderwidth=5, width=25, bg="yellow", fg="red", font=('times', 15, ' bold '))
txt2.place(x=350, y=230)

lbl3 = tk.Label(input_frame_2, text="Notification", width=20, fg="White", bg="#007ACC", height=2,
                font=('times', 15, 'bold'))
lbl3.place(x=40, y=100)
message = tk.Label(input_frame_2, borderwidth=5, text="", bg="yellow", fg="red", width=30, height=2,
                   font=('times', 15, ' bold '))
message.place(x=350, y=95)

lbl4 = tk.Label(input_frame_3, text="Attendance : ", width=20, fg="black", bg="light green", height=2,
                font=('times', 15, 'bold'))
lbl4.place(x=40, y=85)
message2 = tk.Label(input_frame_3, borderwidth=5, text="", fg="red", bg="yellow", width=30, height=2,
                    font=('times', 15, ' bold '))
message2.place(x=350, y=82)


def clear():
    txt.delete(0, 'end')
    res = ""
    message.configure(text=res)


def clear2():
    txt2.delete(0, 'end')
    res = ""
    message.configure(text=res)


def quit_app():
    dialog_title = 'Confirm Quit'
    dialog_text = 'Are you sure you want to quit?'
    answer = messagebox.askquestion(dialog_title, dialog_text)

    if answer == 'yes':
        window.destroy()


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False



# # Define some global variables
# global num_consecutive_blinks
# global previous_landmarks
#
#
# # Load the pre-trained face detector and shape predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
#
# # Define the eye aspect ratio threshold and the consecutive number of frames required for blinking
# EYE_AR_THRESHOLD = 0.2
# EYE_AR_CONSEC_FRAMES = 3
#
# # Define the movement threshold
# MOVEMENT_THRESHOLD = 5
#
# # Define the liveness detection threshold
# LIVENESS_THRESHOLD = 0.8
#
# def detect_landmarks(image, face_rect):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     shape = predictor(gray, face_rect)
#     shape = face_utils.shape_to_np(shape)
#     return shape
#
# def compute_eye_aspect_ratio(eye):
#     # Compute the Euclidean distances between the two sets of vertical eye landmarks
#     A = np.linalg.norm(eye[1] - eye[5])
#     B = np.linalg.norm(eye[2] - eye[4])
#
#     # Compute the Euclidean distance between the horizontal eye landmark
#     C = np.linalg.norm(eye[0] - eye[3])
#
#     # Compute the eye aspect ratio
#     ear = (A + B) / (2.0 * C)
#     return ear
#
# def is_blinking(eye):
#     ear = compute_eye_aspect_ratio(eye)
#     return ear >= EYE_AR_THRESHOLD
#
# def has_movement(landmarks):
#     # Check if there is any movement in the face by comparing the current landmarks with the previous landmarks
#     # You can define your own criteria for movement detection based on your requirements
#     # Here's an example of a simple movement detection based on the difference between the current and previous landmarks
#     global previous_landmarks
#     previous_landmarks = None
#     if previous_landmarks is not None:
#         movement_threshold = 5  # Adjust this threshold based on your needs
#         movement_detected = np.mean(np.abs(landmarks - previous_landmarks)) > movement_threshold
#     else:
#         movement_detected = False
#
#     previous_landmarks = landmarks.copy()
#     return movement_detected
#
# def compute_liveness_score(eye_ar, head_movement, lip_movement):
#     liveness_score = (eye_ar + head_movement + lip_movement) / 3
#     return liveness_score
#
# def liveness_detection(frame):
#     # Detect faces in the frame
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray)
#
#     # Iterate over detected faces
#     for face in faces:
#         landmarks = detect_landmarks(frame, face)
#
#         # Compute the eye aspect ratio
#         left_eye = landmarks[36:42]
#         right_eye = landmarks[42:48]
#         left_ear = compute_eye_aspect_ratio(left_eye)
#         right_ear = compute_eye_aspect_ratio(right_eye)
#
#         # Check if the person is blinking (indicating liveness)
#         # left_blink = is_blinking(left_eye)
#         # right_blink = is_blinking(right_eye)
#
#         # # Check if there is any movement in the face
#         # movement_detected = has_movement(landmarks)
#         #
#         # if left_blink or right_blink or movement_detected:
#         #     cv2.putText(frame, "Live", (face.left(), face.top() - 10),
#         #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#         # else:
#         #     cv2.putText(frame, "Spoof", (face.left(), face.top() - 10),
#         #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#
#         # Compute the head movement
#
#         head_movement = np.mean(np.abs(landmarks[0] - landmarks[16]))
#
#         # Compute the lip movement
#         lip_movement = np.mean(np.abs(landmarks[48] - landmarks[60]))
#
#         # Combine the liveness detection features
#         liveness_score = (left_ear + right_ear) / 2 + head_movement + lip_movement
#
#         # Determine if the face is live
#         if liveness_score > LIVENESS_THRESHOLD:
#             cv2.putText(frame, "Live", (face.left(), face.top() - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#         else:
#             cv2.putText(frame, "Spoof", (face.left(), face.top() - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#
#     return frame

def TakeImages():
    global Id
    Id = (txt.get())
    name = (txt2.get())
    if (is_number(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0

        while (True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # incrementing sample number
                sampleNum = sampleNum + 1
                # saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("TrainingImage\ " + name + "." + Id + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                # display the frame
                cv2.imshow('frame', img)


            # wait for 100 miliseconds

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum > 60:
                break
        cam.release()
        cv2.destroyAllWindows()
        res = "Images Saved for ID : " + Id + " Name : " + name
        row = [Id, name]
        with open('StudentDetails\StudentDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text=res)
    else:
        if (is_number(Id)):
            res = "Enter Alphabetical Name"
            message.configure(text=res)
        if (name.isalpha()):
            res = "Enter Numeric Id"
            message.configure(text=res)


def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()  # recognizer = cv2.face.LBPHFaceRecognizer_create()#$cv2.createLBPHFaceRecognizer()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res = "Image Trained"  # +",".join(str(f) for f in Id)
    message.configure(text=res)

def getImagesAndLabels(path):
    # get the path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # print(imagePaths)

    # create empth face list
    faces = []

    # create empty ID list
    Ids = []

    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')

        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')

        # getting the Id from the image
        Id = int(os.path.split(imagePath)[-1].split(".")[1])

        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids


def TrackImages():
    global df
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);
    df = pd.read_csv("StudentDetails\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # im = liveness_detection(im)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if (conf < 50):
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa = df.loc[df['Id'] == Id]['Name'].values
                tt = str(Id) + "-" + aa
                attendance.loc[len(attendance)] = [Id, aa, date, timeStamp]

            else:
                Id = 'Unknown'
                tt = str(Id)
            if (conf > 75):
                noOfFile = len(os.listdir("ImagesUnknown")) + 1
                cv2.imwrite("ImagesUnknown\Image" + str(noOfFile) + ".jpg", im[y:y + h, x:x + w])
            cv2.putText(im, str(tt), (x, y + h), font, 1, (255, 255, 255), 2)
        attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
        cv2.imshow('im', im)
        if (cv2.waitKey(1) == ord('q')):
            break

    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour, Minute, Second = timeStamp.split(":")
    fileName = "Attendance\Attendance_" + date + "_" + Hour + "-" + Minute + "-" + Second + ".csv"
    attendance.to_csv(fileName, index=False)
    cam.release()
    cv2.destroyAllWindows()
    # print(attendance)
    res = attendance
    message2.configure(text=res)

    # Call the send_email_on_button function with necessary information
    send_email_on_button(date, Hour, Minute, Second, attendance)

def send_email(subject, message, to_email):
    global server
    from_email = 'ssumukh02@gmail.com'
    password = 'ozoonlsoanjomtgp'

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, password)

        msg = f'Subject: {subject}\n\n{message}'
        server.sendmail(from_email, to_email, msg)
        print("Email sent successfully!")
    except Exception as e:
        print("Error sending email:", e)
    finally:
        server.quit()
def send_email_on_button(date, Hour, Minute, Second, attendance):
    # Send email with attendance details
    subject = "Attendance Report"
    message = f"Attendance report for {date} at {Hour}:{Minute}:{Second}\n\n{attendance}"
    to_email = 'sumukha.s2020@vitstudent.ac.in'  # Replace with the actual recipient's email address
    send_email(subject, message, to_email)


clearButton = tk.Button(input_frame_1, text="Clear", command=clear, fg="red", bg="pink", width=10, height=1,
                        activebackground="Red", font=('times', 15, ' bold '))
clearButton.place(x=650, y=95)

clearButton2 = tk.Button(input_frame_1, text="Clear", command=clear2, fg="red", bg="pink", width=10, height=1,
                         activebackground="Red", font=('times', 15, ' bold '))
clearButton2.place(x=650, y=230)

takeImg = tk.Button(input_frame_1, text="Student Enrollment", command=TakeImages, fg="black", bg="white smoke",
                    width=20, height=3,
                    activebackground="Red", font=('times', 15, ' bold '))
takeImg.place(x=350, y=320)

trainImg = tk.Button(input_frame_2, text="Train Images", command=TrainImages, fg="black", bg="white smoke", width=20,
                     height=3,
                     activebackground="Red", font=('times', 15, ' bold '))
trainImg.place(x=350, y=190)

trackImg = tk.Button(input_frame_3, text="Track Enrollment", command=TrackImages, fg="black", bg="white smoke",
                     width=20, height=3,
                     activebackground="Red", font=('times', 15, ' bold '))
trackImg.place(x=40, y=190)

quitButton = tk.Button(input_frame_3, text="Quit", command=quit_app, fg="black", bg="white", width=20, height=3,
                       font=('times', 15, 'bold'))
quitButton.place(x=40, y=330)

# Copy write Label for the GUI
copyWrite = tk.Text(window, bg="#007ACC", borderwidth=5, width=25, height=1.3,
                    font=('times', 16, 'italic bold underline'))
copyWrite.tag_configure("superscript", offset=10)
copyWrite.insert("insert", "Copyright © 2023 Team alpha")
copyWrite.configure(state="disabled", fg="Brown")
copyWrite.pack(side="left")
copyWrite.place(x=850, y=900)

window.mainloop()