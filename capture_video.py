import cv2
from cv2 import imshow

if __name__ == '__main__':

    #Access to IP camera video stream (Axis models)
    #----------------------------------------------
    #TODO: acquisition and processing in independent threads
    # Conect to video Source
    cam = cv2.VideoCapture()
    cam.open("http://pdt:cuenca@138.4.32.13/mjpg/video.mjpg")  # user:pass is necessary to work
    if cam.isOpened():
        print("Camera connection established.")
    else:
        print("Failed to connect to the camera.")
        exit(-1)

    while True:
        ret, frame = cam.read()
        if not ret:
            continue
        imshow("Source video", frame)
        cv2.waitKey(10) #solo es necesario si hay otros procesos corriendo, de normal se puede quitar


#Hay que reajustar el tamaño de la imagen, la adquirida por la camara es enorme. Tb se puede quitar la parte de la imagen que
#no tiene información relevante. Para guardar la relacion de aspecto tn se podria quitar de arriba ya que solo hay pared