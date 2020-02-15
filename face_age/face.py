from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

def worker():
    # initialize the list of class labels
    CLASSES = [str(i) for i in range(1,101)]

    # detection frontalface with opencv-python 
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # load our serialized model from disk
    net = cv2.dnn.readNetFromCaffe("age.prototxt", "dex_chalearn_iccv2015.caffemodel")

    # check if we are going to use GPU
    if args["use_gpu"]:
        # set CUDA as the preferable backend and target
        print("[INFO] setting preferable backend and target to CUDA...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # initialize the video stream and pointer to output video file, then start the FPS timer  
    print("[INFO] accessing video stream...")
    vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
    writer = None
    fps = FPS().start() 
    time.sleep(0.4)

    # loop over the frames from the video stream
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()

        # if the frame was not grabbed, then we have reached the end of the stream
        if not grabbed:
            break
        # resize the frame, grab the frame dimensions and detect faces
        frame = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        print("Found "+str(len(faces))+" face(s)")  
        
        # loop over the detections
        for (x,y,w,h) in faces:
            # convert image_faces to a blob
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0), 2)
            face_img = frame[y:y+h, x:x+w].copy()
            blob = cv2.dnn.blobFromImage(face_img, 1, (224, 224))
            
            # pass the blob through the network and obtain the detections and predictions
            net.setInput(blob)
            preds = net.forward()
            
            # draw the prediction on the frame
            age = CLASSES[preds[0].argmax()]
            overlay_text = "%s years" % (age)
            y = y - 15 if y - 15 > 15 else y + 15
            cv2.putText(frame, overlay_text ,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,0), 2)

        # show the output frame       
        cv2.imshow("Image", frame) 
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # if an output video file path has been supplied and the video
        # writer has not been initialized, do so now
        if args["output"] != "" and writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 30,
                (frame.shape[1], frame.shape[0]), True)
        
        # if the video writer is not None, write the frame to the output
        # video file
        if writer is not None:
            writer.write(frame)

        # update the FPS counter
        fps.update()
        
    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))    

if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str, default="",
        help="path to (optional) input video file")
    ap.add_argument("-o", "--output", type=str, default="",
        help="path to (optional) output video file")
    ap.add_argument("-u", "--use-gpu", type=bool, default=False,
        help="boolean indicating if CUDA GPU should be used")
    args = vars(ap.parse_args())
    worker()