import cv2
import os
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(20)

class Detector:
    def __init__(self):
        pass

    def readclasses(self, classesFilePath):
       
        with open(classesFilePath, 'r') as f:
            self.classeslist = f.read().splitlines()
        self.colorlist = np.random.uniform(low=0, high=255, size=(len(self.classeslist), 3))

    def downloadModel(self, modelURL):
        
        fileName = os.path.basename(modelURL)
        self.modelName = fileName[:fileName.index('.')]
        self.cacheDir = "./pretrained_models"
        os.makedirs(self.cacheDir, exist_ok=True)
        get_file(fname=fileName, origin=modelURL, cache_dir=self.cacheDir, cache_subdir="checkpoints", extract=True)

    def loadModel(self):
       
        print(f"Loading Model {self.modelName}")
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(self.cacheDir, "checkpoints", self.modelName, "saved_model"))
        print(f"Model {self.modelName} loaded successfully...")

    def createBoundingBox(self, image, threshold=0.5):
        
        inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
        inputTensor = inputTensor[tf.newaxis, ...]

        detections = self.model(inputTensor)
        bboxs = detections['detection_boxes'][0].numpy()
        classIndexes = detections['detection_classes'][0].numpy().astype(np.int32)
        classScores = detections['detection_scores'][0].numpy()

        imH, imW, imC = image.shape
        bboxIdx = tf.image.non_max_suppression(
            bboxs, classScores, max_output_size=100, iou_threshold=threshold, score_threshold=threshold)

        if len(bboxIdx) != 0:
            for i in bboxIdx:
                bbox = tuple(bboxs[i].tolist())
                classConfidence = round(100 * classScores[i])
                classIndex = classIndexes[i]
                classLabelText = self.classeslist[classIndex]
                classColor = self.colorlist[classIndex]

                displayText = f'{classLabelText}: {classConfidence}%'
                ymin, xmin, ymax, xmax = bbox
                xmin, xmax, ymin, ymax = (xmin * imW, xmax * imW, ymin * imH, ymax * imH)
                xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=classColor, thickness=2)
                cv2.putText(image, displayText, (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

        return image

    def predictRealTime(self, threshold=0.5):
       
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error accessing webcam")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            bboxFrame = self.createBoundingBox(frame, threshold)
            cv2.imshow("Real-Time Object Detection", bboxFrame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
classFile = "coco.names"
threshold = 0.5


detector = Detector()
detector.readclasses(classFile)
detector.downloadModel(modelURL)
detector.loadModel()
detector.predictRealTime(threshold)
