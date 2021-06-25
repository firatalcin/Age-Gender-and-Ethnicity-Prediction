from PyQt5.QtWidgets import QLabel, QVBoxLayout, QApplication,  QFileDialog, QPushButton, QWidget
import sys
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
import cv2 as cv
from deepface import DeepFace
import time



class Window(QWidget):
    def __init__(self):
        super().__init__()

        self.title = "PhotoAnalitic"
        self.left = 800
        self.top = 100
        self.width = 450
        self.height = 500
        self.iconName = "camera.png"
        self.main_window()


    def main_window(self):
        self.setWindowTitle(self.title)
        self.setWindowIcon(QtGui.QIcon(self.iconName))
        self.setGeometry(self.left, self.top, self.width, self.height)

        vbox = QVBoxLayout()

        self.label = QLabel(self)
        self.label.setStyleSheet("border :3px solid red;")
        vbox.addWidget(self.label)
        self.setLayout(vbox)

        self.button2 = QPushButton("Yaş Analizi", self)
        self.button2.move(120, 200)
        self.button2.clicked.connect(self.yasAnalizi)

        vbox.addWidget(self.button2)


        self.button3 = QPushButton("Cinsiyet Analizi", self)
        self.button3.move(120, 250)
        self.button3.clicked.connect(self.cinsiyetAnalizi)

        vbox.addWidget(self.button3)

        self.button4 = QPushButton("Irk Analizi", self)
        self.button4.move(120, 300)
        self.button4.clicked.connect(self.irkAnalizi)

        vbox.addWidget(self.button4)

        self.button5 = QPushButton("Gerçek Zamanlı Analiz (Yaş, Cinsiyet, Irk)", self)
        self.button5.move(120, 350)
        self.button5.clicked.connect(self.gercekZamanliAnaliz)

        vbox.addWidget(self.button5)

        self.show()



    def yasAnalizi(self):
        print("yas analizi")
        # bir dosyadan resim almak
        fname = QFileDialog.getOpenFileName(self, 'Open file', 'c:/Users/firatalcin/Desktop/bitirme_projesi/image',
                                          'Image files (*.jpg *.gif *.png *.jpeg)')
        imagePath = fname[0]


        #Yüz Tespiti
        def getFaceBox(net, frame, conf_threshold=0.7):
            frameOpencvDnn = frame.copy()
            # Resmin genişliğini ve uzunluğunu tespit etmek
            frameHeight = frameOpencvDnn.shape[0]
            frameWidth = frameOpencvDnn.shape[1]
            # Veri Ön işleme (kenarlar)
            blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

            # Algoritmanın çalıştırılması
            net.setInput(blob)
            detections = net.forward()
            bboxes = []

            # Tespit edilmiş yüzün etrafına bir dikdörtgen eklenmesi
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > conf_threshold:
                    x1 = int(detections[0, 0, i, 3] * frameWidth)
                    y1 = int(detections[0, 0, i, 4] * frameHeight)
                    x2 = int(detections[0, 0, i, 5] * frameWidth)
                    y2 = int(detections[0, 0, i, 6] * frameHeight)
                    bboxes.append([x1, y1, x2, y2])
                    cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
            return frameOpencvDnn, bboxes

        faceProto = "opencv_face_detector.pbtxt"
        faceModel = "opencv_face_detector_uint8.pb"

        ageProto = "age_deploy.prototxt"
        ageModel = "age_net.caffemodel"

        MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        ageList = ['(0-2)', '(3-6)', '(7-12)', '(13-20)', '(22-32)', '(34-43)', '(44-53)', '(60+)']

        # Eğtilmiş veri setlerinin okunması
        ageNet = cv.dnn.readNet(ageModel, ageProto)
        faceNet = cv.dnn.readNet(faceModel, faceProto)

        padding = 20


        #Yaş Tespiti
        def age_detector(frame):
            # Read frame
            t = time.time()
            frameFace, bboxes = getFaceBox(faceNet, frame)
            for bbox in bboxes:
                # Fotoğraftan analizin gerçekleştirilmesi için yüzün çıkarılması
                face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                       max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]

                blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                # En yüksek ayırt edici özelliğe göre sınıflandırma
                age = ageList[agePreds[0].argmax()]


                label = "{}".format(age)
                cv.putText(frameFace, label, (bbox[0], bbox[1] - 35), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2,
                           cv.LINE_AA)

                label1 = "Guven Orani : {:.3f}".format(agePreds[0].max())
                cv.putText(frameFace, label1, (bbox[0] , bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2,
                           cv.LINE_AA)



            return frameFace

        pixmap = QPixmap(imagePath)
        self.label.setPixmap(QPixmap(pixmap))
        self.resize(pixmap.width(), pixmap.height())

        input = cv.imread(imagePath)
        output = age_detector(input)
        cv.imshow('Yas Analizi', output)
        cv.waitKey(0)






    def cinsiyetAnalizi(self):
        print("Cinsiyet analizi")

        fname = QFileDialog.getOpenFileName(self, 'Open file', 'c:/Users/firatalcin/Desktop/bitirme_projesi/image',
                                            'Image files (*.jpg *.gif *.png *.jpeg)')
        imagePath = fname[0]


        #Yüz Tespiti

        def getFaceBox(net, frame, conf_threshold=0.7):
            frameOpencvDnn = frame.copy()
            frameHeight = frameOpencvDnn.shape[0]
            frameWidth = frameOpencvDnn.shape[1]
            blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

            net.setInput(blob)
            detections = net.forward()
            bboxes = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > conf_threshold:
                    x1 = int(detections[0, 0, i, 3] * frameWidth)
                    y1 = int(detections[0, 0, i, 4] * frameHeight)
                    x2 = int(detections[0, 0, i, 5] * frameWidth)
                    y2 = int(detections[0, 0, i, 6] * frameHeight)
                    bboxes.append([x1, y1, x2, y2])
                    cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
            return frameOpencvDnn, bboxes

        faceProto = "opencv_face_detector.pbtxt"
        faceModel = "opencv_face_detector_uint8.pb"

        genderProto = "gender_deploy.prototxt"
        genderModel = "gender_net.caffemodel"

        MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        genderList = ['ERKEK', 'KADIN']


        genderNet = cv.dnn.readNet(genderModel, genderProto)
        faceNet = cv.dnn.readNet(faceModel, faceProto)

        padding = 20

        #Cinsiyet Tespiti

        def gender_detector(frame):
            # Read frame
            t = time.time()
            frameFace, bboxes = getFaceBox(faceNet, frame)
            for bbox in bboxes:
                # print(bbox)
                face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                       max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]

                blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                genderPreds = genderNet.forward()
                gender = genderList[genderPreds[0].argmax()]

                label = "{}".format(gender)
                cv.putText(frameFace, label, (bbox[0], bbox[1] - 35), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                           cv.LINE_AA)

                label1 = "Guven Orani : {:.3f}".format(genderPreds[0].max())
                cv.putText(frameFace, label1, (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                           cv.LINE_AA)



            return frameFace


        pixmap = QPixmap(imagePath)
        self.label.setPixmap(QPixmap(pixmap))
        self.resize(pixmap.width(), pixmap.height())


        input = cv.imread(imagePath)
        output = gender_detector(input)
        cv.imshow('Cinsiyet Analizi', output)
        cv.waitKey(0)






    def irkAnalizi(self):


        fname = QFileDialog.getOpenFileName(self, 'Open file', 'c:/Users/firatalcin/Desktop/bitirme_projesi/image',
                                            'Image files (*.jpg *.gif *.png *.jpeg)')
        imagePath = fname[0]

        # Yüz Tespiti

        def getFaceBox(net, frame, conf_threshold=0.7):
            frameOpencvDnn = frame.copy()
            frameHeight = frameOpencvDnn.shape[0]
            frameWidth = frameOpencvDnn.shape[1]
            blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

            net.setInput(blob)
            detections = net.forward()
            bboxes = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > conf_threshold:
                    x1 = int(detections[0, 0, i, 3] * frameWidth)
                    y1 = int(detections[0, 0, i, 4] * frameHeight)
                    x2 = int(detections[0, 0, i, 5] * frameWidth)
                    y2 = int(detections[0, 0, i, 6] * frameHeight)
                    bboxes.append([x1, y1, x2, y2])
                    cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
            return frameOpencvDnn, bboxes

        faceProto = "opencv_face_detector.pbtxt"
        faceModel = "opencv_face_detector_uint8.pb"

        faceNet = cv.dnn.readNet(faceModel, faceProto)

        padding = 20


        def race_detector(frame):
            # Read frame
            t = time.time()
            frameFace, bboxes = getFaceBox(faceNet, frame)
            for bbox in bboxes:
                # Yüz ile çerçeve arasındaki boşluk verilir(kenarlarla uyumluluk)
                face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                       max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]



                result = DeepFace.analyze(imagePath, actions=['race'])


                label = "{}".format(result["dominant_race"].upper())
                cv.putText(frameFace, label, (bbox[0], bbox[1] - 35), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2,
                           cv.LINE_AA)

                #class = {"asian", "indian", "black", "white", "middle eastern", "latino hispanic"}

                label1 = "Gecen Zaman : {:.3f}".format(time.time() - t)
                cv.putText(frameFace, label1, (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2,
                           cv.LINE_AA)



            return frameFace

        pixmap = QPixmap(imagePath)
        self.label.setPixmap(QPixmap(pixmap))
        self.resize(pixmap.width(), pixmap.height())

        input = cv.imread(imagePath)
        output = race_detector(input)
        cv.imshow('Irk Analizi', output)
        cv.waitKey(0)

    def gercekZamanliAnaliz(self):

        def getFaceBox(net, frame, conf_threshold=0.7):
            frameOpencvDnn = frame.copy()
            frameHeight = frameOpencvDnn.shape[0]
            frameWidth = frameOpencvDnn.shape[1]
            blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

            net.setInput(blob)
            detections = net.forward()
            faceBoxes = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > conf_threshold:
                    x1 = int(detections[0, 0, i, 3] * frameWidth)
                    y1 = int(detections[0, 0, i, 4] * frameHeight)
                    x2 = int(detections[0, 0, i, 5] * frameWidth)
                    y2 = int(detections[0, 0, i, 6] * frameHeight)
                    faceBoxes.append([x1, y1, x2, y2])
                    cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
            return frameOpencvDnn, faceBoxes

        faceProto = "opencv_face_detector.pbtxt"
        faceModel = "opencv_face_detector_uint8.pb"
        ageProto = "age_deploy.prototxt"
        ageModel = "age_net.caffemodel"
        genderProto = "gender_deploy.prototxt"
        genderModel = "gender_net.caffemodel"

        MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        ageList = ['(0-2)', '(3-6)', '(7-12)', '(13-20)', '(22-32)', '(34-43)', '(44-53)', '(54-100)']
        genderList = ['Erkek', 'Kadın']

        faceNet = cv.dnn.readNet(faceModel, faceProto)
        ageNet = cv.dnn.readNet(ageModel, ageProto)
        genderNet = cv.dnn.readNet(genderModel, genderProto)

        kamera = cv.VideoCapture(0)
        padding = 20

        img_counter = 0

        while True:
            ret, videoGoruntu = kamera.read()
            if not ret:
                print("görüntü alma başarısız")
                break
            cv.imshow("Video kamera", videoGoruntu)

            k = cv.waitKey(1)

            if k % 256 == 27:
                # ESC basın
                print("kapatılıyor...")
                break
            elif k % 256 == 32:
                # SPACE basın

                resultImg, faceBoxes = getFaceBox(faceNet, videoGoruntu)

                for faceBox in faceBoxes:
                    face = videoGoruntu[max(0, faceBox[1] - padding):
                                        min(faceBox[3] + padding, videoGoruntu.shape[0] - 1), max(0, faceBox[0] - padding):min(faceBox[2] + padding, videoGoruntu.shape[1] - 1)]

                    blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

                    genderNet.setInput(blob)
                    genderPreds = genderNet.forward()
                    gender = genderList[genderPreds[0].argmax()]
                    print('Cinsiyet: {0}'.format(gender))

                    ageNet.setInput(blob)
                    agePreds = ageNet.forward()
                    age = ageList[agePreds[0].argmax()]
                    print('Yaş: {0} '.format(age))

                    label1 = "{}".format(age)
                    cv.putText(resultImg, label1, (faceBox[0], faceBox[1] - 55), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                               (0, 255, 255), 2, cv.LINE_AA)

                    label2 = "{}".format(gender)
                    cv.putText(resultImg, label2, (faceBox[0], faceBox[1] - 35), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                               (255, 0, 0),2, cv.LINE_AA)

                    result = DeepFace.analyze(videoGoruntu, actions=['race'])
                    label3 = "{}".format(result["dominant_race"].upper())

                    cv.putText(resultImg, label3, (faceBox[0], faceBox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8,
                               (0, 255, 255), 2, cv.LINE_AA)

                    img_name = "image_{}.png".format(img_counter)
                    cv.imwrite("c:/Users/firatalcin/Desktop/bitirme_projesi/image_detected/ " + img_name, resultImg)
                    img_counter += 1

                cv.imshow("Yas, Cinsiyet ve Irk Tespit edildi.", resultImg)



App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())
