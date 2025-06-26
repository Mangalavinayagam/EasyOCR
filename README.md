# EasyOCR

#Modules:
pip install numpy Pillow tflite-runtime


#Execute:
python3 test_tflite_all.py   --detector_model EasyOCR_EasyOCRDetector.tflite   --recognizer_model EasyOCR_EasyOCRRecognizer.tflite   --image test.jpg   --labels labels.txt
