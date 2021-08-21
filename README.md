# Sign_Language_Detection
This project is used to process a video depicting a word in sign language and gives the translation in any preferred language.

asl_recognition.py:
Train the model to recognise sign language alphabets using Convolutional Neural Network(CNN).

sign_language_detection.py: 
Takes in the input video, Extracts the unique frames using Scale Invariant Feature Transform(SIFT) and Support Vector Machine(SVM).
Performs Image preprocessing using Skin Segmentation and predict the alphabet using the trained model.
The predicted alphabets are combined together and translated to the preferred language using Google Translate API.

svm_asl_train_.py:
Train the model to classify unique frames.

svm_asl_test.py:
Test file to classify unique frames.

metrics.py:
To check the accuracy of the svm model.




