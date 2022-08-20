import cv2
import numpy as np
import tensorflow as tf
import win32com.client as wincl
import pyttsx3

engine = pyttsx3.init()
# speak = wincl.Dispatch("SAPI.SpVoice")

model = tf.keras.models.load_model('asl_model_2_layer.h5')

cap = cv2.VideoCapture(0)

predicted_text=''
while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    
    if ret:
        x1, y1, x2, y2 = 100, 100, 500, 500
        #  = frame[y1:y2, x1:x2]new_img
        
        if cv2.waitKey(1) & 0xFF == ord('s'):

            img_cropped = img[y1:y2, x1:x2]
            img_cropped = cv2.resize(img_cropped, (400, 400))
        
            img_array = tf.keras.utils.img_to_array(img_cropped)
            img_array = tf.expand_dims(img_array, 0) # Create a batch

            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            class_names = ['A',
                            'B',
                            'C',
                            'D',
                            'E',
                            'F',
                            'G',
                            'H',
                            'I',
                            'J',
                            'K',
                            'L',
                            'M',
                            'N',
                            'Nothing',
                            'O',
                            'P',
                            'Q',
                            'R',
                            'S',
                            'Space',
                            'T',
                            'U',
                            'V',
                            'W',
                            'X',
                            'Y',
                            'Z']
                

            print(
                "This image most likely belongs to {} with a {:.2f} percent confidence."
                .format(class_names[np.argmax(score)], 100*np.max(score))
            )
            predicted_text = class_names[np.argmax(score)]
            # speak.Speak([predicted_text])
            engine.say(f"The predicted letter is {predicted_text}")
            engine.runAndWait()
        
        cv2.putText(img, predicted_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
        
        cv2.imshow('predicted image',img)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()