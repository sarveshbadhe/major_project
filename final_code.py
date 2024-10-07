
import cv2
import numpy as np
import pyttsx3
import tensorflow as tf

# Initialize the TFLite interpreter with your own TFLite model file
interpreter = tf.lite.Interpreter(model_path="C:/Users/lenovo/Downloads/detect.tflite")
interpreter.allocate_tensors()

# Get the input and output tensors of the model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Initialize the video capture object
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Resize the frame to the input size expected by the model (e.g., 224x224)
    resized_frame = cv2.resize(frame, (320, 320))

    # Preprocess the frame by converting it to a NumPy array and scaling its values
    preprocessed_frame = np.expand_dims(resized_frame, axis=0).astype(np.float32) / 127.5 - 1.0

    # Set the input tensor of the model to the preprocessed frame
    interpreter.set_tensor(input_details[0]['index'], preprocessed_frame)

    # Run inference on the input tensor
    interpreter.invoke()

    # Get the output tensor of the model
    output_tensor = interpreter.get_tensor(output_details[0]['index'])

    # Get the index of the predicted class with the highest probability
    predicted_class_index = np.argmax(output_tensor)

    # Get the name of the predicted class
    class_names = ["10", "20", "50", "100", "200", "500", "2000"]
    predicted_class_name = class_names[predicted_class_index]


    #predicted_class_name = "your_class_names_list[predicted_class_index]"

    # Speak the name of the predicted class using the text-to-speech engine
    engine.say(predicted_class_name)
    engine.runAndWait()

    # Display the frame with the predicted class name overlaid on it
    cv2.putText(frame, predicted_class_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('frame', frame)

    # Break the loop if the user presses the ESC key
    if cv2.waitKey(1) == 27:
        break

# Release the video capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()