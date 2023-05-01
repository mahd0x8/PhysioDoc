# PhysioDoc
A Convolutional LSTM-Based System for Automated Exercise Monitoring

### Real-time exercise recognition with Convolutional LSTM
This project is a real-time exercise recognition system that uses Convolutional LSTM neural networks to classify exercises in real-time. The system can recognize two types of exercises: finger exercises and knee exercises. The project was developed using Python, TensorFlow, OpenCV, and other libraries.

### Getting Started
#### Prerequisites
Before running the project, you will need to install the following libraries:

tensorflow
opencv-python
pafy
numpy
matplotlib
scikit-learn

#### Installing
You can install these libraries using pip. Here is an example:

Copy code
pip install tensorflow opencv-python pafy numpy matplotlib scikit-learn

#### Usage
To use the real-time exercise recognition system, you can run the exercise_recognition.py file using Python. You can specify the input source (e.g., webcam, video file) and other parameters in the file.

Here is an example command to run the program with webcam input:

Copy code
python exercise_recognition.py --input_type webcam

#### Training the Model
The model used in this project was trained using a dataset of finger and knee exercise videos. You can train your own model using the train_model.py file.

#### Dataset
The dataset used in this project is not included in the repository. However, you can create your own dataset by recording videos of finger and knee exercises. Each video should contain one exercise and be labeled with the corresponding exercise type (finger or knee).

#### Preprocessing
Before training the model, you will need to preprocess the videos into sequences of frames. You can use the preprocess_videos.py file to do this. The preprocessed data will be saved in a directory named data.

#### Training
To train the model, run the train_model.py file. You can specify the dataset directory and other parameters in the file. The trained model will be saved in a file named exercise_recognition_model.h5.

#### License
This project is licensed under the APACHE 2.0 License - see the LICENSE.md file for details.

#### Screenshpts


#### Acknowledgments
This project was inspired by the work of Shanmukh Srinivas and Matthew Tancik.
