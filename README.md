🌿 Plant Disease Prediction using CNN & Deep Learning

A deep learning-based project to automatically detect and classify plant diseases from leaf images using Convolutional Neural Networks (CNN).
Achieved 85.6% validation accuracy, making it a reliable tool for early plant disease detection.
Built with TensorFlow, Keras, and Streamlit for training and deployment.

🧩 Project Overview
This is my deep learning project where I built a Convolutional Neural Network (CNN) model to detect and classify plant diseases from leaf images. The main goal of this project is to help identify plant diseases early, allowing farmers to take preventive actions and reduce crop loss.
The model analyzes an image of a plant leaf and predicts whether it is healthy or affected by a specific disease.

🎯 Objective
The objective of this project is to create an AI model that can automatically recognize patterns and visual features in leaves—such as color, texture, and shape—that are often linked to diseases. Instead of manually extracting features, I used a CNN model that learns them directly from images during training.

🧠 Model Training & Evaluation
I trained the CNN model on a dataset containing images of both healthy and diseased plant leaves. The model was trained using TensorFlow and Keras libraries in a Jupyter Notebook environment.
After training, I evaluated the model on the validation dataset, and it achieved a validation accuracy of 85.6%. This means that the model can correctly classify most plant leaf images.
To improve generalization and avoid overfitting, I applied image augmentation and dropout layers during training. The CNN architecture automatically learned important visual features, like spots or color changes, which are strong indicators of disease.

While the results are promising, there’s room for improvement. In the future, I plan to:-
Use deeper models such as ResNet50 or EfficientNet
Train on a larger, more diverse dataset
Tune hyperparameters for better accuracy

⚙️ Technologies Used
Python
TensorFlow / Keras – model building and training
NumPy, Pandas – data processing
Matplotlib – visualization
Streamlit – for the web interface (optional)
Jupyter Notebook – model development and experiments

📈 Results
The trained CNN model achieved:
Validation Accuracy: 85.6%
Loss: 0.6273

