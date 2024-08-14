# Level-1 Self-Driving Car Project

This project demonstrates the development of a Level-1 self-driving car that can autonomously control the steering wheel and maintain a constant speed within a simulated environment. The Udacity Self-Driving Car Simulator is used for simulation, and a Convolutional Neural Network (CNN) model built with TensorFlow and Keras is trained to make driving decisions based on camera input.

## Project Overview

The goal of this project is to train a neural network model to autonomously drive a car in a simulated environment. The model will take in camera images as input and output steering angles, allowing the car to navigate the track without human intervention. A socket connection is established between the simulator and the TensorFlow script to control the car in real-time.

### Key Features
- **Autonomous Steering Control**: The car's steering wheel is controlled based on predictions from the CNN model.
- **Constant Speed Maintenance**: The car runs at a constant speed throughout the simulation.
- **Real-Time Simulation**: The Udacity Self-Driving Car Simulator is used for real-time simulation, with a socket connection to send commands from the machine learning model to the simulator.

## Getting Started

### Prerequisites

To run this project, you will need the following:

- Python 3.x
- TensorFlow
- Keras
- NumPy
- OpenCV (for image preprocessing)
- Flask (for managing the socket connection)
- Udacity Self-Driving Car Simulator

### Installation

1. Install the required dependencies:

    ```
    pip install -r Dependencies.txt
    ```

2. Download and install the Udacity Self-Driving Car Simulator.


The trained model will be saved as `model.h5`.

### Running the Simulation

1. **Start the Udacity Simulator**: Launch the Udacity Self-Driving Car Simulator and select "Autonomous Mode".

2. **Run the Inference Script**: Start the TensorFlow script to begin controlling the car.

    ```
    python drive.py model.h5
    ```

    This will establish a socket connection with the simulator and begin sending steering commands based on the model's predictions.

### Model Architecture

The CNN model is designed to process input images from the car's camera and predict the appropriate steering angle. The architecture consists of the following layers:

- **Convolutional Layers**: Extract spatial features from the images.
- **Dropout Layers**: Prevent overfitting during training.
- **Fully Connected Layers**: Combine the extracted features to make the final steering angle prediction.

### Results

After training, the car should be able to autonomously navigate the track in the Udacity simulator, maintaining a constant speed and steering through curves based on the model's predictions.

### Future Improvements

- **Speed Control**: Implement dynamic speed control based on road conditions.
- **Multiple Tracks**: Train the model on multiple tracks for better generalization.
- **Advanced Models**: Experiment with more complex architectures, such as recurrent neural networks (RNNs) or reinforcement learning.

## Acknowledgments

- The Udacity Self-Driving Car Simulator and the Udacity team for providing the open-source simulator.
- TensorFlow and Keras communities for the deep learning libraries.


