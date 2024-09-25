# Food Item Recognition and Calorie Estimation

This project aims to develop a model capable of recognizing food items from images and estimating their calorie content. The dataset used for this project is the **Food11 Image Dataset**, which contains images of various foods. The goal of this project is to help users track their dietary intake and make informed decisions regarding their eating habits.

## Project Overview

Food item recognition is crucial for health and wellness applications. This project explores the application of deep learning to identify and classify food items into different categories. The model achieves a certain level of accuracy, demonstrating its effectiveness and potential for real-world applications.

## Dataset

The **Food11 Image Dataset** includes images representing 11 classes of food, captured under various conditions. The classes are as follows:

- Bread
- Dairy product
- Dessert
- Egg
- Fried food
- Meat
- Noodles-Pasta
- Rice
- Seafood
- Soup
- Vegetable-Fruit

Each class contains a varying number of images, providing diverse input data to train the model.

## Model Architecture

The model is built using **Keras** and **TensorFlow**. The architecture consists of multiple convolutional layers followed by max-pooling and fully connected (dense) layers. Here’s an overview of the architecture:

- **Input Layer**: RGB images resized to 224x224.
- **Convolutional Layers**: Several convolutional layers with a varying number of filters, followed by ReLU and max-pooling.
- **Dropout Layers**: Used to prevent overfitting by randomly dropping units during training.
- **Fully Connected Layers**: Dense layers, where the first one has a variable number of units and the output layer has 11 units (one for each class).
- **Output Layer**: Uses the Softmax activation function for multiclass classification.

## Training and Evaluation

- **Training**: The model is trained on 80% of the dataset using the Adam optimizer and a categorical cross-entropy loss function.
- **Validation**: 10% of the data is used for validation during training to monitor the model’s performance and prevent overfitting.
- **Testing**: The remaining 10% is used to evaluate the final performance of the model.

## Installation and Setup

1. Clone the repository from GitHub:
   ```bash
   git clone https://github.com/sihemaf/PRODIGY_ML_05.git
2. Change into the project directory:
   ```bash
   cd PRODIGY_ML_05
   
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
## Usage
1. Ensure the dataset is organized in the correct structure as described in the project.

2. To train the model, use the following command: 
   ```bash
   python train_model.py
   
3. Once the model is trained, you can evaluate it on the test set by running:
   ```bash
   python evaluate_model.py
   
4. You can also visualize the training process and the accuracy/loss curves:
   ```bash
    python plot_training_curves.py
   
## Results and Visualizations
The following plots demonstrate the training and validation accuracy and loss over the epochs:

- **Accuracy Curve**:The model achieves high accuracy with smooth convergence.
- **Loss Curve**: The loss function converges consistently, showing minimal signs of overfitting.

## Future Work
- Implement real-time food recognition using a webcam.
- Experiment with different CNN architectures or transfer learning to further improve accuracy.
- Explore data augmentation techniques to make the model more robust to different lighting conditions and food positions.
- Test the model in a real-world application, such as a dietary tracking system.
## Acknowledgments
- Contributors of the **Food11 Image Dataset** for providing a diverse collection of food images.
- The open-source community for tools and resources that supported this project.
- Keras and TensorFlow for making deep learning accessible and efficient.
## License
This project is licensed under the MIT License. See the LICENSE file for more details.
