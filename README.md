# A Hybrid CNN-ViT Architecture for Image Classification

This repository contains the code to a fully functional Deep Learning Model that can be used for CIFAR-100 Classification. This code combines the local and spatial recognition of Convolutional Neural Networks, and the global context harnessed by Vision Transformers.

## Highlights of the Model:
- Combines the outputs of a ViT and CNN
- Uses a prefetch real-time augmentation pipeline
- Made with Tensorflow and Keras Frameworks
- Uses _classification_report_ to generate cohesive evaluation metrics
- Consists of heavy L2 regularization and Dropout
- Modular design by utilizing OOP [(Object-Oriented-Programming)](https://en.wikipedia.org/wiki/Object-oriented_programming)

## Requirements / Installations
- Need Tensorflow compatible Python [(3.9 ≤ Python ≤ 3.12)](https://www.python.org/downloads/)
- Need Tensorflow, Keras, Numpy, and Sklearn libraries 
(```pip install tensorflow keras numpy scikit-learn``` or ```pip install -r requirements.txt```)
- Optional (recommended): A virtual environment

## Usage
_Requirements have to be fulfilled beforehand._ 
1. Make sure dependencies are installed. 
2. Navigate to the project folder. 
3. Run: `python cifar_hybrid_classifier.py`

This command will run the model and automatically save the model as ```cifar_hybrid.keras``` after completion. Training automatically initiates upon running the script. _To terminate the script, close your terminal._  

## Results 
The model was not fully trained because of hardware limitations making training for 20+ epochs last 10+ hours. The results are not included yet, as they would be inconclusive without full training. Training metrics will be added when hardware allows for full training. 

## License
Licensed under the [**MIT License**](https://github.com/IamArav2012/Hybrid-ViT-CNN-Cifar100/blob/master/LICENSE). This license encourages open collaboration between the viewers of this repository. 

## Acknowledgments
This model was written using the CIFAR-100 dataset. Guidance and code structuring ideas were facilitated by [OpenAI's ChatGPT](https://chatgpt.com/). _For educational and research purposes only._

### Feedback or suggestions? Feel free to open an issue or discussion. Any collaboration is greatly encouraged.⭐ This repo If you found it helpful!
