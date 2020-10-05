# Summary of the project ###

* Pretrained state-of-the-art neural networks are used on *University of Oxford's* **FLOWERS17** and **FLOWERS102** dataset.
* Models used     - **Xception, Inception-v3, OverFeat, ResNet50, VGG16, VGG19**.
* Weights used    - **ImageNet**
* Classifier used - **Logistic Regression**
* Tutorial for this work is available at - [Using Pre-trained Deep Learning models for your own dataset](https://gogul09.github.io/software/flower-recognition-deep-learning)

**Update (16/12/2017)**: Included two new deep neural net models namely `InceptionResNetv2` and `MobileNet`.

### Dependencies ###
*  TensorFlow pip install tensorflow`
* Keras ` pip install keras`
* NumPy `pip install numpy`
* matplotlib  pip install matplotlib` and you also need to do this `sudo apt-get install python-dev`
* seaborn `sudo pip install seaborn`
* h5py `pip install h5py`
* scikit-learn ` pip install scikit-learn`

### System requirements
* This project used Windows 10 



### Usage ###
* Organize dataset                      - `python organize_flowers17.py`
* Feature extraction using CNN          - `python extract_features.py`
* Train model using Logistic Regression - `python train.py`

### Show me the numbers ###
The below tables shows the accuracies obtained for every Deep Neural Net model used to extract features from **FLOWERS17** dataset using different parameter settings.

* Result-1
  
  * test_size  : **0.10**
  * classifier : **Logistic Regression**
  


| Model        | Rank-1 accuracy | Rank-5 accuracy |
|--------------|-----------------|-----------------|
| Xception     | 93.38%          | <b>99.75%</b>          |
| Inception-v3 | <b>96.81%</b>          | 99.51%          |
| VGG16        | 88.24%          | 99.02%          |
| VGG19        | 88.73%          | 98.77%          |
| ResNet50     | 59.80%          | 86.52%          |
| MobileNet     | 96.32%         | <b>99.75%</b>         |
| Inception<br>ResNetV2     | 88.48%          | 99.51%          |
