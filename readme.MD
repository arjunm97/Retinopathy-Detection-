# Summary of the project ###

Diabetic Retinopathy (DR) is one of the major causes of blindness . Increasing life expectancy, indulgent lifestyles and other contributing factors mean the number of people with diabetes is projected to continue rising. Regular screening of diabetic patients
for DR has been shown to be a cost-effective and important aspect of their care. The
accuracy and timing of this care is of significant importance to both the cost and effectiveness of treatment. If detected early enough, effective treatment of DR is available,
making this a vital process.Classification of DR involves the weighting of numerous
features and the location of such features. This is highly time consuming for clinicians. Computers are able to obtain much quicker classifications once trained, giving
the ability to aid clinicians in real-time classification. The efficacy of automated grading for DR has been an active area of research in computer imaging with encouraging
conclusions . We use CNN to predict DR or no DR. We develop a network with CNN
architecture and data augmentation which can identify the intricate features involved
in the classification task.
Diabetic Retinopathy (DR) grade classification has been regarded as a critical step for
evaluation and management of diabetes retinopathy. Because of damages of the retina
blood vessels caused by the high blood glucose level, different extent of microstructures, such as microanuerysms, hard exudates, and neovascularization, could occupy
the retina area thus damaging it.Most grading protocols are based on classification systems for DR which track the appearance and progression of disease. Here grading helps
the patients in knowing the severity of DR and to take required treatment.

* Models used =Random Forest,CNN,Vgg16,Navie bayes,Logistic regression,KNN 
* Weights used    - **ImageNet**
* Classifier used - **Logistic Regression**


### Dependencies ###
* TensorFlow pip install tensorflow`
* Keras ` pip install keras`
* NumPy `pip install numpy`
* matplotlib  pip install matplotlib` and you also need to do this `sudo apt-get install python-dev`
* seaborn `sudo pip install seaborn`
* h5py `pip install h5py`
* scikit-learn ` pip install scikit-learn`

### System requirements
* This project used Windows 10 

test.py = tests the vgg16 model 
train.py =trains the vgg16 model 

feature_extracter=extracts features from the images 
Cnn_machine_learning=Jupiter file for all machine learning algorithm (Random Forest,Navie bayes,Logistic regression,KNN )

final_project=Project report 
cnn_aug=cnn model for detecting ratinopathy 

Dataset used=Fundas Dataset [https://www.kaggle.com/linchundan/fundusimage1000]
Divide the dataset into two parts
i)Images with DR detected 
ii)Images where DR is not detected 

Use Augmentation to increase the number of images to 4000
Then use the models
