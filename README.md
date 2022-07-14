INTRODUCTION-
Our project DoctorAI is a web application where users upload the lungs Xrays and
get the diagnosis of their lungs. DoctorAI is a full stack website where authenti-
cated users can upload their lungs Xrays and get the diagnosis and result instantly.
The model in the backend is trained on the NIH chest X-ray image dataset is col-
lected from Kaggle repository and it is fully an open source platform.
With the control of computers along with the huge volume of records being unre-
stricted to the public, this is a high time to resolve this complication. This solution
can put up decreasing medical costs with the enlargement of computer science for
health and medical science projects.



Collection of Data for Creating Model
Created data set from two sources.
1.https://github.com/ieee8023/covid-chestxray-dataset.
2.X-ray image dataset is collected from Kaggle-https://www.kaggle.com/praveengovi/
coronahack-chest-xraydataset
Divided the dataset to two folders ’Pneumonia Positive’, ’Pneumonia Negative’.
For the positive we collected around 500 images and for the negative we collected
around 1600 images.

Training the CNN model
GoogLeNet Model – CNN Architecture
Modified GoogleNet Model
We customized 2 layers in GoogleNet Model.
• 142th layer of GoogleNet is the feature learner layer. We need our model to
learn features from our training dataset folders. We modified the layer to a
fully connected layer with Weight Learn Factor as ten and Bias Learn Factor
as 10.
• 144th layer of GoogleNet is used to classify 1000 different objects, but we
changed it to classify the image in two- Corona Positive and Corona Negative.
We trained our model with training data having the mini-batch size of 4, and The
epochs were 4. The initial learning rate was 0.0003 for minimum loss, though the
training time was increased.

We achieved an accuracy of 95.4%.
The accuracy is much better as compared to standard CNN model

![image](https://user-images.githubusercontent.com/58679695/179061955-924cffd0-bc55-4fc0-8296-88c0f1c6aff9.png)
![image](https://user-images.githubusercontent.com/58679695/179062008-60a2e64c-fe58-4440-96a7-1cfdbb091d4a.png)
![image](https://user-images.githubusercontent.com/58679695/179062025-dab7b755-8de4-4263-b808-bdc5c244998f.png)



