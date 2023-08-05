
# Emotion Text Analysis
This comprehensive documentation outlines the process of performing text classification on the "Emotions Dataset" using various machine learning algorithms and a neural network model. The goal is to classify emotions expressed in textual comments and compare the performance of different approaches. 

Text classification is a fundamental task in natural language processing (NLP) that involves assigning predefined categories or labels to text documents. In this project, we focus on classifying emotions in textual comments using both traditional machine learning algorithms and a neural network model.

### Count Plot of Emotion labels
![](https://github.com/MukeshMushyakhwo/EmotionAnalysis/blob/main/Evaluatino_image/count.png?raw=true)

## Data Loading and Preprocessing
The dataset consists of training, testing, and validation data, with comments paired with corresponding emotion labels. The preprocessing steps include:

* Reading and loading data from files.
* Removing stopwords from text.
* Text Vectorization


# Traditional Machine Learning Models
## Multinomial Naive Bayes
The Multinomial Naive Bayes model is trained using TF-IDF vectorized data. Classification metrics, such as accuracy, confusion matrix, and classification report, are computed and visualized.

## Logistic Regression
Similar to the Naive Bayes model, Logistic Regression is trained and evaluated using TF-IDF features. Accuracy and classification metrics are computed.

## Support Vector Machines (SVM)
An SVM model with a linear kernel is trained and evaluated on the TF-IDF transformed data.

# Neural Network Model
## LSTM-Based Text Classification
An LSTM-based neural network is built for text classification. The text data is preprocessed using a TextVectorization layer, and the model architecture consists of an embedding layer, bidirectional LSTM layers, dense layers, and a softmax output layer

# Model Comparison and Evaluation
To compare the performance of the models, we evaluate them using various metrics:

### Multinomial Naive Bayes:
* Test Accuracy: 0.691

*  Confusion matrix
![](https://github.com/MukeshMushyakhwo/EmotionAnalysis/blob/main/Evaluatino_image/nb%20cm.png?raw=true)

### Logistic Regression:
* Test Accuracy: 0.871
* Confusion matrix
![](https://github.com/MukeshMushyakhwo/EmotionAnalysis/blob/main/Evaluatino_image/lr%20cm.png?raw=true)

### Support Vector Machines (SVM):
* Test Accuracy: 0.886
* Confusion matrix
![](https://github.com/MukeshMushyakhwo/EmotionAnalysis/blob/main/Evaluatino_image/svm%20cm.png?raw=true)

#### LSTM (Neural Network):
* Test Loss: 0.278

* Test Accuracy: 0.902

Training history plot showing train loss and validation loss  per epoch.
![](https://github.com/MukeshMushyakhwo/EmotionAnalysis/blob/main/Evaluatino_image/train_val_loss.png?raw=true)

Training history plot showing train accuracy and validation accuracy per epoch.
![](https://github.com/MukeshMushyakhwo/EmotionAnalysis/blob/main/Evaluatino_image/train_val_accuraacy.png?raw=true)

* Confusion Matrix
![](https://github.com/MukeshMushyakhwo/EmotionAnalysis/blob/main/Evaluatino_image/lstm%20cm.png?raw=true)


# Conclusion
This documentation illustrates the process of text classification on the Emotions Dataset using both traditional machine learning algorithms and a neural network model. The comparative evaluation of the models' performance reveals that the LSTM-based neural network outperforms the traditional approaches in terms of accuracy and overall emotion classification.


This comprehensive documentation provides a detailed explanation of the code, the different models employed, their performance metrics, and a comparative analysis of their results. 



