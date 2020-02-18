# NextVALD Attention Video Classification Model  (Not Finished)
### **An Efficient Neural Network for Video Classification Challenge (No Commercial Using)**
#### Project Introduction
With the prevalence of smart phones and digital cameras in normal life, exponentially increasing images and videos are created, uploaded, watched and shared through internet. In the past few years, _Convolutional Neural Networks_ (CNNs) have been demonstrated as an effective class of models for understanding image content, such as recognition, segmentation, detection and retrieval. The key enabling factors behind these results were mass computation power of GPUs and large-scale datasets such as ImageNet. In a similar vein, the amount and size of video benchmarks has also been growing recently, such as _UCF101_ (UCF), _Kinetics-700_ (Deep Mind) and _Youtube-8M_ (Google AI), which makes video content understanding gradually under an efficient speed of development in many real-world applications. Meanwhile, many techniques related to video representation and video classification still faces a series of challenges.

#### Training DataSet
Youtube-8M is a large-scale benchmark for general multi-label video classification and consists of about 6.1M videos from Youtube.com, each of which has at least 1000 views with video time ranging from 120 to 300 seconds and is labeled with one or multiple tags (labels) from a vocabulary of 3862 visual entities. 
https://research.google.com/youtube8m/explore.html


#### Video Feature Vectorization
1.	decode each video to N frames (N equals from 1 to 360, Maximum 6 mins), one frame per second.

2.	feed the decoded frames into the Inception V3, a CNN based neural network, and fetch the ReLu activation of the last hidden layer, before the classification layer.

3. apply Whitening, PCA, Batch Normalization to reduce feature dimensions to 1024, 4, followed by quantization. PCA are used to reduce the dimension of the data. The goal of whitening is to make the input less redundant
4. apply NetVLAD Aggregation Network.


#### Audio Feature Extraction
1.	use VGGish model converts audio input features into a semantically meaningful, high-level 128-D embedding which can be fed as input to a downstream classification model.

2.	apply PCA (+ whitening) to reduce feature dimensions to 1024, 4, followed by quantization

3.	apply NetVLAD Aggregation Network. This part is the same as video feature vectorization, where V(j,k): video-level descriptor is 128*128. 	

#### Mode Architecture (wait to be added)


# Dependencies
- Tensorflow >= 1.4


# History
- Jan 15, 2020: Basic Model



# Usage Instructions

**Paper** : Wait to be added


**Download Dataset** 

``curl data.yt8m.org/download.py | partition=2/frame/train mirror=us python ``

``curl data.yt8m.org/download.py | partition=2/frame/validate mirror=us python ``

``curl data.yt8m.org/download.py | partition=2/frame/test mirror=us python``

Other methods by Links:

http://us.data.yt8m.org/2/frame/train/index.html  

http://us.data.yt8m.org/2/frame/validate/index.html  

http://us.data.yt8m.org/2/frame/test/index.html 


## **Training Model**
python offline_train.py
