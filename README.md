<!--
Title: Awesome Core ML Models
Description: A curated list of machine learning models in Core ML format.
Author: Kedan Li
-->

<p align="center">
<img src="images/coreml.png" width="329" height="295"/>
</p>

Since iOS 11, Apple released Core ML framework to help developers integrate machine learning models into applications. [The official documentation](https://developer.apple.com/documentation/coreml)

We've put up the largest collection of machine learning models in Core ML format, to help  iOS, macOS, tvOS, and watchOS developers experiment with machine learning techniques.

If you've converted a Core ML model, feel free to submit a [pull request](https://github.com/likedan/Awesome-CoreML-Models/compare) ⭐ 6,942 | 🐛 11 | 🌐 Python | 📅 2025-06-17.

Recently, we've included visualization tools. And here's one [Netron](https://lutzroeder.github.io/Netron).

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) ⭐ 436,588 | 🐛 68 | 📅 2026-01-28
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

# Models

## Image - Metadata/Text

*Models that take image data as input and output useful information about the image.*

* **PoseEstimation** - Estimating human pose from a picture for mobile. [Download](https://github.com/edvardHua/PoseEstimationForMobile/tree/master/release) ⭐ 1,020 | 🐛 91 | 🌐 C++ | 📅 2023-03-24 | [Demo](https://github.com/tucan9389/PoseEstimation-CoreML) ⭐ 699 | 🐛 26 | 🌐 Swift | 📅 2021-08-13 | [Reference](https://github.com/edvardHua/PoseEstimationForMobile) ⭐ 1,020 | 🐛 91 | 🌐 C++ | 📅 2023-03-24
* **YOLO** - Recognize what the objects are inside a given image and where they are in the image. [Download](https://github.com/hollance/YOLO-CoreML-MPSNNGraph/blob/master/TinyYOLO-CoreML/TinyYOLO-CoreML/TinyYOLO.mlmodel) ⭐ 946 | 🐛 27 | 🌐 Swift | 📅 2019-11-19 | [Demo](https://github.com/hollance/YOLO-CoreML-MPSNNGraph) ⭐ 946 | 🐛 27 | 🌐 Swift | 📅 2019-11-19 | [Reference](http://machinethink.net/blog/object-detection-with-yolo)
* **MobileNet** - Detects the dominant objects present in an image. [Download](https://github.com/hollance/MobileNet-CoreML/raw/master/MobileNet.mlmodel) ⭐ 712 | 🐛 5 | 🌐 Swift | 📅 2018-09-22 | [Demo](https://github.com/hollance/MobileNet-CoreML) ⭐ 712 | 🐛 5 | 🌐 Swift | 📅 2018-09-22 | [Reference](https://arxiv.org/abs/1704.04861)
* **Places CNN** - Detects the scene of an image from 205 categories such as bedroom, forest, coast etc. [Download](https://github.com/hollance/MobileNet-CoreML/raw/master/MobileNet.mlmodel) ⭐ 712 | 🐛 5 | 🌐 Swift | 📅 2018-09-22 | [Demo](https://github.com/chenyi1989/CoreMLDemo) ⭐ 34 | 🐛 0 | 🌐 Objective-C | 📅 2017-06-12 | [Reference](http://places.csail.mit.edu/index.html)
* **ImageSegmentation** - Segment the pixels of a camera frame or image into a predefined set of classes. [Download](https://developer.apple.com/machine-learning/models/) | [Demo](https://github.com/tucan9389/ImageSegmentation-CoreML) ⭐ 341 | 🐛 4 | 🌐 Swift | 📅 2021-03-27 | [Reference](https://github.com/tensorflow/models/tree/master/research/deeplab) ⭐ 77,695 | 🐛 1,274 | 🌐 Python | 📅 2026-02-10
* **AgeNet** - Predict a person's age from one's portrait. [Download](https://drive.google.com/file/d/0B1ghKa_MYL6mT1J3T1BEeWx4TWc/view?usp=sharing) | [Demo](https://github.com/cocoa-ai/FacesVisionDemo) ⭐ 326 | 🐛 3 | 🌐 Swift | 📅 2019-10-08 | [Reference](http://www.openu.ac.il/home/hassner/projects/cnn_agegender/)
* **GenderNet** - Predict a person's gender from one's portrait. [Download](https://drive.google.com/file/d/0B1ghKa_MYL6mYkNsZHlyc2ZuaFk/view?usp=sharing) | [Demo](https://github.com/cocoa-ai/FacesVisionDemo) ⭐ 326 | 🐛 3 | 🌐 Swift | 📅 2019-10-08 | [Reference](http://www.openu.ac.il/home/hassner/projects/cnn_agegender/)
* **EmotionNet** - Predict a person's emotion from one's portrait. [Download](https://drive.google.com/file/d/0B1ghKa_MYL6mTlYtRGdXNFlpWDQ/view?usp=sharing) | [Demo](https://github.com/cocoa-ai/FacesVisionDemo) ⭐ 326 | 🐛 3 | 🌐 Swift | 📅 2019-10-08 | [Reference](http://www.openu.ac.il/home/hassner/projects/cnn_emotions/)
* **Inception v3** - Detects the dominant objects present in an image. [Download](https://github.com/yulingtianxia/Core-ML-Sample/blob/master/CoreMLSample/Inceptionv3.mlmodel) ⭐ 221 | 🐛 1 | 🌐 Swift | 📅 2017-08-09 | [Demo](https://github.com/yulingtianxia/Core-ML-Sample/) ⭐ 221 | 🐛 1 | 🌐 Swift | 📅 2017-08-09 | [Reference](https://arxiv.org/abs/1512.00567)
* **Car Recognition** - Predict the brand & model of a car. [Download](https://github.com/likedan/Core-ML-Car-Recognition/blob/master/Convert/CarRecognition.mlmodel) ⭐ 173 | 🐛 4 | 🌐 Swift | 📅 2018-05-22 | [Demo](https://github.com/ytakzk/CoreML-samples) ⭐ 44 | 🐛 1 | 🌐 Jupyter Notebook | 📅 2017-07-20 | [Reference](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html)
* **Food101** - Predict the type of foods from images. [Download](https://drive.google.com/open?id=0B5TjkH3njRqnVjBPZGRZbkNITjA) | [Demo](https://github.com/ph1ps/Food101-CoreML) ⚠️ Archived | [Reference](http://visiir.lip6.fr/explore)
* **DepthPrediction** - Predict the depth from a single image. [Download](https://developer.apple.com/machine-learning/models/) | [Demo](https://github.com/tucan9389/DepthPrediction-CoreML) ⭐ 140 | 🐛 1 | 🌐 Swift | 📅 2021-06-15 | [Reference](https://github.com/iro-cp/FCRN-DepthPrediction) ⭐ 1,121 | 🐛 42 | 🌐 Python | 📅 2019-08-26
* **Nudity** - Classifies an image either as NSFW (nude) or SFW (not nude)
  [Download](https://drive.google.com/open?id=0B5TjkH3njRqncDJpdDB1Tkl2S2s) | [Demo](https://github.com/ph1ps/Nudity-CoreML) ⚠️ Archived | [Reference](https://github.com/yahoo/open_nsfw) ⚠️ Archived
* **Oxford102** - Detect the type of flowers from images. [Download](https://drive.google.com/file/d/0B1ghKa_MYL6meDBHT2NaZGxkNzQ/view?usp=sharing) | [Demo](https://github.com/cocoa-ai/FlowersVisionDemo) ⭐ 104 | 🐛 0 | 🌐 Swift | 📅 2024-03-24 | [Reference](http://jimgoo.com/flower-power/)
* **TextRecognition (ML Kit)** - Recognizing text using ML Kit built-in model in real-time. [Download](origin) | [Demo](https://github.com/tucan9389/TextRecognition-MLKit) ⭐ 95 | 🐛 1 | 🌐 Swift | 📅 2019-06-25 | [Reference](https://firebase.google.com/docs/ml-kit/ios/recognize-text)
* **TextDetection** - Detecting text using Vision built-in model in real-time. [Download](origin) | [Demo](https://github.com/tucan9389/TextDetection-CoreML) ⭐ 82 | 🐛 0 | 🌐 Swift | 📅 2019-02-20 | [Reference](https://developer.apple.com/documentation/vision)
* **PhotoAssessment** - Photo Assessment using Core ML and Metal. [Download](https://github.com/yulingtianxia/PhotoAssessment/blob/master/PhotoAssessment-Sample/Sources/NIMANasnet.mlmodel) ⚠️ Archived | [Demo](https://github.com/yulingtianxia/PhotoAssessment) ⚠️ Archived | [Reference](https://arxiv.org/abs/1709.05424)
* **MNIST** - Predict handwritten (drawn) digits from images. [Download](https://github.com/ph1ps/MNIST-CoreML/raw/master/MNISTPrediction/MNIST.mlmodel) ⚠️ Archived | [Demo](https://github.com/ph1ps/MNIST-CoreML) ⚠️ Archived | [Reference](http://yann.lecun.com/exdb/mnist/)
* **RN1015k500** - Predict the location where a picture was taken. [Download](https://s3.amazonaws.com/aws-bigdata-blog/artifacts/RN1015k500/RN1015k500.mlmodel) | [Demo](https://github.com/awslabs/MXNet2CoreML_iOS_sample_app) ⚠️ Archived | [Reference](https://aws.amazon.com/blogs/ai/estimating-the-location-of-images-using-mxnet-and-multimedia-commons-dataset-on-aws-ec2)
* **FlickrStyle** - Detect the artistic style of images. [Download](https://drive.google.com/file/d/0B1ghKa_MYL6meDBHT2NaZGxkNzQ/view?usp=sharing) | [Demo](https://github.com/cocoa-ai/StylesVisionDemo) ⭐ 48 | 🐛 0 | 🌐 Swift | 📅 2019-10-08 | [Reference](http://sergeykarayev.com/files/1311.3715v3.pdf)
* **ResNet50** - Detects the dominant objects present in an image. [Download](https://github.com/ytakzk/CoreML-samples/blob/master/CoreML-samples/Resnet50.mlmodel) ⭐ 44 | 🐛 1 | 🌐 Jupyter Notebook | 📅 2017-07-20 | [Demo](https://github.com/ytakzk/CoreML-samples) ⭐ 44 | 🐛 1 | 🌐 Jupyter Notebook | 📅 2017-07-20 | [Reference](https://arxiv.org/abs/1512.03385)
* **VGG16** - Detects the dominant objects present in an image. [Download](https://docs-assets.developer.apple.com/coreml/models/VGG16.mlmodel) | [Demo](https://github.com/alaphao/CoreMLExample) ⭐ 37 | 🐛 0 | 🌐 Swift | 📅 2017-06-23 | [Reference](https://arxiv.org/abs/1409.1556)
* **SentimentVision** - Predict positive or negative sentiments from images. [Download](https://drive.google.com/open?id=0B1ghKa_MYL6mZ0dITW5uZlgyNTg) | [Demo](https://github.com/cocoa-ai/SentimentVisionDemo) ⭐ 37 | 🐛 2 | 🌐 Swift | 📅 2019-10-08 | [Reference](http://www.sciencedirect.com/science/article/pii/S0262885617300355?via%3Dihub)

## Image - Image

*Models that transform images.*

* **AnimeScale2x** - Process a bicubic-scaled anime-style artwork [Download](https://github.com/imxieyi/waifu2x-ios/blob/master/waifu2x/models/anime_noise0_model.mlmodel) ⭐ 591 | 🐛 14 | 🌐 Swift | 📅 2022-12-14 | [Demo](https://github.com/imxieyi/waifu2x-ios) ⭐ 591 | 🐛 14 | 🌐 Swift | 📅 2022-12-14 | [Reference](https://arxiv.org/abs/1501.00092)
* **HED** - Detect nested edges from a color image. [Download](https://github.com/s1ddok/HED-CoreML/blob/master/HED-CoreML/Models/HED_so.mlmodel) ⭐ 111 | 🐛 0 | 🌐 Swift | 📅 2017-07-03 | [Demo](https://github.com/s1ddok/HED-CoreML) ⭐ 111 | 🐛 0 | 🌐 Swift | 📅 2017-07-03 | [Reference](http://dl.acm.org/citation.cfm?id=2654889)

## Text - Metadata/Text

*Models that process text data*

* **BERT for Question answering** - Swift Core ML 3 implementation of BERT for Question answering [Download](https://github.com/huggingface/swift-coreml-transformers/blob/master/Resources/BERTSQUADFP16.mlmodel) ⚠️ Archived | [Demo](https://github.com/huggingface/swift-coreml-transformers#-bert) ⚠️ Archived | [Reference](https://github.com/huggingface/pytorch-transformers#run_squadpy-fine-tuning-on-squad-for-question-answering) ⭐ 156,343 | 🐛 2,209 | 🌐 Python | 📅 2026-02-10
* **GPT-2** - OpenAI GPT-2 Text generation (Core ML 3) [Download](https://github.com/huggingface/swift-coreml-transformers/blob/master/Resources/gpt2-512.mlmodel) ⚠️ Archived | [Demo](https://github.com/huggingface/swift-coreml-transformers#-gpt-2) ⚠️ Archived | [Reference](https://github.com/huggingface/pytorch-transformers) ⭐ 156,343 | 🐛 2,209 | 🌐 Python | 📅 2026-02-10
* **Sentiment Polarity** - Predict positive or negative sentiments from sentences. [Download](https://github.com/cocoa-ai/SentimentCoreMLDemo/raw/master/SentimentPolarity/Resources/SentimentPolarity.mlmodel) ⭐ 123 | 🐛 2 | 🌐 Swift | 📅 2018-10-07 | [Demo](https://github.com/cocoa-ai/SentimentCoreMLDemo) ⭐ 123 | 🐛 2 | 🌐 Swift | 📅 2018-10-07 | [Reference](http://boston.lti.cs.cmu.edu/classes/95-865-K/HW/HW3/)
* **DocumentClassification** - Classify news articles into 1 of 5 categories. [Download](https://github.com/toddkramer/DocumentClassifier/blob/master/Sources/DocumentClassification.mlmodel) ⭐ 46 | 🐛 1 | 🌐 Swift | 📅 2019-04-20 | [Demo](https://github.com/toddkramer/DocumentClassifier) ⭐ 46 | 🐛 1 | 🌐 Swift | 📅 2019-04-20 | [Reference](https://github.com/toddkramer/DocumentClassifier/) ⭐ 46 | 🐛 1 | 🌐 Swift | 📅 2019-04-20
* **iMessage Spam Detection** - Detect whether a message is spam. [Download](https://github.com/gkswamy98/imessage-spam-detection/blob/master/MessageClassifier.mlmodel) ⭐ 36 | 🐛 0 | 🌐 Swift | 📅 2017-06-29 | [Demo](https://github.com/gkswamy98/imessage-spam-detection/tree/master) ⭐ 36 | 🐛 0 | 🌐 Swift | 📅 2017-06-29 | [Reference](http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/)
* **NamesDT** - Gender Classification using DecisionTreeClassifier [Download](https://github.com/cocoa-ai/NamesCoreMLDemo/blob/master/Names/Resources/NamesDT.mlmodel) ⭐ 36 | 🐛 1 | 🌐 Swift | 📅 2018-10-07 | [Demo](https://github.com/cocoa-ai/NamesCoreMLDemo) ⭐ 36 | 🐛 1 | 🌐 Swift | 📅 2018-10-07 | [Reference](http://nlpforhackers.io/)
* **Personality Detection** - Predict personality based on user documents (sentences). [Download](https://github.com/novinfard/profiler-sentiment-analysis/tree/master/ios_app/ProfilerSA/ML%20Models) ⭐ 19 | 🐛 2 | 🌐 Swift | 📅 2019-01-14 | [Demo](https://github.com/novinfard/profiler-sentiment-analysis/) ⭐ 19 | 🐛 2 | 🌐 Swift | 📅 2019-01-14 | [Reference](https://github.com/novinfard/profiler-sentiment-analysis/blob/master/dissertation-v6.pdf) ⭐ 19 | 🐛 2 | 🌐 Swift | 📅 2019-01-14

## Miscellaneous

* **GestureAI** - Recommend an artist based on given location and genre. [Download](https://goo.gl/avdMjD) | [Demo](https://github.com/akimach/GestureAI-CoreML-iOS) ⭐ 162 | 🐛 1 | 🌐 Swift | 📅 2023-02-26 | [Reference](https://github.com/akimach/GestureAI-iOS/tree/master/GestureAI) ⭐ 162 | 🐛 1 | 🌐 Swift | 📅 2023-02-26
* **Exermote** - Predicts the exercise, when iPhone is worn on right upper arm. [Download](https://github.com/Lausbert/Exermote/tree/master/ExermoteInference) ⭐ 132 | 🐛 1 | 🌐 Swift | 📅 2025-05-14 | [Demo](https://github.com/Lausbert/Exermote/tree/master/ExermoteInference) ⭐ 132 | 🐛 1 | 🌐 Swift | 📅 2025-05-14 | [Reference](http://lausbert.com/2017/08/03/exermote/)
* **Artists Recommendation** - Recommend an artist based on given location and genre. [Download](https://github.com/agnosticdev/Blog-Examples/blob/master/UsingCoreMLtoCreateASongRecommendationEngine/Artist.mlmodel) ⭐ 19 | 🐛 1 | 🌐 C | 📅 2019-10-31 | [Demo](origin) | [Reference](https://www.agnosticdev.com/blog-entry/python/using-scikit-learn-and-coreml-create-music-recommendation-engine)
* **ChordSuggester** - Predicts the most likely next chord based on the entered Chord Progression. [Download](https://github.com/carlosmbe/Mac-CoreML-Chord-Suggester/blob/main/MLChordSuggester.mlpackage.zip) ⭐ 9 | 🐛 0 | 🌐 Swift | 📅 2023-07-30 | [Demo](https://github.com/carlosmbe/Mac-CoreML-Chord-Suggester/tree/main) ⭐ 9 | 🐛 0 | 🌐 Swift | 📅 2023-07-30 | [Reference](https://medium.com/@huanlui/chordsuggester-i-3a1261d4ea9e)

## Speech Processing

* **Streaming ASR** – Real-time streaming speech recognition engine for iOS. Uses Fast Conformer + CTC, runs fully on device.\
  [Download](https://github.com/Otosaku/OtosakuStreamingASR-iOS/releases) ⭐ 11 | 🐛 0 | 🌐 Swift | 📅 2025-06-14 | [Demo](https://github.com/Otosaku/OtosakuStreamingASR-iOS) ⭐ 11 | 🐛 0 | 🌐 Swift | 📅 2025-06-14 | [Reference](https://github.com/Otosaku/OtosakuStreamingASR-iOS) ⭐ 11 | 🐛 0 | 🌐 Swift | 📅 2025-06-14
* **Keyword Spotting (KWS)** – On-device keyword spotting engine using lightweight CRNN architecture, optimized for mobile devices.\
  [Download](https://github.com/Otosaku/OtosakuKWS-iOS/releases) ⭐ 11 | 🐛 0 | 🌐 Swift | 📅 2025-06-14 | [Demo](https://github.com/Otosaku/OtosakuKWS-iOS) ⭐ 11 | 🐛 0 | 🌐 Swift | 📅 2025-06-14 | [Reference](https://github.com/Otosaku/OtosakuKWS-iOS) ⭐ 11 | 🐛 0 | 🌐 Swift | 📅 2025-06-14

# Visualization Tools

*Tools that help visualize CoreML Models*

* [Netron](https://lutzroeder.github.io/Netron)

# Supported formats

*List of model formats that could be converted to Core ML with examples*

* [Caffe](https://apple.github.io/coremltools/generated/coremltools.converters.caffe.convert.html)
* [Keras](https://apple.github.io/coremltools/generated/coremltools.converters.keras.convert.html)
* [XGBoost](https://apple.github.io/coremltools/generated/coremltools.converters.xgboost.convert.html)
* [Scikit-learn](https://apple.github.io/coremltools/generated/coremltools.converters.sklearn.convert.html)
* [MXNet](https://aws.amazon.com/blogs/ai/bring-machine-learning-to-ios-apps-using-apache-mxnet-and-apple-core-ml/)
* [LibSVM](https://apple.github.io/coremltools/generated/coremltools.converters.libsvm.convert.html)
* [Torch7](https://github.com/prisma-ai/torch2coreml) ⚠️ Archived

# The Gold

*Collections of machine learning models that could be converted to Core ML*

* [TensorFlow Models](https://github.com/tensorflow/models) ⭐ 77,695 | 🐛 1,274 | 🌐 Python | 📅 2026-02-10 - Models for TensorFlow.
* [TensorFlow Slim Models](https://github.com/tensorflow/models/tree/master/research/slim/README.md) ⭐ 77,695 | 🐛 1,274 | 🌐 Python | 📅 2026-02-10 - Another collection of TensorFlow Models.
* [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo) ⭐ 34,835 | 🐛 1,176 | 🌐 C++ | 📅 2024-07-31 - Big list of models in Caffe format.
* [MXNet Model Zoo](https://mxnet.incubator.apache.org/model_zoo/) - Collection of MXNet models.

*Individual machine learning models that could be converted to Core ML. We'll keep adjusting the list as they become converted.*

* [Colorization](https://github.com/richzhang/colorization) ⭐ 3,464 | 🐛 56 | 🌐 Python | 📅 2023-11-27 Automatic colorization using deep neural networks.
* [Image Analogy](https://github.com/msracver/Deep-Image-Analogy) ⭐ 1,371 | 🐛 22 | 🌐 C++ | 📅 2021-09-27 Find semantically-meaningful dense correspondences between two input images.
* [CTPN](https://github.com/tianzhi0549/CTPN) ⭐ 1,287 | 🐛 70 | 🌐 Jupyter Notebook | 📅 2021-10-15 Detecting text in natural image.
* [Illustration2Vec](https://github.com/rezoo/illustration2vec) ⭐ 675 | 🐛 22 | 🌐 Python | 📅 2019-01-18 Estimating a set of tags and extracting semantic feature vectors from given illustrations.
* [mtcnn](https://github.com/CongWeilin/mtcnn-caffe) ⭐ 496 | 🐛 52 | 🌐 Python | 📅 2018-10-01 Joint Face Detection and Alignment.
* [Fashion Detection](https://github.com/liuziwei7/fashion-detection) ⭐ 493 | 🐛 5 | 🌐 MATLAB | 📅 2021-10-09 Cloth detection from images.
* [Saliency](https://github.com/imatge-upc/saliency-2016-cvpr) ⭐ 188 | 🐛 7 | 🌐 Python | 📅 2019-12-10 The prediction of salient areas in images has been traditionally addressed with hand-crafted features.
* [ILGnet](https://github.com/BestiVictory/ILGnet) ⭐ 114 | 🐛 9 | 🌐 Python | 📅 2018-05-06 The aesthetic evaluation of images.
* [iLID](https://github.com/twerkmeister/iLID) ⚠️ Archived Automatic spoken language identification.
* [deephorizon](https://github.com/scottworkman/deephorizon) ⭐ 42 | 🐛 0 | 🌐 Python | 📅 2022-04-28 Single image horizon line estimation.
* [Face Detection](https://github.com/DolotovEvgeniy/DeepPyramid) ⭐ 23 | 🐛 2 | 🌐 C++ | 📅 2016-10-13 Detect face from image.
* [LaMem](https://github.com/MiyainNYC/Visual-Memorability-through-Caffe) ⭐ 18 | 🐛 0 | 🌐 Jupyter Notebook | 📅 2016-04-30 Score the memorability of pictures.

# Contributing and License

* [See the guide](https://github.com/likedan/Awesome-CoreML-Models/blob/master/.github/CONTRIBUTING.md) ⭐ 6,942 | 🐛 11 | 🌐 Python | 📅 2025-06-17
* Distributed under the MIT license. See LICENSE for more information.
