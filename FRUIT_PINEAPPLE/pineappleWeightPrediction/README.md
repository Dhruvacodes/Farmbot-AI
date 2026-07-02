# pineappleWeightPrediction
Pineapple Detection, Ripeness Classification, and Weight Prediction

 This research presents a framework utilizing YOLOv8 for pineapple detection and ripeness classification through drone imagery. After detecting and classifying pineapples, we employ bounding box and binary mask fitting techniques, including Circle Fitting (CF), Ellipse Fitting (EF), Circle Enclosing (CE), and Rotated Rectangle (RR), to estimate pineapple size by converting pixel-based measurements to millimetres. Finally, the current research predicted the weight of the pineapples by employing linear and nonlinear models. This work used linear models such as Linear Regression, Polynomial Regression, and nonlinear-based machine learning models such as Support Vector Regressor, Decision Tree, and K-Nearest Neighbor, with pixel counts as the input parameter for predicting pineapple weight.
The repository includes:
•	Link to datasets for:
o	Pineapple detection- https://drive.google.com/drive/folders/1yn1dZf5tUig7z4v0vjzPoOny1xCcj214?usp=sharing
o	Ripeness classification- https://drive.google.com/drive/folders/1xxwJyaf_T399AbUgSS8_66oiWVoQ39YC?usp=sharing
o	Size estimation dataset in size estimation and weight prediction folder
•	Colab notebooks for training and evaluation of YOLOv8 for pineapple detection and ripeness classification.
•	Size estimation using Bounding Box and mask fitting techniques.
•	Weight prediction using Machine learning models.
