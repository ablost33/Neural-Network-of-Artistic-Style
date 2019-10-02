Alexander Blostein

Project Deliverable 1

Project: Neural Network for Artistic Style


 1. DataSet:  
First of all I would base my project on the following paper: https://arxiv.org/pdf/1508.06576.pdf
For this project I will use the following dataset: https://www.kaggle.com/basu369victor/style-transfer-deep-learning-algorithm/data.
This dataset contains artwork of the 50 most influential artists of all time. I am particularly interested in the resizes.zip file, as it contains the same collection of images but resized and extracted from their folder structure.
I would preprocess the image by modifying the dimensions, and with the help of VGG19. VGG19 is a model whose weights are pretrained on a dataset of roughly 1.3 million images. 


2. Methodology: 
Feasibility: I think this project is very feasible as there is a ton of documentation on how to implement this and its been done many times before. The fact that I’m using a VGG19 will make the data processing also feasible.
Machine Learning Model: For this project, I shall be implementing a Convolutional Neural Network. The reason we need CNNs is because we are performing a style transfer. I essentially want to input an image and a painting into my neural network, and then transfer the style of the painting into the image. The reasoning behind this is that CNNs learn to encode what images represent and what contents are visible in the image and their hidden units can detect more complex features from a given image. I haven’t really considered any other methods as every single paper I’ve read on the topic use CNN.
Final Conceptualization:
	Depending on how much time I have for the project, I was thinking of either creating a web app using NodeJS where you can upload a picture and a painting and then it performs the task. If I have a little more time, I was thinking of doing an iPhone app instead using swift where you can take pictures and upload a painting and then the style transfer is performed.