# mnist
mnist handwritten dataset

1. This repository contains handwritten mnist dataset. The train and test data is collected from kaggle competition.
2. One is required to run the "Mnist.py" file in order to generate a Random forest model and then predict a new digit by using predict method.
3. To create a new image go to paint > create a black background > write a digit(0-9) using a rubber in white color > resize the image to 28x28 > save the image and copy the path of image.
4. In the program replace the test image path with the copied path 
im_1 = Image.open(r'test_image_path') 
