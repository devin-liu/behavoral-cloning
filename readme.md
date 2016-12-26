Start with a lambda and normalization layer
First convolutional layer with 8 2x2 filters
Second convolutional layer with 4 2x2 filters
Flatten out the convolutionals in preparation for dense layer
One 640 dimension dense layer
Second and last 1 dimension dense layer


Preprocessing the images definitely helped moreso than adding more layers.
I had a lot of a hangup when I was accidentally cutting out the lane lines.
Now I have cut out the sky and resized the images to 10% of height and width.
The images I used were from the Udacity dataset, so I will exclude information of gathering.


The lambda layer was incredible for normalizing the data, and helping smooth out the steering angle extremeties
I chose 2x2 convolutions because 3x3 and 5x5 seemed to lose too much data on my image size (16x32)

I saw the comma AI model went 16,32,64, increasing the number of filters on later layers.
However, I found better results by starting high and going lower with my filters.
The output of the flattened second convolutional layer outputted 160, but I found a multiple of that was better.
The higher the amount of parameters in the dense layer, the more granular the steering angles.


Mean Squared Error minimized loss below .0012 | Val loss was below .012

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
lambda_1 (Lambda)                (None, 20, 64, 3)     0           lambda_input_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 10, 32, 8)     104         lambda_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 5, 16, 2)      66          convolution2d_1[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 160)           0           convolution2d_2[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 640)           103040      flatten_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 1)             641         dense_1[0][0]

Epoch 1/16
43394/43394 [==============================] - 7s - loss: 0.0150 - val_loss: 0.0127
Epoch 2/16
43394/43394 [==============================] - 5s - loss: 0.0130 - val_loss: 0.0119
Epoch 3/16
43394/43394 [==============================] - 4s - loss: 0.0126 - val_loss: 0.0119
Epoch 4/16
43394/43394 [==============================] - 4s - loss: 0.0124 - val_loss: 0.0121
Epoch 5/16
43394/43394 [==============================] - 4s - loss: 0.0123 - val_loss: 0.0114
Epoch 6/16
43394/43394 [==============================] - 4s - loss: 0.0122 - val_loss: 0.0115
Epoch 7/16
43394/43394 [==============================] - 4s - loss: 0.0121 - val_loss: 0.0114
Epoch 8/16
43394/43394 [==============================] - 4s - loss: 0.0121 - val_loss: 0.0114
Epoch 9/16
43394/43394 [==============================] - 4s - loss: 0.0119 - val_loss: 0.0111
Epoch 10/16
43394/43394 [==============================] - 4s - loss: 0.0118 - val_loss: 0.0112
Epoch 11/16
43394/43394 [==============================] - 4s - loss: 0.0118 - val_loss: 0.0109
Epoch 12/16
43394/43394 [==============================] - 4s - loss: 0.0117 - val_loss: 0.0108
Epoch 13/16
43394/43394 [==============================] - 4s - loss: 0.0116 - val_loss: 0.0108
Epoch 14/16
43394/43394 [==============================] - 4s - loss: 0.0116 - val_loss: 0.0111
Epoch 15/16
43394/43394 [==============================] - 4s - loss: 0.0114 - val_loss: 0.0105
Epoch 16/16
43394/43394 [==============================] - 4s - loss: 0.0113 - val_loss: 0.0104
