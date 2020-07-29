# AI Art 

This is a branch of artificial intelligence in which AI can create paintings, write stories and compose new music. 

Some strategies used include but not limited to 

## Deep Dream: 

Deep dream is the most freaky, creepy AI algorithms to date. The big question is ever wondered what an AI sees?

Deep dream is a computer vision algorithm developed by Alex Mordvintsev at Google. The algorithm works by creating dream-like effect. It is like giving a human an extremely powerful powerful drug. 
As the image is increasingly fed into the network the more weird features will start to pop up. 

Deep dream works by boostnig the patterns it sees in a given image based on what it has been trained to see in the past during training. 

If a network has been trained to see animals in images, it will try to extract animal features in any given image. 


### How does it work? 
- When an image is fed into a trained ANN, the neurons fire and generate activations. 
- The deep dream algorithm work by trying to change the input image in a way that would make some of these neurona fire more (i.e boost the neurons firing or activations). One can select which neurons in which layer you are interested in making them fire prominently.
- This process is continuously reapeated until the input image now contains all features that a specific layer was originally looking for. 
- For instance if a certain layer was experpt in recognize dog faces and one feed in an image of a blue sky, the deep dream algorithm will continously change the input image and start creating images of dogs on top of the blue sky. The process keep repating until the layer of interest in happy with the results!

### Deep Dream Algorithms Steps: 
1. Forward an image through a trained ANN, CNN, ResNet 
2. Select a layer of choice (first layers capture edges, deep layers capture full shapes such as faces) 
3. Calculate the activations (output) coming out from the layer of interest. 
4. Calculate the gradient of the activations with respect to the input image
5. Modify the image to increase these activations, and thus enhanec the patterns seen by the network resulting in trippy hallucinated image! 
6. Iterate and repeat over multiple scales. 
