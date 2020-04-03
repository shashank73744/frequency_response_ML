The model for data  of this size would be required  to run for atleat 10000 epochs, 
due to time constraint I could not run it for so along.

We can run it for so long on AWS platform.

The currently trained model gives the absolute error of 3.9 and mean square error of 2.5.

I checked simple addition also takes about 10000 epoch on just 2 dimentional data for significant learning on NALU.

We can train all the models(NALU, MLP NAC and LIN) but for now I have just trained 
the Multi Level Perceptron because it takes the least time.

The implementation has Multi level perceptron, NALU , NAC and linear models for training.

Instruction to run the file is on Readme.

1. Install the VS code(https://code.visualstudio.com/download) and conda package(https://www.anaconda.com/distribution/#download-section)
2. Install Pytorch and Tensor flow using command line.
3. On the leftmost bottom of the screen there is option to select the python root, there select the conda root.
4. Run the learning file 
5. I have saved the trained model so that the training can start from I left.
6. to start from the begining comment the lines 53 and 76 to 79 (learning.py)
7. You can select model on line 93 betweeb NALU, NAC, MLP and LIN. (earning.py)

Copy "python.linting.pylintArgs": [
        "--generated-members=numpy.* ,torch.*,--errors-only"
        ]
in setting.json without deleting any thing else.