# SquatFormTracker-FYP
### Installation Guide:
These are the steps taken on Ubuntu LTS 16.04, other OSs may vary.
First thing to do is install TensorFlow preferably on the GPU. There's a good step by step for that here: https://pythonprogramming.net/how-to-cuda-gpu-tensorflow-deep-learning-tutorial/
You're going to need to download the tar.gz file for SSDMobileNet_V1 this can be found on the TensorFlow GitHub.
The next thing to do is change all paths to the appropriate for your machine. Non-Relative filepaths occur in both gymObjectDetector and run. 
Replace the contents of the data dictionary with your videos.
Attempt to run using 'python run.py'. As the import errors occur remedy them with a simple pip install command. This may not work for openCV3 but there's a guid for this here : https://milq.github.io/install-opencv-ubuntu-debian/
