# Computer_Vision
This project,that is developed for the Computer Vision course at University of Padova, implements a Human Hand Detection and Segmentation system based on extraction of Histogram of Oriented Gradients features and Support Vector Machines classifier.

There are two options when you execute the code to be inserted through the command line.
When executing the project, you have two options:
- simply test the trained detector, using the default images and ground-truth files provided together with the code.
- execute the complete code, retraining from scratch the model and then automatically testing it.
        
If you choose the first options (recommended), let the value "Use the trained detector" set to y. So simply compile and enter "n" 
when you see the question "Do you want to change the position of directory? (y/n)", all the needed variables will get their default values.

If you decide to retrain the model, enter "y" when you see the previous question and the parameters you should provide in order to make the code run correctly are:
 - "Positive images: "  				--> path of directory of positive images used to train the model,
 - "Negative images: "					--> path of directory of negative images used to train the model,
 - "RGB images: "   					--> path of directory of RGB images,
 - "Mask images: "   					--> path of directory of mask images,
 - "Bounding Box files: "   			--> path of directory of txt file
 - "Use the trained detector (y/n): "   --> test a trained detector (default value is y); SET THIS VALUE TO "n"!!!. 

Fields "Positive images: " and "Negative images: " are not necessary if you use the already trained model, otherwise they will be the paths to you positive and negative samples used for the training.
Also the next 3 parameters is set to a default value, that is the relative path of the testSet directory of this project, containing the images used for training. 
If you want to change this parameter, please be sure to provide the test images, the test masks and the corresponding ground-truth using the same structure adopted in this project.
The structure required for the test set is (the main folder is build): 
 - put your images into the project directory "exam/rgb",
 - put your masks into the project directory "exam/mask",
 - put the ground-truth files into the project directory "exam/det".
 
 The output will be saved in the directory:
 - Metrics result               -->	"exam/results/doc",
 - Human hands detection images --> "exam/results/handBox",
 - Human hands segmentation		  --> "exam/results/handColor".
 
 In order to avoid problems with the test of the detector, I suggest to choose the first option and just try the provided model of the detector, without re-training it. 
 This suggestion is for two main reasons: the time required to train the model is not neglettable, and also, varying the training sets may lead to completely different results with respect to the current ones.
