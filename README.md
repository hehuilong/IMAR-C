IMAR-C
======

IMAR-C (Intermedia Action Recognition in C/C++ language) is a set of action recognition tools allowing the robot NAO recognize the human actions.
It contains:
NAO Vision Teacher: generate vision learning model for the robot.
visionAction: can be implemented on the robot to recognize human actions through his camera.
recordVideoLocal: can be implemented on the robot to capture video.
argvToSpeech: can send the words to the robot and make it speek.
decreaseSize: decrease the size of video.
detectFps: detect the fps of video.

## Activities
For the moment the activities had to be recorded in a 2-second video (our video analysis programm is too long):

	- applaud
	- fall
	- walk
	- sit down
	- stand up
	- drink
	- remote control 
	- open / close a door
	- write	 
	- phone call

## Sources:
* SVM: [http://www.csie.ntu.edu.tw/~cjlin/libsvm/](http://www.csie.ntu.edu.tw/~cjlin/libsvm/):
Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for support
vector machines. ACM Transactions on Intelligent Systems and
Technology, 2:27:1--27:27, 2011. Software available at
http://www.csie.ntu.edu.tw/~cjlin/libsvm

* KMeans: [http://www.cs.umd.edu/~mount/Projects/KMeans/](http://www.cs.umd.edu/~mount/Projects/KMeans/);
* Feature points: [http://lear.inrialpes.fr/people/wang/dense_trajectories](http://lear.inrialpes.fr/people/wang/dense_trajectories);
* NAO sofwares: [https://developer.aldebaran-robotics.com/home/](https://developer.aldebaran-robotics.com/home/);

## Licence:
This work is under the free software licence [CeCILL](http://www.cecill.info/).
For more information open the LICENCE file.

## Members:
HE Huilong and ROUALDES Fabien (Institut Mines-Télécom)
