#		asmDataset

Here is the master program for Carnegie Mellon University Material Science Department.

Language Using: Python3.5

##		Test&Learning


Floder Test&Learning is the current Testing & Learning methods which may be used in the future programming

##		Workflow

-	Put the code floder under the same directory with the Micrograph floder

-	Initialize.py to get cropped images and their labels

-	>TODO


###	Crop the graphs

Data initialization, cut the graphs into certain shape, e.g. $200\times 200$ pixels for these reasons:

-	Using the valuable data for the further image representation. Remove the pixels that are useless, including scales, black edges, etc.

-	The Micrographs usually have $>400$ pixels shape, by cropping the graphs, we could get more training data for the training system.

>TODO
>
>Questions:
>
>is there any loss of transfering from rgb to gray?

*TEST VERSION: To use the crop.py, run terminal in the data floder, run the crop.py file.*