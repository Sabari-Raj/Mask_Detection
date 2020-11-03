# Mask_Detection
Detect masked images using Pytorch

I used the dataset in Kaggle https://www.kaggle.com/muhammeddalkran/masked-facerecognition (RMFRD).I used 3000 images from the dataset for training and testing, 1400 where masked and the rest were unmasked dataset. As the dataset was highly unbalanced I reduced the amount of unmasked dataset. I created a list of lists with the image and label in the nested list where the label indicates whether the images was Masked(1) or Unmasked(0) and images was in numpy array form.

A lot of effort in solving any machine learning problem goes in to preparing the data. PyTorch provides many tools to make data loading easy and hopefully. torch.utils.data.Dataset is an abstract class representing a dataset. Our custom dataset should inherit Dataset and override the following methods:

__len__ so that len(dataset) returns the size of the dataset.
__getitem__ to support the indexing such that dataset[i] can be used to get ith sample
Most neural networks expect the images of a fixed size. Therefore, there is a need to write some prepocessing code in the MaskDataset class. I applied four tranformations, convert to PIL then resizing, random horizonatal flip, convert to tensor and then normalization. 

Then we split our data into training, test and validation set. Now a Dataset object loads training or test data into memory, and a DataLoader object fetches data from a Dataset and serves the data up in batches.
PyTorch's DataLoader class, which in addition to our Dataset class, also takes in the following important arguments:

BATCH_SIZE:      Which denotes the number of samples contained in each generated batch.
SHUFFLE:         If set to True, we will get a new order of exploration at each pass (or just keep a linear exploration scheme otherwise). Shuffling the order in which examples are fed to the classifier is helpful so that batches between epochs do not look alike. Doing so will eventually make our model more robust.
NUM_WORKERS:     Which denotes the number of processes that generate batches in parallel. A high enough number of workers assures that CPU computations are efficiently managed, i.e. that the bottleneck is indeed the neural network's forward and backward operations on the GPU (and not data generation).

Then we create the model using nn.Sequential and pass our dataset through it.We get an acuracy of 96.25

