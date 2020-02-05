Attention-based Deep Multiple Instance Learning
================================================

This is an experimental implemenation based on this paper "Attention-based Deep Multiple Instance Learning"(https://arxiv.org/pdf/1802.04712.pdf). However, instead of creating a variable-length sequence of instances' features with corresponding bag-level label, this experiment tries to train the neural network with bag-level label without prepared instances, e.g. the model needs to figure out instances' features by itself. 

Based on MNIST dataset, two kinds of MIL-like experiments are implemented 
- With or Without a digit:

  Randomly generate ~50% of bags with specified digit, ~50% of bags without specified digit. 
  Then generate an image for each bag whose containing digits are randomly located without overlapping.  

- Thresholded a digit:

  Randomly generate ~50% of bags with specified digit (more than min_target_count), ~50% of bags without spcified digit or having specified digit less than min_target_count.
  Then generate an image for each bag whole containing digits are randomly located without overlapping.


