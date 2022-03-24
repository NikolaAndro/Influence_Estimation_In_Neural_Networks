# -*- coding: utf-8 -*-
"""
Programmer: Nikola Andric
Date: 3/12/2022
Email: nikolazeljkoandric@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import time, copy
import torch
import torch.nn.functional as F
import random
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp1d
import tensorflow as tf
import sys ; sys.path.append('..')  # useful if you're running locally
import mnist1d
from numpy.random.mtrand import rand

from numpy.random.mtrand import rand
import random
from datetime import datetime

from mnist1d.data import get_templates, get_dataset_args, get_dataset
from mnist1d.train import get_model_args, train_model
from mnist1d.models import ConvBase, GRUBase, MLPBase, LinearBase
from mnist1d.utils import set_seed, plot_signals, ObjectView, from_pickle

# tqdm - package used to shoe a progress bar when loops executing
# tqdm - "progress" in arabic and obriviation for  "Te Quiero DeMaciado" 
# in Spanish (I love you so much)
from tqdm import tqdm 


class ObjectView(object):
    def __init__(self, d): self.__dict__ = d

"""## Attaching GPU if any"""

# Try attaching to GPU
DEVICE = str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
print('Using:', DEVICE)

"""## Download the MNIST-1D dataset"""

args = mnist1d.get_dataset_args()
data = mnist1d.get_dataset(args, path='./mnist1d_data.pkl', download=True) # This is the default setting

print("Examples in training set: {}".format(len(data['y'])))
print("Examples in test set: {}".format(len(data['y_test'])))

print("Length of each input: {}".format(data['x'].shape[-1]))

print("Number of classes: {}".format(len(data['templates']['y'])))

train_data_size = len(data['x'])
test_data_size = len(data['x_test'])

print(train_data_size)
print(test_data_size)

print(list(data))

"""## Initialize Variables"""

train_data_size = len(data['x'])
test_data_size = len(data['x_test'])

"""## Initialize the hyperparameters"""

t = 100 # This should be the number of training images but using just 100 images for the sake of time.
m = int( 0.7 * train_data_size)

"""## Sample t random subsets of [n] of size m: I_1, I_2, I_3..."""

# generate a random subset of indices for the training and test data
random_num_generator = np.random.RandomState(15)

# Generate subset of random indices of size m from (0,train_data_size) without replacement.
random_indices = np.random.choice(train_data_size, size = m, replace = False)

train_images = data["x"][random_indices]

def create_subsets(dataset, t_iterations, m_ratio):
  '''Creates a subset of a dataset with given ratio. Parameter t_iterations - number of iterations / subsets. m_ratio - size of the subset.'''

  random.seed(50)
  # Create a list of subsets
  list_of_subsets = []


  possible_choices = [item for item in range(0, train_data_size)]

  for i in range(t_iterations):
    # Create a subset structure with the same testing data
    subset = {'x':None,'y':None, 'x_test':dataset['x_test'],'y_test':dataset['y_test'],'indices':None}
    
    # Generate subset of random indices of size m from (0,train_data_size) without replacement.
    # random_indices = random_num_generator.choice(train_data_size, size = m, replace = False)
    random_indices = random.choices(possible_choices, k = m)
    
    # Save the random indices in the subset structure
    subset['indices'] = random_indices
    
    # Create a subset of training images and then of testing images / labels
    subset['x'] = dataset["x"][random_indices]
    subset['y'] = dataset["y"][random_indices]

    # Append the subset to the lists of subsets
    list_of_subsets.append(subset)

  return list_of_subsets

#get_model_args(as_dict=True)

"""## Create all necessary subsets"""

subsets = create_subsets(data,t,m) # this should be t instead of 3

"""## Algorithm A"""

# get the model info
args = get_model_args()

# list to keep all the models
list_of_mlp_models = []
list_of_ConvBase_models = []

# list ot keep all the training results
trained_mlp_model_results = []
trained_ConvBase_model_results = []

def train_MLPBase_models(t,subsets,args):
  '''Creates t MLPBase models. Parameter t - number of trials.'''
  # Create and traing t models
  for k in tqdm(range(t)):
    # set the seed
    set_seed(k)

    # create a model
    model = MLPBase(args.input_size, args.output_size)
    
    # append model to the list of models
    list_of_mlp_models.append(model)

    # define the subset of data you want to use
    data_subset = subsets[k]

    # train the model
    mlp_training_results = train_model(data_subset, model, args)

    # append the results of the model
    trained_mlp_model_results.append(mlp_training_results)

  return list_of_mlp_models, trained_mlp_model_results

def train_ConvBase_models(t,subsets,args):
  '''Creates t ConvBase models. Parameter t - number of trials.'''
  # Create and traing t models
  for k in tqdm(range(t)):
    # set the seed
    #set_seed(args.seed)
    set_seed(k)

    # create a model
    model = ConvBase(output_size=args.output_size)
    
    # append model to the list of models
    list_of_ConvBase_models.append(model)

    # define the subset of data you want to use
    data_subset = subsets[k]

    # train the model
    ConvBase_training_results = train_model(data_subset, model, args)

    # append the results of the model
    trained_ConvBase_model_results.append(ConvBase_training_results)

  return list_of_ConvBase_models, trained_ConvBase_model_results

"""## Train the models on the subsets"""

# MLP base
list_of_mlp_models, trained_mlp_model_results = train_MLPBase_models(t,subsets,args)

# ConvBase
list_of_ConvBase_models, trained_ConvBase_model_results = train_ConvBase_models(t,subsets,args)


"""# Memorization"""

from mnist1d import train

def estimate_mem_infl(list_of_models, subsets, x_i_index):
  '''Computes memorization and influence estimates for a single x_i from the 
  dataset. 
  Parameters: list of all models and their results as well as the subset index 
  in the list of the subsets, the list of subsets, and the index of an x in 
  the training set for which we are looking to find memorization for.'''

  # list of all subset indices where the x_i is present
  # trainset_mask = np.zeros(train_data_size, dtype=np.bool)
  x_in_subsets= []
  x_not_in_subsets = []

  # Check in which subsets is the x_i present
  for sub_index, subset in enumerate(subsets):
    if x_i_index in subset['indices']:
      # Append the index to the list of indices for the chosen subsets 
      x_in_subsets.append(sub_index)
    else:
      # Append the index of a subset where x is NOT present
      x_not_in_subsets.append(sub_index)

  x_in_picked_models = []
  x_not_in_picked_models = []

  # print(len(x_not_in_subsets))
  # Take the models that are trained based on those subsets where x is present
  for index in x_in_subsets:
    x_in_picked_models.append(list_of_models[index])

  # Take the models that are trained based on those subsets where x_i is NOT present
  for index in x_not_in_subsets:
    x_not_in_picked_models.append(list_of_models[index])

  # picked_models = [model for index,model in enumerate(list_of_models) if index] 

  # Get the image and its label
  # # Convert splits into the proper forms
  # x_train, x_test = torch.Tensor(subsets[subset_index]['x']), torch.Tensor(subsets[subset_index]['x_test'])
  # y_train, y_test = torch.LongTensor(subsets[subset_index]['y']), torch.LongTensor(subsets[subset_index]['y_test'])
  x_train, x_test = torch.Tensor(data['x']), torch.Tensor(data['x_test'])
  y_train, y_test = torch.LongTensor(data['y']), torch.LongTensor(data['y_test'])

  input_image = x_train[x_i_index]
  image_label = y_train[x_i_index]

  # Count correct predictions
  num_correct_pred = 0

  # Check how many of these models are not making the error in classification
  for model in x_in_picked_models:
    
    prediction = model(input_image).argmax(-1).cpu().numpy()
    target = image_label.cpu().numpy().astype(np.float32)

    # Check if the predicion is the same as the label
    # If so, increment the count of correct rpedictions
    if prediction == target:
      num_correct_pred += 1
  
  # Take the fraction of the number of models that are predicting correctly
  # over the total number of models.
  # The result is the first probability in the equation .

  p_x_i_in = num_correct_pred / len(x_in_subsets)




  # Count correct predictions
  num_correct_pred = 0

  # Check how many of these models are not making the error in classification
  for model in x_not_in_picked_models:
    
    prediction = model(input_image).argmax(-1).cpu().numpy()
    target = image_label.cpu().numpy().astype(np.float32)

    # Check if the predicion is the same as the label
    # If so, increment the count of correct rpedictions
    if prediction == target:
      num_correct_pred += 1
  
  # Take the fraction of the number of models that are predicting correctly
  # over the total number of models.
  # The result is the first probability in the equation .

  p_x_i_not_in = num_correct_pred / len(x_not_in_subsets)


  memorization_estimate_x_i = p_x_i_in - p_x_i_not_in

  #*************** Influence of x_i on every j in test data *****************

  #Create a list for influence of x_i on each x_j
  x_i_inluence_list = []

  # # Check the influence by the models created based on datasets where x_i is present
  # # for j in range(np.shape(x_test)[0]):
  # for j in range(np.shape(x_test)[0]):
  #   # Take a single image from the testing part of the dataset
  #   input_image = x_test[j]
  #   image_label = y_test[j]

  #   # Count correct predictions
  #   num_correct_pred_x_in = 0
  #   num_correct_pred_x_not_in = 0

  #   # check how each model performs from the models where x_i is present
  #   for modelo in x_in_picked_models:  
  #       prediction = modelo(input_image).argmax(-1).cpu().numpy()
  #       target = image_label.cpu().numpy().astype(np.float32)

  #       # Check if the predicion is the same as the label
  #       # If so, increment the count of correct rpedictions
  #       if prediction == target:
  #         num_correct_pred_x_in += 1
    
  #   #Calculate the influence of x_i on the x_j from the test data
  #   infl_x_i_in_s_on_x_j = num_correct_pred_x_in / np.shape(x_test)[0]

  #   # check how each model performs from the models where x_i is NOT present
  #   for modelo in x_not_in_picked_models:  
  #       prediction = modelo(input_image).argmax(-1).cpu().numpy()
  #       target = image_label.cpu().numpy().astype(np.float32)

  #       # Check if the predicion is the same as the label
  #       # If so, increment the count of correct rpedictions
  #       if prediction == target:
  #         num_correct_pred_x_not_in += 1

  #   #Calculate the influence of x_i on the x_j from the test data
  #   infl_x_i_not_in_s_on_x_j = num_correct_pred_x_not_in / np.shape(x_test)[0]

  #   # Append the influence of x_i on every x_j from the test data
  #   x_i_inluence_list.append(infl_x_i_not_in_s_on_x_j)


  return memorization_estimate_x_i, x_i_inluence_list


# keep a list of all memorization values for histogram
mem_values_mlp = []
infl_values_mlp = []

# For every x_i in the training dataset find its memorization value
# and list of influences on each x_j from the testing dataset.
for x_i in tqdm(range(t)): #np.shape(data['x'])[0])
  # print()  
  mem, infl = estimate_mem_infl(list_of_mlp_models,subsets,x_i )

  # print("\nmemorization value for x_i is" , mem )
  # print("\n Its influence on every point in test data is: ", infl)

  # save the memorization in the list of mem values
  mem_values_mlp.append(mem)

  # save the influence in the list of infl_values_mlp
  # infl_values_mlp.append(infl)

mem_values_conv_base = []
infl_values_conv_base = []

# DO the same for ConvBase Model
for x_i in tqdm(range(t)): #np.shape(data['x'])[0])
  # print()  
  mem, infl = estimate_mem_infl(list_of_ConvBase_models ,subsets,x_i )

  # print("\nmemorization value for x_i is" , mem )
  # print("\n Its influence on every point in test data is: ", infl)

  # save the memorization in the list of mem values
  mem_values_conv_base.append(mem)

  # save the influence in the list of  infl_values_conv_base
  # infl_values_conv_base.append(infl)

"""# Histogram of Memorization MLP Model"""

# Fixing random state for reproducibility
np.random.seed(19680801)
# the histogram of the data
n, bins, patches = plt.hist(mem_values_mlp, 20, facecolor='g', alpha=0.75)

plt.title('Memorizaztion Values Distribution')
plt.xlabel('Memorizaztion Values')
plt.ylabel('Freequency')
plt.xlim(-1, 1)
plt.savefig("MLP_model_histogram.png")
plt.show()


"""# Histogram of insfluence of X0 on the test data samples (MLP)"""

# Fixing random state for reproducibility
# np.random.seed(19680801)
# # the histogram of the data
# n, bins, patches = plt.hist(infl_values_mlp[0], 20, facecolor='g', alpha=0.75)

# plt.title('Influence Values Distribution MLP')
# plt.xlabel('Influence Values')
# plt.ylabel('Freequency')
# plt.xlim(-1, 1)
# plt.show()
# plt.savefig("MLP_model_histogram_influence.png")

"""# Histogram of Memorization ConvBase Model"""

# Fixing random state for reproducibility
np.random.seed(19680801)
# the histogram of the data
n, bins, patches = plt.hist(mem_values_conv_base, 20, facecolor='g', alpha=0.75)

plt.title('Memorizaztion Values Distribution')
plt.xlabel('Memorizaztion Values')
plt.ylabel('Freequency')
plt.xlim(-1, 1)
plt.savefig("ConvBase_model_histogram.png")
plt.show()


"""# Histogram of insfluence of X0 on the test data samples (ConvBase)"""

# Fixing random state for reproducibility
# np.random.seed(19680801)
# # the histogram of the data
# n, bins, patches = plt.hist(infl_values_conv_base[0], 20, facecolor='g', alpha=0.75)

# plt.title('Influence Values Distribution Conv Base')
# plt.xlabel('Influence Values')
# plt.ylabel('Freequency')
# plt.xlim(-1, 1)
# plt.show()
# plt.savefig("ConvBase_model_histogram_influence.png")

"""# Creatign a new dataset with more samples with a high memorization"""

# transformations of the templates which will make them harder to fit
def pad(x, padding):
    low, high = padding
    p = low + int(np.random.rand()*(high-low+1))
    return np.concatenate([x, np.zeros((p))])

def shear(x, scale=10):
    coeff = scale*(np.random.rand() - 0.5)
    return x - coeff*np.linspace(-0.5,.5,len(x))

def translate(x, max_translation):
    k = np.random.choice(max_translation)
    return np.concatenate([x[-k:], x[:-k]])

def corr_noise_like(x, scale):
    noise = scale * np.random.randn(*x.shape)
    return gaussian_filter(noise, 2)

def iid_noise_like(x, scale):
    noise = scale * np.random.randn(*x.shape)
    return noise

def interpolate(x, N):
    scale = np.linspace(0,1,len(x))
    new_scale = np.linspace(0,1,N)
    new_x = interp1d(scale, x, axis=0, kind='linear')(new_scale)
    return new_x

def transform(x, y, args, eps=1e-8):
    new_x = pad(x+eps, args.padding) # pad
    new_x = interpolate(new_x, args.template_len + args.padding[-1])  # dilate
    new_y = interpolate(y, args.template_len + args.padding[-1])
    new_x *= (1 + args.scale_coeff*(np.random.rand() - 0.5))  # scale
    new_x = translate(new_x, args.max_translation)  #translate
    
    # add noise
    mask = new_x != 0
    new_x = mask*new_x + (1-mask)*corr_noise_like(new_x, args.corr_noise_scale)
    new_x = new_x + iid_noise_like(new_x, args.iid_noise_scale)
    
    # shear and interpolate
    new_x = shear(new_x, args.shear_scale)
    new_x = interpolate(new_x, args.final_seq_length) # subsample
    new_y = interpolate(new_y, args.final_seq_length)
    return new_x, new_y

def get_dataset_args(as_dict=False):
    arg_dict = {'num_samples': 5000,
            'train_split': 0.8,
            'template_len': 12,
            'padding': [36,60],
            'scale_coeff': .4, 
            'max_translation': 48,
            'corr_noise_scale': 0.25,
            'iid_noise_scale': 2e-2,
            'shear_scale': 0.75,
            'shuffle_seq': False,
            'final_seq_length': 40,
            'seed': 42}
    return arg_dict if as_dict else ObjectView(arg_dict)

def apply_ablations(arg_dict, n=7):
    ablations = [('shear_scale', 0),
                ('iid_noise_scale', 0),
                ('corr_noise_scale', 0),
                 ('max_translation', 1),
                 ('scale_coeff', 0),
                 ('padding', [arg_dict['padding'][-1], arg_dict['padding'][-1]]),
                 ('padding', [0, 0]),]
    num_ablations = min(n, len(ablations))
    for i in range(num_ablations):
        k, v = ablations[i]
        arg_dict[k] = v
    return arg_dict

templates = get_templates()
for i, n in enumerate(reversed(range(8))):
    np.random.seed(0)
    arg_dict = get_dataset_args(as_dict=True)
    arg_dict = apply_ablations(arg_dict, n=n)
    args = ObjectView(arg_dict)
    do_transform = args.padding[0] != 0
    # fig = plot_signals(templates['x'], templates['t'], labels=None if do_transform else templates['y'],
    #              args=args, ratio=2.2 if do_transform else 0.8,
    #              do_transform=do_transform)
#     fig.savefig(PROJECT_DIR + 'static/transform_{}.png'.format(i))

def make_dataset(args=None, template=None, ):
    templates = get_templates() if template is None else template
    args = get_dataset_args() if args is None else args
    np.random.seed(args.seed) # reproducibility
    
    xs, ys = [], []
    samples_per_class = args.num_samples // len(templates['y'])
    for label_ix in range(len(templates['y'])):
        for example_ix in range(samples_per_class):
            x = templates['x'][label_ix]
            t = templates['t']
            y = templates['y'][label_ix]
            x, new_t = transform(x, t, args) # new_t transformation is same each time
            xs.append(x) ; ys.append(y)
    
    batch_shuffle = np.random.permutation(len(ys)) # shuffle batch dimension
    xs = np.stack(xs)[batch_shuffle]
    ys = np.stack(ys)[batch_shuffle]
    
    if args.shuffle_seq: # maybe shuffle the spatial dimension
        seq_shuffle = np.random.permutation(args.final_seq_length)
        xs = xs[...,seq_shuffle]
    
    new_t = new_t/xs.std()
    xs = (xs-xs.mean())/xs.std() # center the dataset & set standard deviation to 1

    # train / test split
    split_ix = int(len(ys)*args.train_split)
    dataset = {'x': xs[:split_ix], 'x_test': xs[split_ix:],
               'y': ys[:split_ix], 'y_test': ys[split_ix:],
               't':new_t, 'templates': templates}
    return dataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

args = get_dataset_args()
set_seed(args.seed)
args.corr_noise_scale = 0.5
args.iid_noise_scale = 2e-1
args.max_translation = 58 # Change these values to get more examples with a high
                          # memorization value
data = make_dataset(args=args)    # make the dataset

"""# Apply the estimates function on the new dataset"""

# generate a random subset of indices for the training and test data
random_num_generator = np.random.RandomState(15)

# Generate subset of random indices of size m from (0,train_data_size) without replacement.
random_indices = np.random.choice(train_data_size, size = m, replace = False)

print(len(random_indices))

train_images = data["x"][random_indices]
print(np.shape(train_images))

subsets = create_subsets(data,t,m)


# get the model info
args = get_model_args()

# list to keep all the models
list_of_mlp_models = []
list_of_ConvBase_models = []

# list ot keep all the training results
trained_mlp_model_results = []
trained_ConvBase_model_results = []


# MLP base
list_of_mlp_models, trained_mlp_model_results = train_MLPBase_models(t,subsets,args)

# ConvBase
list_of_ConvBase_models, trained_ConvBase_model_results = train_ConvBase_models(t,subsets,args)


# keep a list of all memorization values for histogram
mem_values_mlp_modified = []
infl_values_mlp_modified = []

# For every x_i in the training dataset find its memorization value
# and list of influences on each x_j from the testing dataset.
for x_i in tqdm(range(t)): 
  mem, infl = estimate_mem_infl(list_of_mlp_models,subsets,x_i )

  # save the memorization in the list of mem values
  mem_values_mlp_modified.append(mem)

  # save the influence in the list of infl_values_mlp_modified
  # infl_values_mlp_modified.append(infl)


mem_values_conv_base_modified = []
infl_values_conv_base_modified = []

# DO the same for ConvBase Model
for x_i in tqdm(range(t)): 
  mem, infl = estimate_mem_infl(list_of_ConvBase_models ,subsets,x_i )

  # save the memorization in the list of mem values
  mem_values_conv_base_modified.append(mem)

  # save the influence in the list of infl_values_conv_base_modified
  # infl_values_conv_base_modified.append(infl)

"""# Histogram of Memorization MLP Model on Modified Dataset

# Histogram of Memorization MLP Model
"""

# Fixing random state for reproducibility
np.random.seed(19680801)
# the histogram of the data
n, bins, patches = plt.hist(mem_values_mlp_modified, 20, facecolor='g', alpha=0.75)

plt.title('Memorizaztion Values Distribution')
plt.xlabel('Memorizaztion Values')
plt.ylabel('Freequency')
plt.xlim(-1, 1)
plt.savefig("MLP_model_histogram_modified.png")
plt.show()


"""# Histogram of insfluence of X0 on the test data samples (MLP)"""

# Fixing random state for reproducibility
# np.random.seed(19680801)
# # the histogram of the data
# n, bins, patches = plt.hist(infl_values_mlp_modified[0], 20, facecolor='g', alpha=0.75)

# plt.title('Influence Values Distribution MLP')
# plt.xlabel('Influence Values')
# plt.ylabel('Freequency')
# plt.xlim(-1, 1)
# plt.show()
# plt.savefig("MLP_model_histogram_influence_modified.png")

"""# Histogram of Memorization ConvBase Model"""

# Fixing random state for reproducibility
np.random.seed(19680801)
# the histogram of the data
n, bins, patches = plt.hist(mem_values_conv_base_modified, 20, facecolor='g', alpha=0.75)

plt.title('Memorizaztion Values Distribution')
plt.xlabel('Memorizaztion Values')
plt.ylabel('Freequency')
plt.xlim(-1, 1)
plt.savefig("ConvBase_model_histogram_modified.png")
plt.show()


"""# Histogram of insfluence of X0 on the test data samples (ConvBase)"""

# # Fixing random state for reproducibility
# np.random.seed(19680801)
# # the histogram of the data
# n, bins, patches = plt.hist(infl_values_conv_base_modified[0], 20, facecolor='g', alpha=0.75)

# plt.title('Influence Values Distribution Conv Base')
# plt.xlabel('Influence Values')
# plt.ylabel('Freequency')
# plt.xlim(-1, 1)
# plt.show()
# plt.savefig("ConvBase_model_histogram_influence_modified.png")

