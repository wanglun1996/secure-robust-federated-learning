import torch
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

AGGR_MEAN = 'mean'
AGGR_GEO_MED = 'geom_median'
AGGR_FOOLSGOLD='foolsgold'
AGGR_KRUM = 'krum'
AGGR_TRIIMMEDMEAN = 'trimmedmean'
AGGR_BULYAN_KRUM = 'bulyan_krum'
AGGR_BULYAN_TRIM = 'bulyan_trimmed_mean'
AGGR_BULYAN_MEDIAN = 'bulyan_median'
AGGR_FILTERL2 = 'filterl2'
AGGR_EXNOREGRET = 'ex_noregret'
AGGR_MEDIAN = 'median'
AGGR_CLUSTERING = 'clustering'
AGGR_HISTORY = 'history'
AGGR_BUCKETING = 'bucketing'
MAX_UPDATE_NORM = 1000
patience_iter=20

TYPE_LOAN='loan'
TYPE_CIFAR='cifar'
TYPE_MNIST='mnist'
TYPE_FASHION='fashion-mnist'
TYPE_TINYIMAGENET='tiny-imagenet-200'
