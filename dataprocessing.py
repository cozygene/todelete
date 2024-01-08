#%%
# import libraries
import numpy as np
# load npz file
save_model_name = 'ResNet18Scratch'
# save_model_name = 'ViVitPretrain'
# save_model_name = 'ViVitPretrain_dim96_depth6'
# /scratch/pterway/slivit/SLIViT/ViVitPretrain_dim192_depth6.pth
# save_model_name = 'ResNet50Pretrained'
# save_model_name = 'ResNet50Scratch'
# save_model_name = 'ResNet18Pretrain'

# /scratch/pterway/slivit/SLIViT/npzfiles/ViVitPretrain.npz
# /scratch/pterway/slivit/SLIViT/ViVitPretrain_dim192_depth6.pth
# /scratch/pterway/slivit/SLIViT/npzfiles/ViVitPretrain_dim192_depth6.npz
file_path = '/scratch/pterway/slivit/SLIViT/npzfiles/' + save_model_name + '.npz'

loaded_data = np.load(file_path)
auc_scores = loaded_data['auc_scores']
auprc_scores = loaded_data['auprc_scores']

#%%
# Assuming auc_scores is a 1D array
mean_auc = np.mean(auc_scores)
confidence_interval = np.percentile(auc_scores, [2.5, 97.5])

print("Mean:", mean_auc)
print("Confidence Interval (95%):", confidence_interval)

#%%
mean_auprc = np.mean(auprc_scores)
confidence_interval_auprc = np.percentile(auprc_scores, [2.5, 97.5])
print("Mean:", mean_auprc)
print("Confidence Interval (95%):", confidence_interval_auprc)

# %%
########## ViVIT Results ############
# Results
# ViVit: Baseline dim 192, dept 6
# AUC -ROC
## Mean: 0.8703225553817059 
## Confidence Interval (95%): [0.83944522 0.90199878]

# AUPRC
# Mean: 0.7041933018476877
# Confidence Interval (95%): [0.62599976 0.76967464]

#%%
# ViVit: Baseline dim 192, dept 4
# AUC -ROC
# Mean: 0.8520298060131951
# Confidence Interval (95%): [0.8142657  0.88909096]
# AUPRC
# Mean: 0.6600450932820014
# Confidence Interval (95%): [0.56500555 0.7524684 ]
#%%
# ViVit: Baseline dim 96, dept 6
# AUC -ROC
# Mean: 0.8666121388176173
# Confidence Interval (95%): [0.83934102 0.89531751]

# AUPRC
# Mean: 0.6375189270377107
# Confidence Interval (95%): [0.54832883 0.71803863]

#%%
############ ResNEt Results ############
# ResNet50Pretrained
# AUC -ROC
# Mean: 0.8407921469846122
# Confidence Interval (95%): [0.78903153 0.88574899]
# # AUPRC
# Mean: 0.6733130660004149
# Confidence Interval (95%): [0.59199371 0.73541596]

#%%
# ResNet50Scratch
# AUC -ROC
# Mean: 0.8642979144688815
# Confidence Interval (95%): [0.82884401 0.90031046]
# # AUPRC
# Mean: 0.6949362573892506
# Confidence Interval (95%): [0.63300392 0.76507247]
#%%
# ResNet18Pretrained
# AUC -ROC
# Mean: 0.7899879011884248
# Confidence Interval (95%): [0.75344867 0.84287318]
# # AUPRC
# Mean: 0.6116994126239735
# Confidence Interval (95%): [0.54030853 0.69405856]
#%%
# ResNet18Scratch
# AUC -ROC
# Mean: 0.8399573425071125
# Confidence Interval (95%): [0.80103328 0.87604376]
# # AUPRC
# Mean: 0.6766833640638478
# Confidence Interval (95%): [0.60333322 0.73888967]