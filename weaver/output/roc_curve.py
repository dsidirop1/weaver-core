import uproot
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load the ROOT file
file_path_mlp = "mlp_predict.root"
file_path_ak8 = "deepak8_predict.root"
file_path_pnet = "particlenet_predict.root"

with uproot.open(file_path_mlp) as root_file_mlp:
    # Replace 'tree_name' and 'branch_name' with the actual names from your file
    tree_mlp = root_file_mlp["Events"]  # Adjust the key based on file structure
    labels_mlp = tree_mlp["is_signal_new"].array()  # True labels
    scores_mlp = tree_mlp["score_is_signal_new"].array()  # Predicted scores or probabilities
with uproot.open(file_path_ak8) as root_file_ak8:
    # Replace 'tree_name' and 'branch_name' with the actual names from your file
    tree_ak8 = root_file_ak8["Events"]  # Adjust the key based on file structure
    labels_ak8 = tree_ak8["is_signal_new"].array()  # True labels
    scores_ak8 = tree_ak8["score_is_signal_new"].array()  # Predicted scores or probabilities
with uproot.open(file_path_pnet) as root_file_pnet:
    # Replace 'tree_name' and 'branch_name' with the actual names from your file
    tree_pnet = root_file_pnet["Events"]  # Adjust the key based on file structure
    labels_pnet = tree_pnet["is_signal_new"].array()  # True labels
    scores_pnet = tree_pnet["score_is_signal_new"].array()  # Predicted scores or probabilities

# Convert data to numpy arrays
labels_mlp = np.array(labels_mlp)
scores_mlp = np.array(scores_mlp)
labels_ak8 = np.array(labels_ak8)
scores_ak8 = np.array(scores_ak8)
labels_pnet = np.array(labels_pnet)
scores_pnet = np.array(scores_pnet)

# Compute ROC curve and AUC
fpr_mlp, tpr_mlp, thresholds_mlp = roc_curve(labels_mlp, scores_mlp)
roc_auc_mlp = auc(fpr_mlp, tpr_mlp)
fpr_ak8, tpr_ak8, thresholds_ak8 = roc_curve(labels_ak8, scores_ak8)
roc_auc_ak8 = auc(fpr_ak8, tpr_ak8)
fpr_pnet, tpr_pnet, thresholds_pnet = roc_curve(labels_pnet, scores_pnet)
roc_auc_pnet = auc(fpr_pnet, tpr_pnet)

# Plot ROC curve
plt.figure()
plt.plot(fpr_mlp, tpr_mlp, color="aqua", lw=2, label=f"ROC curve MultiLayer Perceptron (MLP) (area = {roc_auc_mlp:.2f})")
plt.plot(fpr_ak8, tpr_ak8, color="darkorange", lw=2, label=f"ROC curve DeepAK8 (area = {roc_auc_ak8:.2f})")
plt.plot(fpr_pnet, tpr_pnet, color="gold", lw=2, label=f"ROC curve ParticleNet (area = {roc_auc_pnet:.2f})")
plt.plot([0, 1], [0, 1], color="k", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")

# Save the plot as a .jpg file
plt.savefig("roc_curve_3_models.jpg", format="jpg")

# Show the plot
plt.show()
