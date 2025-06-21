import pickle
with open("client_checkpoints_mydataset_FedAvg.pkl", "rb") as f:
    checkpoint_dict = pickle.load(f)
print(checkpoint_dict)  # See what clients/rounds are stored