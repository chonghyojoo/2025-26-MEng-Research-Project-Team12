#%%
import inference_api as api

#%%
# Same with your trained model info
class Args:
    def __init__(self):
        self.hidden_size = 200             # D-MPNN hidden dim
        self.ffn_hidden_size = 100         # Feedforward hidden dim
        self.output_size = 2               # # of properties
        self.dropout = 0.1                 # 
        self.bias = True
        self.depth = 2                     
        self.activation = "ReLU"           # activation
        self.cuda = True                   # GPU
        self.property = "solvation"
        self.aggregation = "mean"
        self.atomMessage = False           # False: only atom

#%%
# load
model = api.load_model("best_model.pt", args= Args())                  
dataset = api.load_dataset("preprocessing/processed_binarysolv_exp.pkl") # testdataset

# %%
# example1: single data point
y0 = api.predict_single(model, dataset[1])
# example2: batch data
batch_size = 8
yb = api.predict_batch(model, [dataset[i] for i in range(batch_size)])

#%%
print("single: G_solv298 = ", y0[:,0])
print("single: H_solv298 = ", y0[:,1])
print("batch : G_solv298 = ", yb[:,0])
print("batch : H_solv298 = ", yb[:,1])
# %%

