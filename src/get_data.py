import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

#******************************************************************************
# Read in data
#******************************************************************************
def getData(name, train_bs=128, test_bs=256):

    if name == 'doublegyre4':
       D = torch.load('datasets/doublegyre/doublegyre_train.npy') 
       train_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=True)
       D = torch.load('datasets/doublegyre/doublegyre_test_1.npy') 
       test1_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('datasets/doublegyre/doublegyre_val_1.npy') 
       val1_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('datasets/doublegyre/doublegyre_test_2.npy') 
       test2_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('datasets/doublegyre/doublegyre_val_2.npy') 
       val2_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=False)
       return train_loader, test1_loader, val1_loader, test2_loader, val2_loader   

    elif name == 'doublegyre8':
       D = torch.load('datasets/doublegyre/doublegyre_train.npy') 
       train_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=True)
       D = torch.load('datasets/doublegyre/doublegyre_test_1.npy') 
       test1_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('datasets/doublegyre/doublegyre_val_1.npy') 
       val1_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('datasets/doublegyre/doublegyre_test_2.npy') 
       test2_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('datasets/doublegyre/doublegyre_val_2.npy') 
       val2_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=False)
       return train_loader, test1_loader, val1_loader, test2_loader, val2_loader     

    elif name == 'isoflow4':
       D = torch.load('datasets/isoflow/isoflow_train.npy') 
       train_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=True)
       D = torch.load('datasets/isoflow/isoflow_test_1.npy') 
       test1_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('datasets/isoflow/isoflow_val_1.npy') 
       val1_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('datasets/isoflow/isoflow_test_2.npy') 
       test2_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('datasets/isoflow/isoflow_val_2.npy') 
       val2_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=False)
       return train_loader, test1_loader, val1_loader, test2_loader, val2_loader     
 

    elif name == 'isoflow8':
       D = torch.load('datasets/isoflow/isoflow_train.npy') 
       train_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=True)
       D = torch.load('datasets/isoflow/isoflow_test_1.npy') 
       test1_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('datasets/isoflow/isoflow_val_1.npy') 
       val1_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('datasets/isoflow/isoflow_test_2.npy') 
       test2_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('datasets/isoflow/isoflow_val_2.npy') 
       val2_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=False)
       return train_loader, test1_loader, val1_loader, test2_loader, val2_loader     
 

    elif name == 'rbc4':
       D = torch.load('datasets/rbc/rbc_train.npy') 
       train_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=True)
       D = torch.load('datasets/rbc/rbc_test_1.npy') 
       test1_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('datasets/rbc/rbc_val_1.npy') 
       val1_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('datasets/rbc/rbc_test_2.npy') 
       test2_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('datasets/rbc/rbc_val_2.npy') 
       val2_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=False)
       return train_loader, test1_loader, val1_loader, test2_loader, val2_loader   

    elif name == 'rbc8':
       D = torch.load('datasets/rbc/rbc_train.npy') 
       train_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=True)
       D = torch.load('datasets/rbc/rbc_test_1.npy') 
       test1_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('datasets/rbc/rbc_val_1.npy') 
       val1_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('datasets/rbc/rbc_test_2.npy') 
       test2_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('datasets/rbc/rbc_val_2.npy') 
       val2_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=False)
       return train_loader, test1_loader, val1_loader, test2_loader, val2_loader     
 
    elif name == 'rbcsc4':
       D = torch.load('datasets/rbcsc/rbc_train.npy') 
       train_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=True)
       D = torch.load('datasets/rbcsc/rbc_test_1.npy') 
       test1_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('datasets/rbcsc/rbc_val_1.npy') 
       val1_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('datasets/rbcsc/rbc_test_2.npy') 
       test2_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('datasets/rbcsc/rbc_val_2.npy') 
       val2_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=False)
       return train_loader, test1_loader, val1_loader, test2_loader, val2_loader   

    elif name == 'rbcsc8':
       D = torch.load('datasets/rbcsc/rbc_train.npy') 
       train_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=True)
       D = torch.load('datasets/rbcsc/rbc_test_1.npy') 
       test1_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('datasets/rbcsc/rbc_val_1.npy') 
       val1_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('datasets/rbcsc/rbc_test_2.npy') 
       test2_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('datasets/rbcsc/rbc_val_2.npy') 
       val2_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=False)
       return train_loader, test1_loader, val1_loader, test2_loader, val2_loader      
 
    
    elif name == 'sst4':
       D = torch.load('datasets/sst/sst_train.npy') 
       train_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=True)
       D = torch.load('datasets/sst/sst_test_1.npy') 
       test1_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('datasets/sst/sst_val_1.npy') 
       val1_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('datasets/sst/sst_test_2.npy') 
       test2_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('datasets/sst/sst_val_2.npy') 
       val2_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=False)
       return train_loader, test1_loader, val1_loader, test2_loader, val2_loader   

    elif name == 'sst8':
       D = torch.load('datasets/sst/sst_train.npy') 
       train_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=True)
       D = torch.load('datasets/sst/sst_test_1.npy') 
       test1_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('datasets/sst/sst_val_1.npy') 
       val1_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('datasets/sst/sst_test_2.npy') 
       test2_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('datasets/sst/sst_val_2.npy') 
       val2_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=False)
       return train_loader, test1_loader, val1_loader, test2_loader, val2_loader                  

    else:
        raise ValueError('dataset {} not recognized'.format(name))






    
    