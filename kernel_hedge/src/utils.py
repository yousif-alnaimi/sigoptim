import torch

def batches_to_path(batches):
    """
    input: in batches of size [num_batches, path_length, dim]
    
    output: one continuous path of size [num_batches*(path_length-1), dim]
    """

    path = torch.cat([torch.ones(1,1), torch.diff(batches,dim=1).flatten(end_dim=1).cumsum(0)+1],dim=0)

    return path