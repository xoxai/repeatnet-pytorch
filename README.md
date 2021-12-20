# What's here

Here is a little bit modernized implementation of RepeatNet from orginal paper. It includes data preprocessing step and a little style enhancements.

# Running the training process
To run training process on YooChoose dataset, just do following in source root:

`torchrun ./Run.py --mode='train'`

You can see in terminal something like:

```
Torch version: 1.10.1
Data size: 952
init item_emb.weight torch.Size([52739, 128])
init enc.weight_ih_l0 torch.Size([192, 128])
init enc.weight_hh_l0 torch.Size([192, 64])
init enc.bias_ih_l0 torch.Size([192])
init enc.bias_hh_l0 torch.Size([192])
init enc.weight_ih_l0_reverse torch.Size([192, 128])
init enc.weight_hh_l0_reverse torch.Size([192, 64])
init enc.bias_ih_l0_reverse torch.Size([192])
init enc.bias_hh_l0_reverse torch.Size([192])
init mode_attn.linear_key.weight torch.Size([128, 128])
init mode_attn.linear_query.weight torch.Size([128, 128])
init mode_attn.linear_query.bias torch.Size([128])
init mode_attn.v.weight torch.Size([1, 128])
init mode.weight torch.Size([2, 128])
init mode.bias torch.Size([2])
init repeat_attn.linear_key.weight torch.Size([128, 128])
init repeat_attn.linear_query.weight torch.Size([128, 128])
init repeat_attn.linear_query.bias torch.Size([128])
init repeat_attn.v.weight torch.Size([1, 128])
init explore_attn.linear_key.weight torch.Size([128, 128])
init explore_attn.linear_query.weight torch.Size([128, 128])
init explore_attn.linear_query.bias torch.Size([128])
init explore_attn.v.weight torch.Size([1, 128])
init explore.weight torch.Size([52739, 128])
init explore.bias torch.Size([52739])
Method train Epoch 0 Batch  1 Loss  [7.986617088317871] Time  27.70906114578247
Method train Epoch 1 Batch  1 Loss  [7.9715495109558105] Time  8.815345048904419
Method train Epoch 2 Batch  1 Loss  [7.954436302185059] Time  9.516870021820068
Method train Epoch 3 Batch  1 Loss  [7.938322067260742] Time  9.64490294456482
Method train Epoch 4 Batch  1 Loss  [7.924196243286133] Time  9.166892051696777
Method train Epoch 5 Batch  1 Loss  [7.906176567077637] Time  9.503098011016846
Method train Epoch 6 Batch  1 Loss  [7.890024185180664] Time  9.123106002807617
Method train Epoch 7 Batch  1 Loss  [7.869630813598633] Time  9.68349814414978
Method train Epoch 8 Batch  1 Loss  [7.855503082275391] Time  19.01116394996643
Method train Epoch 9 Batch  1 Loss  [7.839035987854004] Time  7.93090295791626
Method train Epoch 10 Batch  1 Loss  [7.816061019897461] Time  8.038913011550903
```

The first part of logs described built Neural Network structure. The second part is about training and losses on each of epoches.