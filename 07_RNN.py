import torch
input_data = torch.Tensor([ [ [1], [2], [3]] ])
MyRNNModel = torch.nn.RNN(input_size = 1, hidden_size = 3, batch_first = True)

outputs, last_hs = MyRNNModel(input_data)

print(outputs.shape, '\n\n', outputs, '\n')
print(last_hs.shape, '\n\n', last_hs)