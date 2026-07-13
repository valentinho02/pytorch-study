import torch
import torch.nn as nn


input_size = 10   
hidden_size = 20  
num_layers = 2  

lstm_model = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

gru_model = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

dummy_input = torch.randn(3, 5, 10)

lstm_out, (lstm_h, lstm_c) = lstm_model(dummy_input)
gru_out, gru_h = gru_model(dummy_input)

print(f"LSTM 출력 형태: {lstm_out.shape}") 
print(f"LSTM 최종 Cell State 형태: {lstm_c.shape}")
print(f"GRU 출력 형태: {gru_out.shape}")