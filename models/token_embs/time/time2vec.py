
import torch
import torch.nn as nn

class Time2Vec(nn.Module):
    def __init__(self, k=32, act="sin", in_feats=6):
        super(Time2Vec, self).__init__()

        if k % 2 == 0:
            k1 = k // 2
            k2 = k // 2
        else:
            k1 = k // 2
            k2 = k // 2 + 1
        
        self.fc1 = nn.Linear(in_feats, k1)

        self.fc2 = nn.Linear(in_feats, k2)
        self.d2 = nn.Dropout(0.3)
 
        if act == 'sin':
            self.activation = torch.sin
        else:
            self.activation = torch.cos

        self.fc3 = nn.Linear(k, k // 2)
        self.d3 = nn.Dropout(0.3)
        
        self.fc4 = nn.Linear(k // 2, in_feats)
        
        self.fc5 = torch.nn.Linear(in_feats, in_feats)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.d2(self.activation(self.fc2(x)))
        out = torch.cat([out1, out2], 1) # This is the embedding

        # Next part is decoder for training
        out = self.d3(self.fc3(out))
        out = self.fc4(out)
        out = self.fc5(out)
        return out

    def encode(self, x):
        x = x.float()
        out1 = self.fc1(x)
        out2 = self.activation(self.fc2(x))
        out = torch.cat([out1, out2], -1)
        return out
    
    def load_model(state_dict_path):
        model = Time2Vec()
        model.load_state_dict(torch.load(state_dict_path, map_location='cpu'))
        return model