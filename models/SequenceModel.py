import torch
import torch.nn as nn
import torchvision

class SequenceModel(nn.Module):
    def __init__(self, ch_in=1024):
        super(SequenceModel, self).__init__()
        drop_out = 0.5
        hidden = 96
        lstm_layers = 2
        ratio = 1
        self.ratio=ratio
        
        # seq model 1
        self.fea_conv = nn.Sequential(nn.Dropout2d(drop_out),
                                      nn.Conv2d(ch_in, 512, kernel_size=(1, 1), stride=(1, 1),padding=(0, 0), bias=False),
                                      nn.BatchNorm2d(512),
                                      nn.ReLU(),
                                      nn.Dropout2d(drop_out),
                                      nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(),
                                      nn.Dropout2d(drop_out),
                                      )

        self.fea_first_final = nn.Sequential(nn.Conv2d(128, 6, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True))

        # # bidirectional GRU
        self.hidden_fea = hidden
        self.fea_lstm = nn.GRU(128, self.hidden_fea, num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.fea_lstm_final = nn.Sequential(nn.Conv2d(1, 6, kernel_size=(1, self.hidden_fea*2), stride=(1, 1), padding=(0, 0), dilation=1, bias=True))
        
        
        # seq model 2
        self.conv_first = nn.Sequential(nn.Conv2d(12, 128*ratio, kernel_size=(5, 1), stride=(1,1),padding=(2,0),dilation=1, bias=False),
                                        nn.BatchNorm2d(128*ratio),
                                        nn.ReLU(),
                                        nn.Conv2d(128*ratio, 64*ratio, kernel_size=(3, 1), stride=(1, 1), padding=(2, 0),dilation=2, bias=False),
                                        nn.BatchNorm2d(64*ratio),
                                        nn.ReLU())

        self.conv_res = nn.Sequential(nn.Conv2d(64 * ratio, 64 * ratio, kernel_size=(3, 1), stride=(1, 1),padding=(4, 0),dilation=4, bias=False),
                                      nn.BatchNorm2d(64 * ratio),
                                      nn.ReLU(),
                                      nn.Conv2d(64 * ratio, 64 * ratio, kernel_size=(3, 1), stride=(1, 1),padding=(2, 0),dilation=2, bias=False),
                                      nn.BatchNorm2d(64 * ratio),
                                      nn.ReLU())

        self.conv_final = nn.Sequential(nn.Conv2d(64*ratio, 6, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), dilation=1,bias=False))

        # bidirectional GRU
        self.hidden = hidden
        self.lstm = nn.GRU(64*ratio, self.hidden, num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.final = nn.Sequential(nn.Conv2d(1, 6, kernel_size=(1, self.hidden*2), stride=(1, 1), padding=(0, 0), dilation=1, bias=True))


    def forward(self, features, x):
        
        batch_size, _, _, _ = features.shape
        
        ############### Seq1 ################
        # stem_fc
        x_fc = self.fea_conv(features) # (N, LenFeat, LenSeq, 1)
        
        # fc
        out11 = self.fea_first_final(x_fc) # (N, 6, LenSeq, 1)

        # lstm
        x_lstm, _ = self.fea_lstm(x_fc.reshape(batch_size, -1, 128)) # (N, LenSeq, 192)
        x_lstm = x_lstm.reshape(batch_size, 1, -1, self.hidden_fea*2) # (N, 1, LenSeq, 192)
        
        # fc after lstm
        out12 = self.fea_lstm_final(x_lstm) # (N, 6, LenSeq, 1)
        
        # seq1 output (Elementwise Sum)
        out1 = out11 + out12
        out1_sigmoid = torch.sigmoid(out1) # (N, 6, LenSeq, 1)
        
        # concat cnn out, seq1 out
        x = torch.cat([x, out1], dim=1) # (N, 12, LenSeq, 1)
        
        ############### Seq2 ################
        # stem_fc
        x = self.conv_first(x) # (N, 64, LenSeq, 1)
        x = self.conv_res(x) # (N, 64, LenSeq, 1)
        
        # fc
        out21 = self.conv_final(x) # (N, 6, LenSeq, 1)
        
        # lstm
        x, _ = self.lstm(x.reshape(batch_size, -1, 64)) # (N, LenSeq, 64) => (N, LenSeq, 192)
        x = x.reshape(batch_size, 1, -1, self.hidden*2) # (N, 1, LenSeq, 192)
        # fc after lstm
        out22 = self.final(x) #(N, 6, LenSeq, 1)
        
        # seq2 output (Elementwise Sum)
        out2 = out21 + out22
        out2_sigmoid = torch.sigmoid(out2) # (N, 6, LenSeq, 1)
        
        return out1_sigmoid, out2_sigmoid

