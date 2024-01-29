from torch import nn
import torch
from typing import Dict, Literal, Optional
from outputs import OutputClassification, OutputEmbedding


class RNN(nn.Module):
    def __init__(
            self, 
            input_size, 
            output_size, 
            hidden_dim, 
            n_layers, 
            model_type: Literal['rnn', 'lstm', 'gru'], 
            bidirectional=False, 
            dropout_prob=0, 
            input_mapping_fn=lambda input_size, input_mapping_size: nn.Identity(),
            input_mapping_size=None):
        super().__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        if input_mapping_size is None:
            input_mapping_size = input_size
        self.input_mapping_size = input_mapping_size
        self.input_mapping = input_mapping_fn(input_size, input_mapping_size)


        #Defining the layers
        # RNN Layer
        if model_type == 'rnn':
            self.rnn = nn.RNN(input_mapping_size, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectional)
        elif model_type == 'lstm':
            self.rnn = nn.LSTM(input_mapping_size, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectional)
        elif model_type == 'gru':
            self.rnn = nn.GRU(input_mapping_size, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectional)
        else:
            raise NotImplementedError(model_type)

        # Fully connected layer
        nb_directions = 1
        if bidirectional:
            nb_directions = 2

        self.linear_input_size = hidden_dim * nb_directions
        self.fc = nn.Linear(self.linear_input_size, output_size)
        self.bidirectional = bidirectional
        self.model_type = model_type

        if dropout_prob > 0:
            self.dropout = nn.Dropout(p=dropout_prob)
        else:
            self.dropout = None
    
    def forward(self, x):
        
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size, x.device)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, _ = self.rnn(x, hidden)
        assert out.shape[0] == batch_size
        assert out.shape[1] == x.shape[1]
        assert out.shape[2] == self.linear_input_size
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(out.shape[0], -1, self.linear_input_size)
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.fc(out)
        
        return out
    
    def init_hidden(self, batch_size, device):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        nb_directions = 1
        if self.bidirectional:
            nb_directions = 2
        
        if self.model_type == 'lstm':
            hidden = torch.zeros(self.n_layers * nb_directions, batch_size, self.hidden_dim, device=device), \
                     torch.zeros(self.n_layers * nb_directions, batch_size, self.hidden_dim, device=device)
        else:    
            hidden = torch.zeros(self.n_layers * nb_directions, batch_size, self.hidden_dim, device=device)
        return hidden


class ModelLstm(nn.Module):
    def __init__(
            self,
            rnn: nn.Module = RNN(input_size=32, output_size=2, hidden_dim=128, n_layers=1, model_type='rnn', bidirectional=False), 
            weight_volume_name: Optional[str] = None,
            weight_false_positive: Optional[float] = None) -> None:
        super().__init__()
        self.rnn = rnn
        self.weight_volume_name = weight_volume_name
        self.weight_false_positive = weight_false_positive

    def forward(self, batch):
        sequence_input = batch['sequence_input']  # expecting (1, Sequence, features) shape
        sequence_output = batch['sequence_output']
        assert len(sequence_input) == len(sequence_output)
        assert len(sequence_input.shape) == 3
        assert sequence_input.shape[0] == 1, 'single sample at once!'
        assert sequence_output.shape[0] == 1, 'single sample at once!'

        o_raw = self.rnn(sequence_input)
        o = o_raw.transpose(1, 2)  # swap the <C, X> dim for the cross entropy
        o_raw = o_raw.squeeze(0)
        sequence_output = sequence_output.squeeze(0)

        weighting = None
        if self.weight_volume_name is not None:
            weighting = batch.get(self.weight_volume_name)
            if weighting is None and self.training:
                raise RuntimeError('if training and weighing specified, it should be here!')
            if weighting is not None:
                weighting = weighting.squeeze(0)
        if weighting is None:
            weighting = torch.ones([o_raw.shape[0]], dtype=torch.float32, device=o_raw.device)

        if self.weight_false_positive is not None:
            output_found = o_raw.argmax(dim=1)
            error_indices = torch.where(output_found != sequence_output)
            fp_indices_indices = torch.where(sequence_output[error_indices] == 0)
            weighting_fp = torch.ones([o_raw.shape[0]], dtype=torch.float32, device=o_raw.device)
            weighting_fp[error_indices[0][fp_indices_indices[0]]] = self.weight_false_positive

            # combine the multiple weights
            weighting = weighting * weighting_fp

        output_classification = OutputClassification(o_raw, sequence_output.unsqueeze(1), weights=weighting)
        #criterion = nn.CrossEntropyLoss(reduction='none')
        #loss = criterion(o, sequence_output)

        # here recreate the segmentation map to be compatible with
        # existing validation code
        suv_shape = batch['suv'].shape
        seg_map = torch.zeros([o.shape[0], 2] + list(suv_shape), dtype=torch.float32)
        seg_map[0, 0][:] = 1.0  # by default, background
        sequence_label = batch['sequence_label']
        assert o.shape[0] == 1, 'TODO handle batch size > 1'
        assert o.shape[1] == 2, 'Binary classification'
        for n in range(o.shape[2]):
            segmentation_id = n + 1

            segmentation_reclassification = o[0, :, n]
            assert segmentation_reclassification.shape == (2,)
            indices = torch.where(sequence_label == segmentation_id)
            seg_map[0, 0][indices] = segmentation_reclassification[0]
            seg_map[0, 1][indices] = segmentation_reclassification[1]

        # do not collect continuously the segmentations, too large! `clean_loss_term_each_batch`
        # will force removing this input
        seg_map_output = OutputEmbedding(seg_map, clean_loss_term_each_batch=True)

        return {
            #'sequence': OutputLoss(loss.mean(dim=1)),
            'sequence': output_classification,
            'seg': seg_map_output
        }
