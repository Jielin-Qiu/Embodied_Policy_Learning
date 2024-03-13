from torch import nn
import torch
from transformers import EncoderDecoderModel
import numpy as np
import torch.nn.functional as F


class APM(nn.Module):
    def __init__(self, llm_model, args):
        super().__init__()
        self.llm_model = llm_model
        self.args = args

    def forward(self, decoder_out, exec_input_ids, state):
        

        output = self.llm_model(inputs_embeds = decoder_out, labels = exec_input_ids, output_hidden_states = True)

        decoder_hidden = output.decoder_hidden_states[-1]
        loss= output.loss
        if self.args.learning == 'reinforce':
            x = state_to_tensor(state)
            x = F.relu(self.layers(x))
            actions = F.softmax(self.layers(x))
            action = self.get_action(actions)
            log_prob_action = torch.log(actions.squeeze(0))[action]
            return decoder_hidden, loss, x, log_prob_action
        else:
            return decoder_hidden, loss

    def llm_generate(self, inputs_embeds):
        output_ids = self.llm_model.generate(inputs_embeds = inputs_embeds)
        return output_ids
    
def state_to_tensor(self, I):
    if I is None:
        return torch.zeros(1, 6000)
    I = I[35:185] 
    I = I[::2,::2,0] 
    I[I == 144] = 0 
    I[I == 109] = 0 
    I[I != 0] = 1 
    return torch.from_numpy(I.astype(np.float32).ravel()).unsqueeze(0)

def get_action(self,a):
    return np.random.choice([0, 1],p=a.squeeze(0).detach().cpu().numpy())