# Copyright (c) Puyuan Liu
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ctcdecode import CTCBeamDecoder
import torch
from torch import TensorType
from .ctc_decoder_base import CTCDecoderBase
from typing import Dict, List
#from fairseq.data.ctc_dictionary import CTCDictionary


class CTCBeamSearchDecoder(CTCDecoderBase):
    """
    CTC Beam Search Decoder
    """

    def __init__(self, decoder, decoder_parameters: Dict) -> None:
        super().__init__(decoder)
        self.attn_decoder = decoder 
      
        self.decoder = CTCBeamDecoder(decoder.get_symbols(), model_path= None,
                                      alpha=0, beta=0,
                                      cutoff_top_n=30,
                                      cutoff_prob=1,
                                      beam_width=decoder_parameters["beam"],
                                      num_processes=4,
                                      blank_id=decoder.get_blank_idx(),
                                      log_probs_input=False)  # This is true since our criteria script returns log_prob.
        
    def decode(self, log_prob: TensorType, **kwargs) -> List[List[int]]:
        """
        Decoding function for the CTC beam search decoder.
        """

        '''
        if log_prob.dtype != torch.float16:
            log_prob = log_prob.cpu() 
        '''
        beam_results, beam_scores, timesteps, out_lens = self.decoder.decode(log_prob)
        top_beam_tokens = beam_results[:, 0, :]  # extract the most probable beam
        top_beam_len = out_lens[:, 0]
        mask = torch.arange(0, top_beam_tokens.size(1)).type_as(top_beam_len). \
            repeat(top_beam_len.size(0), 1).lt(top_beam_len.unsqueeze(1))
        top_beam_tokens[~mask] = self.attn_decoder.get_pad_idx()  # mask out nonsense index with pad index.
        #top_beam_tokens = top_beam_tokens.cpu().tolist()
       
        return top_beam_tokens, beam_scores[:,0] # then use truth decoding from decoder to get the results
