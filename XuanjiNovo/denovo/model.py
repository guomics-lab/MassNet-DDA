"""A de novo peptide sequencing model with configurable logging."""

import threading
import logging
import re
import sys
import operator
import os
from datetime import datetime
import torch.nn.functional as F
import random
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import depthcharge.masses
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.tensorboard import SummaryWriter
#from torch_imputer.imputer import best_alignment
from . import pmc  # Precise Mass Control module
from ..components import ModelMixin, PeptideDecoder, SpectrumEncoder
from .ctc_beam_search import CTCBeamSearchDecoder
from . import evaluate

# Configure logging
def setup_logger(verbosity: str = "INFO") -> logging.Logger:
    """
    Set up the logger with the specified verbosity level.
    
    Parameters
    ----------
    verbosity : str
        The logging level to use. One of: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        Default is "INFO".
    
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a unique log file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"xuanjinovo_{timestamp}.log")
    
    # Set up logging format
    log_format = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Map string level to logging constant
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    level = level_map.get(verbosity.upper(), logging.INFO)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger("xuanjinovo")
    logger.setLevel(level)
    
    return logger

# Initialize default logger
logger = setup_logger()
aa2mas = { 'G': 57.021464, 'A': 71.037114, 'S': 87.032028, 'P': 97.052764, 'V': 99.068414, 'T': 101.04767, 'C+57.021': 160.030649, 'L': 113.084064, 'I': 113.084064, 'N': 114.042927, 'D': 115.026943, 'Q': 128.058578, 'K': 128.094963, 'E': 129.042593, 'M': 131.040485, 'H': 137.058912, 'F': 147.068414, 'R': 156.101111, 'Y': 163.063329, 'W': 186.079313, 'M+15.995': 147.0354, 'N+0.984': 115.026943, 'Q+0.984': 129.042594, '+42.011': 42.010565, '+43.006': 43.005814, '-17.027': 100000, '+43.006-17.027': 25.980265, "_":0}

logger = logging.getLogger("casanovo")
def mass_cal(sequence):
    sequence = sequence.replace("I", "L")
    sequence = re.split(r"(?<=.)(?=[A-Z])", sequence)
    total = 0
    for each in sequence:
        
        # total += aa2mas[each]
        
        try:
            total += aa2mas[each]
        except:
            h1 = each.count("+42.011")
            h2 = each.count("+43.006")
            h3 = each.count("-17.027")
            total += h1 * 42.010565 + h2 * 43.005814  + h3 * -17.026549
            each = each.replace("+42.011", "")
            each = each.replace("+43.006", "")
            each = each.replace("-17.027", "")
            if each:
                total += aa2mas[each]
                    
    return total , sequence
def remove_repentance(index_list: List[int]) -> List[int]:
        """
        Eliminate repeated index in list. e.g., [1, 1, 2, 2, 3] --> [1, 2, 3]
        """
        return [a for a, b in zip(index_list, index_list[1:] + [not index_list[-1]]) if a != b]
    
def ctc_post_processing( sentence_index: List[int]) -> List[int]:
        """
        Merge repetitive tokens, then eliminate <blank> tokens and <pad> tokens.
        The input sentence_index is expected to be a 1-D index list
        """
        sentence_index = remove_repentance(sentence_index)
        #sentence_index = list(filter((27).__ne__, sentence_index))
        temp = []
        for each in sentence_index:
            if each != 27:
                temp.append(each)
        return temp

class Spec2Pep(pl.LightningModule, ModelMixin):
    """
    A Transformer model for de novo peptide sequencing.

    Use this model in conjunction with a pytorch-lightning Trainer.

    Parameters
    ----------
    dim_model : int
        The latent dimensionality used by the transformer model.
    n_head : int
        The number of attention heads in each layer. ``dim_model`` must be
        divisible by ``n_head``.
    dim_feedforward : int
        The dimensionality of the fully connected layers in the transformer
        model.
    n_layers : int
        The number of transformer layers.
    dropout : float
        The dropout probability for all layers.
    dim_intensity : Optional[int]
        The number of features to use for encoding peak intensity. The remaining
        (``dim_model - dim_intensity``) are reserved for encoding the m/z value.
        If ``None``, the intensity will be projected up to ``dim_model`` using a
        linear layer, then summed with the m/z encoding for each peak.
    custom_encoder : Optional[Union[SpectrumEncoder, PairedSpectrumEncoder]]
        A pretrained encoder to use. The ``dim_model`` of the encoder must be
        the same as that specified by the ``dim_model`` parameter here.
    max_length : int
        The maximum peptide length to decode.
    residues: Union[Dict[str, float], str]
        The amino acid dictionary and their masses. By default ("canonical) this
        is only the 20 canonical amino acids, with cysteine carbamidomethylated.
        If "massivekb", this dictionary will include the modifications found in
        MassIVE-KB. Additionally, a dictionary can be used to specify a custom
        collection of amino acids and masses.
    max_charge : int
        The maximum precursor charge to consider.
    precursor_mass_tol : float, optional
        The maximum allowable precursor mass tolerance (in ppm) for correct
        predictions.
    isotope_error_range : Tuple[int, int]
        Take into account the error introduced by choosing a non-monoisotopic
        peak for fragmentation by not penalizing predicted precursor m/z's that
        fit the specified isotope error:
        `abs(calc_mz - (precursor_mz - isotope * 1.00335 / precursor_charge))
        < precursor_mass_tol`
    n_beams: int
        Number of beams used during beam search decoding.
    n_log : int
        The number of epochs to wait between logging messages.
    tb_summarywriter: Optional[str]
        Folder path to record performance metrics during training. If ``None``,
        don't use a ``SummaryWriter``.
    warmup_iters: int
        The number of warm up iterations for the learning rate scheduler.
    max_iters: int
        The total number of iterations for the learning rate scheduler.
    out_writer: Optional[str]
        The output writer for the prediction results.
    **kwargs : Dict
        Additional keyword arguments passed to the Adam optimizer.
    """

    def __init__(
        self,
        mass_control_tol = 0.1,
        PMC_enable = True,
        dim_model: int = 512,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 9,
        dropout: float = 0.0,
        dim_intensity: Optional[int] = None,
        custom_encoder: Optional[SpectrumEncoder] = None,
        max_length: int = 100,
        residues: Union[Dict[str, float], str] = "canonical",
        max_charge: int = 5,
        precursor_mass_tol: float = 50,
        isotope_error_range: Tuple[int, int] = (0, 1),
        n_beams: int = 5,
        n_log: int = 10,
        tb_summarywriter: Optional[
            torch.utils.tensorboard.SummaryWriter] = None,
        warmup_iters: int = 100_000,
        max_iters: int = 600_000,
        out_writer : str = None,
        ctc_dic: dict = {},
        refine_iters: int = 3,
        log_level: str = "INFO",
        mask_schedule: Dict[str, float] = {
            "initial_peek": 0.93,
            "epoch_decay": 0.01,
            "min_peek": 0.00
        },
        result_output_dir: str = None,
        **kwargs: Dict,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Configure logger
        self.model_logger = setup_logger(log_level)  # Renamed to avoid conflict with pl.LightningModule's logger
        self.model_logger.debug("Initializing Spec2Pep model")
        
        self.ctc_dic = ctc_dic
        self.PMC_enable = PMC_enable
        self.model_logger.info(f"PMC enabled: {PMC_enable}, Mass control tolerance: {mass_control_tol}")
        self.mass_control_tol = mass_control_tol
        self.ctc_dic["beam"] = n_beams
        
        
        # Build the model.
        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            self.encoder = SpectrumEncoder(
                dim_model=dim_model,
                n_head=n_head,
                dim_feedforward=dim_feedforward,
                n_layers=n_layers,
                dropout=dropout,
                dim_intensity=dim_intensity,
            )
        self.decoder = PeptideDecoder(
            dim_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            residues=residues,
            max_charge=max_charge,
            max_pep_len = max_length

        )
        self.n_layers = n_layers
        
        self.ctc_decoder = CTCBeamSearchDecoder(self.decoder, self.ctc_dic)
        self.calctime = 0.0
        #print("ctc_decoder:", self.ctc_decoder)
        '''
        decoding_params = {
                'force_length': getattr(args, 'force_length'),
                "desired_length": getattr(args, 'desired_length'),
                "use_length_ratio": getattr(args, 'use_length_ratio'),
                "k": getattr(args, 'k'),
                "beam_size": getattr(args, 'beam_size'),
                "scope": getattr(args, 'scope'),
                "marg_criteria": getattr(args, 'marg_criteria', 'max'),
                # truncate_summary is a dummy variable since length control does not need it
                "truncate_summary": getattr(args, 'truncate_summary'),
                "scaling_factor": getattr(args, 'bucket_size'),
            }
        '''
        #self.ctc_length_control_decoder = CTCScopeSearchCharLengthControlDecoder(self.decoder, {} )
        self.softmax = torch.nn.Softmax(2)
        self.celoss = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.ctcloss = torch.nn.CTCLoss(blank = self.decoder.get_blank_idx(), zero_infinity=True) #to do
        # Optimizer settings.
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.opt_kwargs = kwargs

        # Data properties.
        self.max_length = max_length
        self.residues = residues
        self.precursor_mass_tol = precursor_mass_tol
        self.isotope_error_range = isotope_error_range
        self.n_beams = n_beams
        self.peptide_mass_calculator = depthcharge.masses.PeptideMass(
            self.residues)
        #self.stop_token = self.decoder._aa2idx["$"] ## need to change 

        # Logging.
        self.n_log = n_log
        self._history = []
        if tb_summarywriter is not None:
            self.tb_summarywriter = SummaryWriter(tb_summarywriter)
        else:
            self.tb_summarywriter = tb_summarywriter

        # Output writer during predicting.
        self.out_writer = out_writer
        self.result_output_dir = result_output_dir
    

    def forward(
            self, spectra: torch.Tensor,
            precursors: torch.Tensor, true_peps) -> Tuple[List[List[str]], torch.Tensor]:
        """
        Predict peptide sequences for a batch of MS/MS spectra.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra for which to predict peptide sequences.
            Axis 0 represents an MS/MS spectrum, axis 1 contains the peaks in
            the MS/MS spectrum, and axis 2 is essentially a 2-tuple specifying
            the m/z-intensity pair for each peak. These should be zero-padded,
            such that all of the spectra in the batch are the same length.
        precursors : torch.Tensor of size (n_spectra, 3)
            The measured precursor mass (axis 0), precursor charge (axis 1), and
            precursor m/z (axis 2) of each MS/MS spectrum.

        Returns
        -------
        peptides : List[List[str]]
            The predicted peptide sequences for each spectrum.
        aa_scores : torch.Tensor of shape (n_spectra, length, n_amino_acids)
            The individual amino acid scores for each prediction.
        """
        self.model_logger.debug(f"Forward pass started with batch size: {spectra.shape[0]}")
        start_time = datetime.now()
        prev = None 
        for i in range(self.hparams.refine_iters):
            output_logits, _, output_list = self.decoder(None, precursors, *self.encoder(spectra, precursors), prev)
            prev = output_logits.argmax(-1)
        
        top_tokens, beamscores = self.ctc_decoder.decode(F.softmax(output_logits, -1))
        batchscores = beamscores.tolist()
        batchscores = 1 / torch.exp(beamscores)
        top_tokens_beam = top_tokens.tolist()
        
        batch_size = output_logits.shape[0]
        self.mass_offset_total = 0
        self.mass_offset_count = 0
        top_tokens = [[] for i in range(batch_size)]

        def worker(logits, mass, i, output_logits):
            mass = mass.clone().detach()
            if not self.PMC_enable:
                top_tokens[i] = top_tokens_beam[i]
                return
                
            mass_true = mass[0].item() - 18.01
            sequence = list(filter((self.decoder.get_pad_idx()).__ne__, top_tokens_beam[i]))
            token_true = [self.decoder._idx2aa[each] for each in sequence]
            
           
            
            pred_mass, seq = mass_cal("".join(token_true))
            if abs(mass_true - pred_mass) < 0.1:
                top_tokens[i] = top_tokens_beam[i]
            else:
                temp = pmc.knapDecode(logits, mass, self.mass_control_tol)
                temp = ctc_post_processing(temp)
                if temp:
                    top_tokens[i] = temp
                    token_temp = [self.decoder._idx2aa[each] for each in temp]
                    token_temp = list(reversed(token_temp))
                else:
                    top_tokens[i] = top_tokens_beam[i]
        threads = []
        log_prob = F.log_softmax(output_logits, -1)
        for i in range(batch_size):
            t = threading.Thread(target=worker, args=(log_prob[[i], :, :], precursors[[i], 0], i, output_logits))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        self.model_logger.debug(f"Forward pass completed in {processing_time:.2f} seconds")
        
        # Log prediction statistics
        peptides = [self.decoder.detokenize_truth(t, True) for t in top_tokens]
        avg_peptide_length = sum(len(p) for p in peptides) / len(peptides)
        self.model_logger.info(f"Average predicted peptide length: {avg_peptide_length:.2f}")
        
        return peptides, batchscores

    def _forward_step(
        self,
        spectra: torch.Tensor,
        precursors: torch.Tensor,
        sequences: List[str],
        prev = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward learning step.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra for which to predict peptide sequences.
            Axis 0 represents an MS/MS spectrum, axis 1 contains the peaks in
            the MS/MS spectrum, and axis 2 is essentially a 2-tuple specifying
            the m/z-intensity pair for each peak. These should be zero-padded,
            such that all of the spectra in the batch are the same length.
        precursors : torch.Tensor of size (n_spectra, 3)
            The measured precursor mass (axis 0), precursor charge (axis 1), and
            precursor m/z (axis 2) of each MS/MS spectrum.
        sequences : List[str] of length n_spectra
            The partial peptide sequences to predict.

        Returns
        -------
        scores : torch.Tensor of shape (n_spectra, length, n_amino_acids)
            The individual amino acid scores for each prediction.
        tokens : torch.Tensor of shape (n_spectra, length)
            The predicted tokens for each spectrum.
        """
        return self.decoder(sequences, precursors, *self.encoder(spectra, precursors), prev)

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, List[str]],
        *args,
        mode: str = "train",
    ) -> torch.Tensor:
        """
        A single training step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, List[str]]
            A batch of (i) MS/MS spectra, (ii) precursor information, (iii)
            peptide sequences as torch Tensors.
        mode : str
            Logging key to describe the current stage.

        Returns
        -------
        torch.Tensor
            The loss of the training step.
        """
        self.model_logger.debug(f"Starting {mode} step with batch size: {len(batch[0])}")
        step_start_time = datetime.now()
        """
        A single training step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, List[str]]
            A batch of (i) MS/MS spectra, (ii) precursor information, (iii)
            peptide sequences as torch Tensors.
        mode : str
            Logging key to describe the current stage.

        Returns
        -------
        torch.Tensor
            The loss of the training step.

        """
        glat_prev = None
        
        # Calculate peek factor based on mask schedule
        peek_factor = max(
            self.hparams.mask_schedule["initial_peek"] - 
            self.current_epoch * self.hparams.mask_schedule["epoch_decay"],
            self.hparams.mask_schedule["min_peek"]
        )
        
        if mode == "train":
            with torch.no_grad():
                word_ins_out, tgt_tokens, _ = self._forward_step(*batch)
                nonpad_positions = tgt_tokens.ne(self.decoder.get_pad_idx())
                target_lens = (nonpad_positions).sum(1)
                pred_tokens = word_ins_out.argmax(-1)
                out_lprobs = F.log_softmax(word_ins_out, dim=-1)
                seq_lens = torch.full(size=(pred_tokens.size()[0],), fill_value=pred_tokens.size()[1]).to(self.device)
                best_aligns = best_alignment(out_lprobs.transpose(0, 1), tgt_tokens, seq_lens, target_lens, self.decoder.get_blank_idx(),
                                            zero_infinity=True)
                best_aligns_pad = torch.tensor([a  for a in best_aligns],
                                            device=word_ins_out.device)
                oracle_pos = (best_aligns_pad // 2).clip(max=tgt_tokens.shape[1] - 1)
                oracle = tgt_tokens.gather(-1, oracle_pos)
                oracle_empty = oracle.masked_fill(best_aligns_pad % 2 == 0, self.decoder.get_blank_idx())
                same_num = ((pred_tokens == oracle_empty)).sum(1)
                keep_prob = ((seq_lens - same_num) / seq_lens * peek_factor ).unsqueeze(-1)
                keep_word_mask = (torch.rand(pred_tokens.shape, device=word_ins_out.device) < keep_prob).bool()
                glat_prev = oracle_empty.masked_fill(~keep_word_mask, self.decoder.get_mask_idx())
        
        total_loss = torch.tensor([0]).to(self.device)
        
        for i in range(2):
            pred, truth, output_list = self._forward_step(*batch, glat_prev)
            pred_temp = pred.permute(1, 0, 2)
            
            input_lengths = torch.full(size=(pred_temp.size()[1],), fill_value=pred_temp.size()[0]) 
            target_lengths = (truth != self.decoder.get_pad_idx()).sum(axis=1) 
            loss = self.ctcloss( torch.nn.functional.log_softmax(pred_temp, dim=-1), truth, input_lengths, target_lengths ) 
            glat_prev = pred.argmax(-1)
            total_loss = total_loss + loss
        
        assert len(output_list) == self.n_layers 
        
        tokens = torch.argmax(pred, axis=2)
        peptides_pred = []
        peptides_true = []
        for idx in range(tokens.size()[0]):
            tokens_true = truth[idx,:]
            tokens_true = self.decoder.detokenize_truth(tokens_true)
            peptides_true.append(''.join(tokens_true))

            tokens_pred = tokens[idx,:]
            tokens_pred = self.decoder.detokenize(tokens_pred)
            peptides_pred.append(tokens_pred)

        aa_precision, aa_recall, pep_recall = evaluate.aa_match_metrics(
            *evaluate.aa_match_batch(peptides_pred, peptides_true,
                                     self.decoder._peptide_mass.masses))
        
        rand = random.random()
        sampling_factor = 0.3 if mode == "train" else 0.8
        
        if rand < sampling_factor:
            peptides_pred_sample = [''.join(tokenlist) for tokenlist in peptides_pred]
            peptides_pair_list = list(zip(batch[1].cpu().numpy().tolist(), peptides_true, peptides_pred_sample))
            peptides_pair = random.choices(peptides_pair_list, k=15)
            if self.logger is not None:
                self.logger.experiment[mode+"/peptides_pair"].append("Epoch: "+str(self.trainer.current_epoch)+str(peptides_pair))
        
        log_args = dict(on_step=True, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
        if mode == "train":
            self.log("train/aa_precision", aa_precision, **log_args)
            self.log("train/aa_recall", aa_recall, **log_args)
            self.log("train/pep_recall", pep_recall, **log_args)
        elif mode == "valid" and self.n_beams == 0:
            log_args.update(on_step=False)
            self.log("valid/aa_precision", aa_precision, **log_args)
            self.log("valid/aa_recall", aa_recall, **log_args)
            self.log("valid/pep_recall", pep_recall, **log_args)
        elif mode == "test" and self.n_beams == 0:
            log_args.update(on_step=False)
            self.log("test/aa_precision", aa_precision, **log_args)
            self.log("test/aa_recall", aa_recall, **log_args)
            self.log("test/pep_recall", pep_recall, **log_args)
        
        
        
        if (mode == "train"):
            if(not self.logger==None):
                self.logger.experiment["train_CELoss_step"].append(loss)
            self.log(
                "train/CELoss",
                loss.detach(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                add_dataloader_idx=False
            )

        step_end_time = datetime.now()
        step_duration = (step_end_time - step_start_time).total_seconds()
        
        # Log detailed metrics
        self.model_logger.debug(f"{mode} step completed in {step_duration:.2f} seconds")
        self.model_logger.debug(f"Total loss: {total_loss.item():.4f}")
        
        if mode == "train":
            self.model_logger.debug(f"Current peek factor: {peek_factor:.4f}")
            self.model_logger.debug(f"Amino acid precision: {aa_precision:.4f}")
            self.model_logger.debug(f"Amino acid recall: {aa_recall:.4f}")
            self.model_logger.debug(f"Peptide recall: {pep_recall:.4f}")
        
        return total_loss

    def validation_step(self, batch, batch_idx=None, dataloader_idx=None) -> torch.Tensor:
        """
        A single validation step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, List[str]]
            A batch of (i) MS/MS spectra, (ii) precursor information, (iii)
            peptide sequences.
        batch_idx : Optional[int]
            The index of the current batch
        dataloader_idx : Optional[int]
            The index of the dataloader

        Returns
        -------
        torch.Tensor
            The loss of the validation step.
        """
        val_start_time = datetime.now()
        self.model_logger.debug(f"Starting validation step {batch_idx if batch_idx is not None else 'N/A'}")
        if dataloader_idx is None:
            dataloader_idx = 0
        key = "valid" if dataloader_idx == 0 else "test"
        
        loss = self.training_step(batch, mode=key)
        self.log(
            "valid/CELoss" if dataloader_idx == 0 else "test/CELoss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            add_dataloader_idx=False
        )

        if self.n_beams > 0:
            peptides_pred_raw, inferscores = self.forward(batch[0], batch[1], batch[2])
            peptides_pred, peptides_true = [], []
            
            # Process predictions
            for peptide_pred, peptide_true in zip(peptides_pred_raw, batch[2]):
                if len(peptide_pred) > 0:
                    if peptide_pred[0] == "$":
                        peptide_pred = peptide_pred[1:]  # Remove stop token
                    if "$" not in peptide_pred and len(peptide_pred) > 0:
                        peptides_pred.append(peptide_pred)
                        peptides_true.append(peptide_true)
            
            # Calculate per-amino-acid precision
            targets = ["M+15.995", "F", "Q", "K"]
            counts = {char: {"numChar": 1e-13, "numTrue": 0} for char in targets}

            for pred, true in zip(peptides_pred, peptides_true):
                if isinstance(pred, str):
                    pred = re.split(r"(?<=.)(?=[A-Z])", pred)
                if isinstance(true, str):
                    true = re.split(r"(?<=.)(?=[A-Z])", true)

                for j, c in enumerate(true):
                    if c in counts:
                        counts[c]["numChar"] += 1
                        if j < len(pred) and pred[j] == c:
                            counts[c]["numTrue"] += 1

            # Log per-amino-acid precision
            for char, data in counts.items():
                precision = data["numTrue"] / data["numChar"]
                self.log(
                    f"{key}/{char}_precision",
                    precision,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    add_dataloader_idx=False
                )
            
            # Calculate overall metrics
            aa_precision, aa_recall, pep_recall = evaluate.aa_match_metrics(
                *evaluate.aa_match_batch(peptides_pred, peptides_true,
                                       self.decoder._peptide_mass.masses))
            
            # Log overall metrics
            log_args = dict(on_step=True, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
            self.log(f"{key}/aa_precision", aa_precision, **log_args)
            self.log(f"{key}/aa_recall", aa_recall, **log_args)
            self.log(f"{key}/pep_recall", pep_recall, **log_args)
            
            # Validation metrics are already logged through self.log
        
        val_end_time = datetime.now()
        val_duration = (val_end_time - val_start_time).total_seconds()
        
        # Log validation metrics
        self.model_logger.debug(f"Validation step completed in {val_duration:.2f} seconds")
        self.model_logger.debug(f"Validation loss: {loss.item():.4f}")
        
        if self.n_beams > 0:
            self.model_logger.debug(f"Validation metrics for {key}:")
            self.model_logger.debug(f"AA precision: {aa_precision:.4f}")
            self.model_logger.debug(f"AA recall: {aa_recall:.4f}")
            self.model_logger.debug(f"Peptide recall: {pep_recall:.4f}")
        
        return loss

    def predict_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], *args
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]], torch.Tensor]:
        """
        A single prediction step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A batch of (i) MS/MS spectra, (ii) precursor information, (iii)
            spectrum identifiers as torch Tensors.

        Returns
        -------
        spectrum_idx : torch.Tensor
            The spectrum identifiers.
        precursors : torch.Tensor
            Precursor information for each spectrum.
        peptides : List[List[str]]
            The predicted peptide sequences for each spectrum.
        aa_scores : torch.Tensor of shape (n_spectra, length, n_amino_acids)
            The individual amino acid scores for each prediction.
        """
        peptides , inferscores = self.forward(batch[0], batch[1], batch[2])
        import os
        
        file_path = os.path.join(self.result_output_dir, "denovo.tsv")
        headers = "title\tprediction\tcharge\tscore\n"

        # Check if the file exists and whether it contains headers
        if not os.path.exists(file_path) or open(file_path, 'r').readline().strip() != headers.strip():
            with open(file_path, 'a') as f:
                f.write(headers)

        # Append data
        with open(file_path,'a') as f:
            for i in range(len(peptides)):
                sequence = ""
                for el in peptides[i]:
                    if len(el) > 1:
                        if el[0] in ('+', '-'):
                            sequence += '[' + el + ']'
                        elif el[0].isdigit():
                            sequence += '[' + el + ']'
                        else:
                            sequence += el[0] + '[' + el[1:] + ']'
                    else:
                        sequence += el

                # print("label:",batch[2][i], ":" , peptides[i] , "\n")
                # if batch[2][i].replace("$", "").replace("N+0.984", "D").replace("Q+0.984", "E").replace("L","I") == "".join(peptides[i]).replace("$", "").replace("N+0.984", "D").replace("Q+0.984", "E").replace("L","I"):
                #     answer_is_correct = "correct"
                # else:
                #     answer_is_correct = "incorrect"
                #each line output this: label (title if label is none), predictions, charge, and confidence score
                f.write(batch[2][i].replace("\t", " ") + "\t" + sequence + "\t" + str(int(batch[1][i][1])) + "\t" + str(float(inferscores[i])) + "\n")
                
                
        
        return batch[2], batch[1], peptides  #batch[2]: identifier

    def on_validation_epoch_end(self) -> None:
        """
        Log the validation metrics at the end of each epoch.
        """
        pass

    def on_predict_epoch_end(
        self, results: List[List[Tuple[np.ndarray, List[str],
                                       torch.Tensor]]]) -> None:
        """
        Write the predicted peptide sequences and amino acid scores to the
        output file.
        """
        pass

    def _get_output_peptide_and_scores(
            self, aa_tokens: List[str],
            aa_scores: torch.Tensor) -> Tuple[str, List[str], float, str]:
        """
        Get peptide to output, amino acid and peptide-level confidence scores.

        Parameters
        ----------
        aa_tokens : List[str]
            Amino acid tokens of the peptide sequence.
        aa_scores : torch.Tensor
            Amino acid-level confidence scores for the predicted sequence.

        Returns
        -------
        peptide : str
            Peptide sequence.
        aa_tokens : List[str]
            Amino acid tokens of the peptide sequence.
        peptide_score : str
            Peptide-level confidence score.
        aa_scores : str
            Amino acid-level confidence scores for the predicted sequence.
        """
        # Omit stop token.
        aa_tokens = aa_tokens[1:] if self.decoder.reverse else aa_tokens[:-1]
        peptide = "".join(aa_tokens)

        # If this is a non-finished beam (after exceeding `max_length`), return
        # a dummy (empty) peptide and NaN scores.
        if len(peptide) == 0:
            aa_tokens = []

        # Take scores corresponding to the predicted amino acids. Reverse tokens
        # to correspond with correct amino acids as needed.
        step = -1 if self.decoder.reverse else 1
        top_aa_scores = [
            aa_score[self.decoder._aa2idx[aa_token]].item()
            for aa_score, aa_token in zip(aa_scores, aa_tokens[::step])
        ][::step]

        # Get peptide-level score from amino acid-level scores.
        peptide_score = _aa_to_pep_score(top_aa_scores)
        aa_scores = ",".join(list(map("{:.5f}".format, top_aa_scores)))
        return peptide, aa_tokens, peptide_score, aa_scores

    def _log_history(self) -> None:
        """
        Write log to console, if requested.
        """
        # Log only if all output for the current epoch is recorded.
        if len(self._history) > 0 and len(self._history[-1]) == 6:
            if len(self._history) == 1:
                logger.info(
                    "Epoch\tTrain loss\tValid loss\tAA precision\tAA recall\t"
                    "Peptide recall")
            metrics = self._history[-1]
            if metrics["epoch"] % self.n_log == 0:
                logger.info(
                    "%i\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f",
                    metrics["epoch"] + 1,
                    metrics.get("train", np.nan),
                    metrics.get("valid", np.nan),
                    metrics.get("valid_aa_precision", np.nan),
                    metrics.get("valid_aa_recall", np.nan),
                    metrics.get("valid_pep_recall", np.nan),
                )
                if self.tb_summarywriter is not None:
                    for descr, key in [
                        ("loss/train_crossentropy_loss", "train"),
                        ("loss/dev_crossentropy_loss", "valid"),
                        ("eval/dev_aa_precision", "valid_aa_precision"),
                        ("eval/dev_aa_recall", "valid_aa_recall"),
                        ("eval/dev_pep_recall", "valid_pep_recall"),
                    ]:
                        self.tb_summarywriter.add_scalar(
                            descr,
                            metrics.get(key, np.nan),
                            metrics["epoch"] + 1,
                        )

    def configure_optimizers(
        self, ) -> Tuple[torch.optim.Optimizer, Dict[str, Any]]:
        """
        Initialize the optimizer.

        This is used by pytorch-lightning when preparing the model for training.

        Returns
        -------
        Tuple[torch.optim.Optimizer, Dict[str, Any]]
            The initialized Adam optimizer and its learning rate scheduler.
        """
        optimizer = torch.optim.AdamW(self.parameters(), **self.opt_kwargs)
        #optimizer = Lion(self.parameters(), **self.opt_kwargs)
        # Apply learning rate scheduler per step.
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.warmup_iters,
                                             max_iters=self.max_iters)
        return [optimizer], {"scheduler": lr_scheduler, "interval": "step"}

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        #add by xiang 
        '''
        if self.trainer.current_epoch >= 7:
            if total_norm >= 2.5:
                torch.nn.utils.clip_grad_norm(self.parameters(), 0.1)
        '''
        if(not self.logger==None):
            self.logger.experiment["/grad_norm_before_clip"].append(total_norm)
    '''
    def on_validation_batch_end(self, outputs, batch, batch_idx):
        print("hello!!!!!")

        print("val outputs: ", outputs )
        sys.stdout.flush()
    '''
    def on_train_batch_end(self, outputs, batch, batch_idx):
        #print("hello outputs??: ", outputs)
        # sys.stdout.flush()
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        if(not self.logger==None):
            self.logger.experiment["/grad_norm_after_clip"].append(total_norm)
            
    def on_after_backward(self) -> None:
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break

        if not valid_gradients:
            logger.warning(f'detected inf or nan values in gradients. not updating model parameters')
            self.zero_grad()

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with linear warm up followed by cosine shaped decay.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer object.
    warmup : int
        The number of warm up iterations.
    max_iters : torch.optim
        The total number of iterations.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, warmup: int,
                 max_iters: int):
        self.warmup, self.max_iters = warmup, max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):

        # Cosine annealing after a constant period
        # Author: Sheng Xu
        # Date: 20230214

        decay=self.warmup/self.max_iters
        if epoch <= self.warmup and self.warmup>0:
            #lr_factor = 1
            
            lr_factor = 1 * (epoch / self.warmup)
        else:
            
            lr_factor = 0.5 * (1 + np.cos(np.pi * ( (epoch - (decay * self.max_iters)) / ((1-decay) * self.max_iters))))
            #lr_factor = 0.75

        return lr_factor

def _aa_to_pep_score(aa_scores: List[float]) -> float:
    """
    Calculate peptide-level confidence score from amino acid level scores.

    Parameters
    ----------
    aa_scores : List[float]
        Amino acid level confidence scores.

    Returns
    -------
    float
        Peptide confidence score.
    """
    return np.mean(aa_scores)
def _calc_mass_error(calc_mz: float,
                     obs_mz: float,
                     charge: int,
                     isotope: int = 0) -> float:
    """
    Calculate the mass error in ppm between the theoretical m/z and the observed
    m/z, optionally accounting for an isotopologue mismatch.

    Parameters
    ----------
    calc_mz : float
        The theoretical m/z.
    obs_mz : float
        The observed m/z.
    charge : int
        The charge.
    isotope : int
        Correct for the given number of C13 isotopes (default: 0).

    Returns
    -------
    float
        The mass error in ppm.
    """
    return (calc_mz - (obs_mz - isotope * 1.00335 / charge)) / obs_mz * 10**6

