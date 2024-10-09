"""A model for fine-tuning on classification tasks."""
import torch
import math
from collections import defaultdict
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.nn import LayerNorm
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, BinaryF1Score
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import LightningModule

from ..data.types import PytorchBatch
from .config import StructuredEventProcessingMode, StructuredTransformerConfig, OptimizationConfig
from .model_output import StreamClassificationModelOutput
from .transformer import (
    ConditionallyIndependentPointProcessTransformer,
    NestedAttentionPointProcessTransformer,
    StructuredTransformerPreTrainedModel,
    ConditionallyIndependentPointProcessInputLayer
)
from .utils import safe_masked_max, safe_weighted_avg

from ..data.vocabulary import VocabularyConfig

import logging

# Set up the standard logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ESTForStreamClassification(LightningModule):
    def __init__(self, config: StructuredTransformerConfig, vocabulary_config: VocabularyConfig, optimization_config: OptimizationConfig, oov_index: int, save_dir: str = "./model_outputs"):
        super().__init__()
        self.config = config
        self.optimization_config = optimization_config
        self.oov_index = oov_index
        self.save_dir = save_dir

        self.encoder = ConditionallyIndependentPointProcessTransformer(config, vocabulary_config, oov_index=oov_index, save_dir=self.save_dir)
        
        self.logit_layer = torch.nn.Linear(config.hidden_size, 1)
        self.criteria = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.dropout = torch.nn.Dropout(config.intermediate_dropout)
        
        self.static_indices_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        
        self.layer_norm = nn.LayerNorm(config.hidden_size) if config.use_layer_norm else nn.Identity()
        self.batch_norm = nn.BatchNorm1d(config.hidden_size) if config.use_batch_norm else nn.Identity()
        self.dynamic_values_norm = nn.BatchNorm1d(1)
        
        self.intermediate = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            self.layer_norm,
            self.batch_norm,
            self.dropout
        )
        
        self.static_weight = nn.Parameter(torch.tensor(config.static_embedding_weight))
        self.dynamic_weight = nn.Parameter(torch.tensor(config.dynamic_embedding_weight))
        
        self.apply(self._init_weights)
        
        self.auroc = BinaryAUROC()
        self.auprc = BinaryAveragePrecision()
        self.f1_score = BinaryF1Score()

        # Initialize metric accumulator
        self.metric_accumulator = defaultdict(list)

        # Ensure all parameters require gradients
        self.ensure_all_params_require_grad()
            
    def ensure_all_params_require_grad(self):
        for name, param in self.named_parameters():
            if not param.requires_grad:
                logger.warning(f"Parameter {name} does not require gradients. Setting requires_grad=True.")
                param.requires_grad = True

    def set_dtype(self, dtype):
        self.to(dtype)
        
    def forward(self, batch=None, output_attentions=False, **kwargs):
        if batch is not None:
            kwargs.update(batch)
        
        # Extract inputs from kwargs
        dynamic_indices = kwargs.get('dynamic_indices')
        dynamic_values = kwargs.get('dynamic_values')
        dynamic_measurement_indices = kwargs.get('dynamic_measurement_indices')
        static_indices = kwargs.get('static_indices')
        static_measurement_indices = kwargs.get('static_measurement_indices')
        time = kwargs.get('time')
        labels = kwargs.get('labels')
        seq_attention_mask = kwargs.get('attention_mask')

        # Convert indices to long tensors
        dynamic_indices = dynamic_indices.long()
        if dynamic_measurement_indices is not None:
            dynamic_measurement_indices = dynamic_measurement_indices.long()
        if static_indices is not None:
            static_indices = static_indices.long()
        if static_measurement_indices is not None:
            static_measurement_indices = static_measurement_indices.long()

        # Log shapes and dtypes
        logger.debug(f"dynamic_indices shape: {dynamic_indices.shape}, dtype: {dynamic_indices.dtype}")
        logger.debug(f"dynamic_values shape: {dynamic_values.shape if dynamic_values is not None else None}, dtype: {dynamic_values.dtype if dynamic_values is not None else None}")
        logger.debug(f"dynamic_measurement_indices shape: {dynamic_measurement_indices.shape if dynamic_measurement_indices is not None else None}, dtype: {dynamic_measurement_indices.dtype if dynamic_measurement_indices is not None else None}")
        logger.debug(f"static_indices shape: {static_indices.shape if static_indices is not None else None}, dtype: {static_indices.dtype if static_indices is not None else None}")
        logger.debug(f"static_measurement_indices shape: {static_measurement_indices.shape if static_measurement_indices is not None else None}, dtype: {static_measurement_indices.dtype if static_measurement_indices is not None else None}")
        logger.debug(f"time shape: {time.shape if time is not None else None}, dtype: {time.dtype if time is not None else None}")

        # Update dtype conversion:
        dtype = self.encoder.dtype  # Use the encoder's dtype
        dynamic_indices = dynamic_indices.to(dtype)
        if dynamic_values is not None:
            dynamic_values = dynamic_values.to(dtype)
        if dynamic_measurement_indices is not None:
            dynamic_measurement_indices = dynamic_measurement_indices.to(dtype)
        if static_indices is not None:
            static_indices = static_indices.to(dtype)
        if static_measurement_indices is not None:
            static_measurement_indices = static_measurement_indices.to(dtype)
        if time is not None:
            time = time.to(dtype)

        # Create seq_attention_mask if needed
        if seq_attention_mask is None:
            seq_attention_mask = torch.ones_like(dynamic_indices, dtype=torch.bool)

        # Get the batch size from dynamic_indices
        batch_size = dynamic_indices.size(0)
        
        # Prepare input features for the encoder
        input_features = {
            'dynamic_indices': dynamic_indices,
            'dynamic_values': dynamic_values,
            'dynamic_measurement_indices': dynamic_measurement_indices,
            'static_indices': static_indices,
            'static_measurement_indices': static_measurement_indices,
            'time': time,
            'seq_attention_mask': seq_attention_mask
        }

        # Only pass labels to encoder if use_fake_feature is True
        if getattr(self.config, 'use_fake_feature', False):
            input_features['labels'] = labels

        # Add gradient clipping
        if self.training:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.config.max_grad_norm)

        # Encode the input
        logger.debug("Encoder forward pass")
        encoded = self.encoder(**input_features, output_attentions=output_attentions)
        
        logger.debug("Processing encoder output")
        event_encoded = encoded.last_hidden_state

        # Check for NaN values
        if torch.isnan(event_encoded).any():
            logger.warning("NaN values detected in encoded output. Replacing with zeros.")
            event_encoded = torch.where(torch.isnan(event_encoded), torch.zeros_like(event_encoded), event_encoded)

        # Log attention mechanism used
        if self.config.use_flash_attention:
            self.log("use_flash_attention", 1.0, on_step=True, on_epoch=False)
        else:
            self.log("use_flash_attention", 0.0, on_step=True, on_epoch=False)

        # Handle 4D input
        if event_encoded.dim() == 4:
            event_encoded = event_encoded.squeeze(2)  # Remove the dep_graph dimension

        # Pool the dynamic embeddings
        if self.config.task_specific_params["pooling_method"] == "mean":
            pooled_dynamic = event_encoded.mean(dim=1)
        elif self.config.task_specific_params["pooling_method"] == "max":
            pooled_dynamic = event_encoded.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling method: {self.config.task_specific_params['pooling_method']}")

        # Ensure pooled_dynamic has the correct batch size
        if pooled_dynamic.size(0) != batch_size:
            logger.warning(f"Adjusting pooled_dynamic size from {pooled_dynamic.size(0)} to {batch_size}")
            pooled_dynamic = pooled_dynamic[:batch_size]

        # Process static indices only if use_static_features is True
        if self.config.use_static_features and static_indices is not None:
            static_embed = self.static_indices_embedding(static_indices.long()).mean(dim=1)
            combined_embed = self.static_weight * static_embed + self.dynamic_weight * pooled_dynamic
        else:
            combined_embed = pooled_dynamic

        # Process through intermediate layers
        intermediate = self.intermediate(combined_embed)
        logits = self.logit_layer(intermediate).squeeze(-1)

        # Ensure logits have the correct batch size
        if logits.size(0) != batch_size:
            logger.warning(f"Adjusting logits size from {logits.size(0)} to {batch_size}")
            logits = logits[:batch_size]

        probs = torch.sigmoid(logits)

        # Calculate metrics if labels are provided
        loss, accuracy, auc, auprc, f1 = None, None, None, None, None
        if labels is not None:
            # Ensure labels have the correct batch size
            if labels.size(0) != batch_size:
                logger.warning(f"Adjusting labels size from {labels.size(0)} to {batch_size}")
                labels = labels[:batch_size]

            loss = self.criteria(logits, labels)
            loss = loss.mean()
            with torch.no_grad():
                self.auroc.update(probs, labels.int())
                self.auprc.update(probs, labels.int())
                self.f1_score.update(probs, labels.int())
                auc = self.auroc.compute()
                auprc = self.auprc.compute()
                f1 = self.f1_score.compute()
                accuracy = ((probs > 0.5) == labels).float().mean()

        # Process attention outputs
        attention_outputs = None
        if output_attentions and encoded.attentions is not None:
            attention_outputs = []
            for layer_attention in encoded.attentions:
                if isinstance(layer_attention, torch.Tensor):
                    attention_outputs.append({'attn_weights': layer_attention})
                elif isinstance(layer_attention, dict) and 'attn_weights' in layer_attention:
                    attention_outputs.append(layer_attention)

        logger.debug("Completed forward pass in ESTForStreamClassification")
        return StreamClassificationModelOutput(
            loss=loss,
            preds=probs,
            labels=labels if labels is not None else None,
            accuracy=accuracy,
            auc=auc,
            auprc=auprc,
            f1=f1,
            attentions=attention_outputs
        )

    def on_epoch_start(self):
        # Reset all metrics at the start of each epoch
        self.auroc.reset()
        self.auprc.reset()
        self.f1_score.reset()

    def gradient_checkpointing_enable(self):
        self.encoder.gradient_checkpointing = True
        if hasattr(self.encoder, 'gradient_checkpointing_enable'):
            self.encoder.gradient_checkpointing_enable()
        else:
            print("Warning: encoder does not have gradient_checkpointing_enable method")
            
    def _set_static_graph(self):
        for module in self.children():
            if hasattr(module, '_set_static_graph'):
                module._set_static_graph()

        if hasattr(self.encoder, 'h'):
            for block in self.encoder.h:
                if hasattr(block, '_set_static_graph'):
                    block._set_static_graph()
                
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            embedding_dim = module.embedding_dim
            std = math.sqrt(1.0 / embedding_dim)
            nn.init.normal_(module.weight, mean=0, std=std)

    def _init_input_embeddings(self):
        if hasattr(self.encoder, 'input_layer'):
            if hasattr(self.encoder.input_layer, 'data_embedding_layer'):
                data_embedding_layer = self.encoder.input_layer.data_embedding_layer
                if hasattr(data_embedding_layer, 'embedding'):
                    embedding = data_embedding_layer.embedding
                    embedding_dim = embedding.embedding_dim
                    std = math.sqrt(1.0 / embedding_dim)
                    nn.init.normal_(embedding.weight, mean=0, std=std)
                elif hasattr(data_embedding_layer, 'categorical_embedding'):
                    embedding = data_embedding_layer.categorical_embedding
                    embedding_dim = embedding.embedding_dim
                    std = math.sqrt(1.0 / embedding_dim)
                    nn.init.normal_(embedding.weight, mean=0, std=std)

            if hasattr(self.encoder.input_layer, 'time_embedding_layer'):
                if isinstance(self.encoder.input_layer.time_embedding_layer, nn.Embedding):
                    embedding_dim = self.encoder.input_layer.time_embedding_layer.embedding_dim
                    std = math.sqrt(1.0 / embedding_dim)
                    nn.init.normal_(self.encoder.input_layer.time_embedding_layer.weight, mean=0, std=std)
                elif hasattr(self.encoder.input_layer.time_embedding_layer, 'sin_div_term') and \
                     hasattr(self.encoder.input_layer.time_embedding_layer, 'cos_div_term'):
                    nn.init.normal_(self.encoder.input_layer.time_embedding_layer.sin_div_term)
                    nn.init.normal_(self.encoder.input_layer.time_embedding_layer.cos_div_term)

    def initialize_weights(self):
        # Apply Xavier initialization to all parameters except embeddings
        self.apply(self._init_weights)
        
        # Apply Gaussian initialization to input embeddings
        self._init_input_embeddings()

        # Initialize other components if needed
        if hasattr(self, 'encoder'):
            if hasattr(self.encoder, 'initialize_weights'):
                self.encoder.initialize_weights()
            else:
                self.encoder.apply(self._init_weights)

        if hasattr(self, 'decoder'):
            if hasattr(self.decoder, 'initialize_weights'):
                self.decoder.initialize_weights()
            else:
                self.decoder.apply(self._init_weights)

    @classmethod
    def from_pretrained(cls, pretrained_weights_fp, config, vocabulary_config, optimization_config, oov_index):
        model = cls(config, vocabulary_config, optimization_config, oov_index)
        # Load pretrained weights here
        return model

    @property
    def current_epoch(self):
        return self._current_epoch

    @current_epoch.setter
    def current_epoch(self, value):
        self._current_epoch = value
        if hasattr(self.encoder, 'current_epoch'):
            self.encoder.current_epoch = value
            
    def on_train_epoch_start(self):
        self.current_epoch += 1
        if hasattr(self.encoder, 'current_epoch'):
            self.encoder.current_epoch = self.current_epoch

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.optimization_config['batch_size'],
            shuffle=True,
            num_workers=self.optimization_config['num_dataloader_workers'],
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.optimization_config['validation_batch_size'],
            shuffle=False,
            num_workers=self.optimization_config['num_dataloader_workers'],
            pin_memory=True
        )
    
    @property
    def _uses_dep_graph(self):
        return self.config.structured_event_processing_mode == StructuredEventProcessingMode.NESTED_ATTENTION