"""A model for fine-tuning on classification tasks."""
import torch
import math
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.nn import LayerNorm
from torchmetrics.classification import BinaryAUROC
from pytorch_lightning.callbacks import EarlyStopping

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

# Configure the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ESTForStreamClassification(nn.Module):
    def __init__(self, config: StructuredTransformerConfig, vocabulary_config: VocabularyConfig, optimization_config: OptimizationConfig, oov_index: int):
        super().__init__()
        self.config = config
        self.vocabulary_config = vocabulary_config
        self.optimization_config = optimization_config
        self.oov_index = oov_index
        self._current_epoch = 0

        if not hasattr(self.config, 'save_dir'):
            self.config.save_dir = './ESTForStreamClassificationmodel_outputs'

        dynamic_indices_vocab_size = vocabulary_config.vocab_sizes_by_measurement.get("dynamic_indices", 0)
        self.oov_index = dynamic_indices_vocab_size + 1

        if self._uses_dep_graph:
            self.encoder = NestedAttentionPointProcessTransformer(self.config)
        else:
            self.encoder = ConditionallyIndependentPointProcessTransformer(self.config, vocabulary_config, oov_index=self.oov_index)

        self.encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
        self.pooling_method = config.task_specific_params["pooling_method"]
        
        self.logit_layer = torch.nn.Linear(config.hidden_size, 1)
        self.criteria = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.dropout = torch.nn.Dropout(config.intermediate_dropout)
        
        self.static_indices_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.static_measurement_indices_embedding = nn.Embedding(len(vocabulary_config.measurements_idxmap), config.hidden_size)

        # Keep the individual static feature embeddings for backward compatibility
        self.categorical_embeddings = nn.ModuleDict({
            'Female': nn.Embedding(2, config.hidden_size),
            'Married': nn.Embedding(2, config.hidden_size),
            'GovIns': nn.Embedding(2, config.hidden_size),
            'English': nn.Embedding(2, config.hidden_size),
            'Veteran': nn.Embedding(2, config.hidden_size)
        })
        
        self.continuous_encoder = nn.Linear(3, config.hidden_size)

        self.dynamic_values_encoder = nn.Linear(1, config.hidden_size)
        
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
        self.max_grad_norm = getattr(config, 'max_grad_norm', 1.0)
        self.use_grad_value_clipping = getattr(optimization_config, 'use_grad_value_clipping', False)
        self.clip_grad_value = getattr(optimization_config, 'clip_grad_value', 1.0)
        
        self.missing_value_embedding = nn.Parameter(torch.randn(config.hidden_size))
        
        self.encoder.gradient_checkpointing = getattr(self.config, 'use_gradient_checkpointing', False)

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

    def forward(self, batch: dict, labels=None):
        device = self.logit_layer.weight.device
        
        # Print out all keys in the batch
        print("Batch keys:", batch.keys())
        
        # Use .get() method with default values for potentially missing keys
        dynamic_indices = batch.get('dynamic_indices', torch.tensor([])).to(device)
        dynamic_measurement_indices = batch.get('dynamic_measurement_indices', torch.tensor([])).to(device)
        dynamic_values = batch.get('dynamic_values', torch.tensor([])).to(device)
        static_indices = batch.get('static_indices', torch.tensor([])).to(device)
        static_measurement_indices = batch.get('static_measurement_indices', torch.tensor([])).to(device)
        time = batch.get('time', torch.tensor([])).to(device)

        # Check if we have the necessary data to proceed
        if dynamic_indices.numel() == 0 or static_indices.numel() == 0:
            print("Warning: Missing crucial data in batch")
            return StreamClassificationModelOutput(
                loss=torch.tensor(0.0, device=device),
                preds=torch.tensor([], device=device),
                labels=labels,
                accuracy=torch.tensor(0.0, device=device),
                auc=torch.tensor(0.0, device=device)
            )

        # Create masks for padded sequences
        dynamic_mask = (dynamic_indices != 0).float()
        static_mask = (static_indices != 0).float()

        # Process static embeddings
        static_embeds = self.static_indices_embedding(static_indices)
        static_measurement_embeds = self.static_measurement_indices_embedding(static_measurement_indices)
        static_embed = (static_embeds * static_measurement_embeds * static_mask.unsqueeze(-1)).sum(dim=1) / (static_mask.sum(dim=1, keepdim=True) + 1e-8)

        # Process dynamic embeddings
        pytorch_batch = {
            'dynamic_indices': dynamic_indices,
            'dynamic_measurement_indices': dynamic_measurement_indices,
            'dynamic_values': dynamic_values,
            'dynamic_mask': dynamic_mask,
            'time': time
        }
        
        encoded = self.encoder(pytorch_batch)
        event_encoded = encoded.last_hidden_state

        if self.pooling_method == "mean":
            pooled_dynamic = (event_encoded * dynamic_mask.unsqueeze(-1)).sum(dim=1) / (dynamic_mask.sum(dim=1, keepdim=True) + 1e-8)
        elif self.pooling_method == "max":
            pooled_dynamic = torch.max(event_encoded * dynamic_mask.unsqueeze(-1) + (1 - dynamic_mask.unsqueeze(-1)) * -1e9, dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")

        # Combine static and dynamic embeddings
        combined_embed = self.static_weight * static_embed + self.dynamic_weight * pooled_dynamic
        
        intermediate = self.intermediate(combined_embed)
        logits = self.logit_layer(intermediate).squeeze(-1)
        probs = torch.sigmoid(logits)

        loss, accuracy, auc = None, None, None
        if labels is not None:
            labels = labels.to(logits.device).float()
            loss = self.criteria(logits, labels)
            loss = loss.mean()

            with torch.no_grad():
                self.auroc.update(probs, labels.int())
                auc = self.auroc.compute()
                accuracy = ((probs > 0.5) == labels).float().mean()
        
        return StreamClassificationModelOutput(
            loss=loss,
            preds=logits,
            labels=labels,
            accuracy=accuracy,
            auc=auc
        )
    
    @property
    def _uses_dep_graph(self):
        return self.config.structured_event_processing_mode == StructuredEventProcessingMode.NESTED_ATTENTION