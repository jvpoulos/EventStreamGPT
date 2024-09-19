"""A model for fine-tuning on classification tasks."""
import torch
import math
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.nn import LayerNorm
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision, BinaryF1Score
from pytorch_lightning.callbacks import EarlyStopping

from ..data.types import PytorchBatch
from .config import StructuredEventProcessingMode, StructuredTransformerConfig, OptimizationConfig
from .model_output import StreamClassificationModelOutput
from .transformer import (
    ConditionallyIndependentPointProcessTransformer,
    NestedAttentionPointProcessTransformer,
    StructuredTransformerPreTrainedModel,
    ConditionallyIndependentPointProcessInputLayer,
    CustomLayerNorm
)
from .utils import safe_masked_max, safe_weighted_avg

from ..data.vocabulary import VocabularyConfig

import logging

# Configure the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ESTForStreamClassification(nn.Module):
    def __init__(self, config: StructuredTransformerConfig, vocabulary_config: VocabularyConfig, optimization_config: OptimizationConfig, oov_index: int, save_dir: str = "./model_outputs"):
        super().__init__()
        self.config = config
        self.optimization_config = optimization_config
        self.oov_index = oov_index
        self.save_dir = save_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  # Move the entire model to the device
        
        self.encoder = ConditionallyIndependentPointProcessTransformer(config, vocabulary_config, oov_index=oov_index, save_dir=self.save_dir).to(self.device)
        
        self.logit_layer = torch.nn.Linear(config.hidden_size, 1).to(self.device)
        self.criteria = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.dropout = torch.nn.Dropout(config.intermediate_dropout)
        
        self.static_indices_embedding = nn.Embedding(config.vocab_size, config.hidden_size).to(self.device)
        self.categorical_embeddings = nn.ModuleDict({
            'Female': nn.Embedding(2, config.hidden_size),
            'Married': nn.Embedding(2, config.hidden_size),
            'GovIns': nn.Embedding(2, config.hidden_size),
            'English': nn.Embedding(2, config.hidden_size),
            'Veteran': nn.Embedding(2, config.hidden_size)
        }).to(self.device)
        
        self.dynamic_values_encoder = nn.Linear(1, config.hidden_size).to(self.device)
        
        self.layer_norm = CustomLayerNorm(config.hidden_size) if config.use_layer_norm else nn.Identity()
        self.batch_norm = nn.BatchNorm1d(config.hidden_size) if config.use_batch_norm else nn.Identity()
        self.dynamic_values_norm = nn.BatchNorm1d(1).to(self.device)
        
        self.intermediate = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            self.layer_norm,
            self.batch_norm,
            self.dropout
        ).to(self.device)
        
        self.static_weight = nn.Parameter(torch.tensor(config.static_embedding_weight)).to(self.device)
        self.dynamic_weight = nn.Parameter(torch.tensor(config.dynamic_embedding_weight)).to(self.device)
        
        self.apply(self._init_weights)
        
        self.auroc = BinaryAUROC().to(self.device)
        self.auprc = BinaryAveragePrecision().to(self.device)
        self.f1_score = BinaryF1Score().to(self.device)

    def forward(self, dynamic_indices, dynamic_values=None, dynamic_measurement_indices=None, static_indices=None, static_measurement_indices=None, time=None, labels=None):

        # Encode the input
        encoded = self.encoder(
            dynamic_indices=dynamic_indices,
            dynamic_values=dynamic_values,
            dynamic_measurement_indices=dynamic_measurement_indices,
            static_indices=static_indices,
            static_measurement_indices=static_measurement_indices,
            time=time
        )

        event_encoded = encoded.last_hidden_state

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

        # Process static indices only if use_static_features is True
        if self.config.use_static_features and static_indices is not None:
            static_embed = self.static_indices_embedding(static_indices).mean(dim=1)
            combined_embed = self.static_weight * static_embed + self.dynamic_weight * pooled_dynamic
        else:
            combined_embed = pooled_dynamic
        
        # Process through intermediate layers
        intermediate = self.intermediate(combined_embed)
        logits = self.logit_layer(intermediate).squeeze(-1)
        probs = torch.sigmoid(logits)

        # Calculate metrics if labels are provided
        loss, accuracy, auc, auprc, f1 = None, None, None, None, None
        if labels is not None:
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
        
        return StreamClassificationModelOutput(
            loss=loss,
            preds=probs,
            labels=labels if labels is not None else None,
            accuracy=accuracy,
            auc=auc,
            auprc=auprc,
            f1=f1
        )

    def to(self, device):
        super().to(device)
        self.device = device
        return self

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