from transformers import CLIPProcessor, CLIPModel, AutoModelForImageTextToText, AutoProcessor # type: ignore
import torch # type: ignore
from PIL import Image # type: ignore
import numpy as np # type: ignore
from typing import Any, Dict, Union, List, Optional
from einops import rearrange, reduce # type: ignore

from .base import BaseFeatureExtractor

class VisionLanguageModelFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor from vision-language models.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the vision-language model feature extractor.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing:
                - model_name (str): Name of the vision-language model to use
                - device (str): Device to run the model on ('cuda' or 'cpu')
        """
        super().__init__(config)
        
        self.model_name = config["model_name"]
        self.layer_idx = config.get("layer_idx", -1)
        self.last_token = config.get("last_token", True)
        self.model_kwargs = config.get("model_kwargs", {})
        self.device = config.get("device", "cpu")


        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Initialize model and processor
        if "clip" in self.model_name.lower():
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(self.model_name, **self.model_kwargs).to(self.device)
            self.processor = AutoProcessor.from_pretrained(self.model_name)

        self.model.eval()

    def extract_features(self,
                        stimuli: Union[str, List[str], Image.Image, List[Image.Image]],
                        layer_idx: Optional[int] = None) -> Union[np.ndarray, Dict[int, np.ndarray]]:
        """Extract features from the input stimuli."""

        if layer_idx is None:
            layer_idx = self.layer_idx

        if isinstance(stimuli, str):
            stimuli = [stimuli]

        # Process each stimulus individually
        print(f"Processing {len(stimuli)} texts one at a time...")

        # Check if we're extracting all layers
        first_features = self._extract_features(stimuli[0])
        is_all_layers = isinstance(first_features, dict)

        if is_all_layers:
            # Return dict of layers: {layer_idx: np.array of shape (n_stimuli, hidden_size)}
            all_layer_features = {layer_idx: [] for layer_idx in first_features.keys()}
            
            # Add first stimulus features
            for layer_idx, feats in first_features.items():
                all_layer_features[layer_idx].append(feats.clone().detach().cpu().numpy())
            
            # Process remaining stimuli
            for i, text in enumerate(stimuli[1:], 1):
                if i % 10 == 0:
                    print(f"Processing text {i+1}/{len(stimuli)}")
                
                features = self._extract_features(text)
                for layer_idx, feats in features.items():
                    all_layer_features[layer_idx].append(feats.clone().detach().cpu().numpy())
            
            # Stack each layer
            return {layer_idx: np.vstack(feats_list) for layer_idx, feats_list in all_layer_features.items()}
        
        else:
            # Single layer extraction
            all_features = [first_features.clone().detach().cpu().numpy()]
            
            for i, text in enumerate(stimuli[1:], 1):
                if i % 10 == 0:
                    print(f"Processing text {i+1}/{len(stimuli)}")
                
                features = self._extract_features(text)
                all_features.append(features.clone().detach().cpu().numpy())
            
            return np.vstack(all_features)

    def _extract_features(self, stimulus: Union[str, 
                                                Image.Image, 
                                                List[Image.Image], 
                                                Dict[str, Image.Image]]
                                                ) -> torch.Tensor:

        if stimulus == "":
            # Get hidden size based on model type
            if "clip" in self.model_name.lower():
                hidden_size = self.model.config.text_config.hidden_size
            else:
                hidden_size = self.model.config.hidden_size
            
            zeros = torch.zeros((1, hidden_size), device=self.device)
            features = rearrange(zeros, 'b d -> b d')
            return features.to(self.device)

        # Process the stimulus using the processor
        inputs = self.processor(text=stimulus, images=None, return_tensors="pt", padding=True) # text only for now
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # For CLIP, use text_model directly for text-only inputs
            if "clip" in self.model_name.lower():
                outputs = self.model.text_model(**inputs, output_hidden_states=True)
            else:
                outputs = self.model(**inputs, output_hidden_states=True)

            # aggregate features from the specified layer
            if self.layer_idx != -1 and self.layer_idx < len(outputs.hidden_states):
                hidden_states = outputs.hidden_states[self.layer_idx]
            
                if self.last_token:
                    # Take the features of the last token
                    hidden_states = hidden_states[:, -1, :]  # (batch_size, hidden_size)
                else:
                    # Mean pool over the sequence length
                    hidden_states = reduce(hidden_states, 'b n d -> b d', reduction='mean')  # (batch_size, hidden_size)
                features = rearrange(hidden_states, 'b d -> b d')
                return features.to(self.device)
        
            elif self.layer_idx == -1:

                all_layer_features = {}

                for idx, layer_hidden_states in enumerate(outputs.hidden_states):
                    if self.last_token:
                        layer_hidden_states = layer_hidden_states[:, -1, :]  # (batch_size, hidden_size)
                    else:
                        layer_hidden_states = reduce(layer_hidden_states, 'b n d -> b d', reduction='mean')  # (batch_size, hidden_size)
                    
                    features = rearrange(layer_hidden_states, 'b d -> b d')
                    all_layer_features[idx] = features.to(self.device)

                return all_layer_features
            
    def _validate_config(self) -> None:
        """Validate the configuration parameters."""
        required_params = ["model_name"]
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing required parameter: {param}")

        if "layer_idx" in self.config:
            if not isinstance(self.config["layer_idx"], int):
                raise ValueError("layer_idx must be an integer")

        if "device" in self.config:
            if self.config["device"] not in ["cuda", "cpu", "mps"]:
                raise ValueError("device must be either 'cuda', 'cpu', or 'mps'")

        if "context_type" in self.config:
            valid_context_types = ["fullcontext", "nocontext", "halfcontext"]
            if self.config["context_type"] not in valid_context_types:
                raise ValueError(f"context_type must be one of {valid_context_types}")