"""Loss functions for Multi-Weight Neural Networks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class MultiPathwayLoss(nn.Module):
    """Loss function that combines primary task loss with pathway-specific regularization."""
    
    def __init__(self,
                 primary_loss: nn.Module = None,
                 pathway_regularization: float = 0.1,
                 balance_pathways: bool = True):
        """
        Args:
            primary_loss: Main task loss (e.g., CrossEntropyLoss)
            pathway_regularization: Weight for pathway regularization
            balance_pathways: Whether to balance color and brightness pathways
        """
        super().__init__()
        self.primary_loss = primary_loss or nn.CrossEntropyLoss()
        self.pathway_regularization = pathway_regularization
        self.balance_pathways = balance_pathways
    
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor,
                pathway_outputs: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            outputs: Model predictions
            targets: Ground truth labels
            pathway_outputs: Optional dict with 'color' and 'brightness' pathway outputs
        
        Returns:
            Total loss
        """
        # Primary task loss
        loss = self.primary_loss(outputs, targets)
        
        # Pathway regularization
        if pathway_outputs and self.pathway_regularization > 0:
            reg_loss = self._compute_pathway_regularization(pathway_outputs)
            loss = loss + self.pathway_regularization * reg_loss
        
        return loss
    
    def _compute_pathway_regularization(self, pathway_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute regularization to encourage pathway specialization."""
        color_out = pathway_outputs.get('color')
        brightness_out = pathway_outputs.get('brightness')
        
        if color_out is None or brightness_out is None:
            return torch.tensor(0.0)
        
        # Encourage different representations
        # Using negative cosine similarity
        color_norm = F.normalize(color_out, dim=1)
        brightness_norm = F.normalize(brightness_out, dim=1)
        
        similarity = torch.mean(torch.sum(color_norm * brightness_norm, dim=1))
        
        # We want to minimize similarity (maximize diversity)
        diversity_loss = 1.0 + similarity  # Range [0, 2], 0 when orthogonal
        
        if self.balance_pathways:
            # Encourage balanced activation magnitudes
            color_mag = torch.mean(torch.norm(color_out, dim=1))
            brightness_mag = torch.mean(torch.norm(brightness_out, dim=1))
            
            balance_loss = torch.abs(color_mag - brightness_mag) / (color_mag + brightness_mag + 1e-8)
            
            return diversity_loss + balance_loss
        
        return diversity_loss


class IntegrationRegularizationLoss(nn.Module):
    """Loss that regularizes the integration weights in continuous integration models."""
    
    def __init__(self,
                 sparsity_weight: float = 0.01,
                 temporal_smoothness: float = 0.01):
        """
        Args:
            sparsity_weight: Encourage sparse integration (rely on one pathway)
            temporal_smoothness: Encourage smooth changes in integration weights
        """
        super().__init__()
        self.sparsity_weight = sparsity_weight
        self.temporal_smoothness = temporal_smoothness
        self.previous_weights = None
    
    def forward(self, integration_weights: Dict[str, Dict[str, float]]) -> torch.Tensor:
        """
        Compute integration regularization loss.
        
        Args:
            integration_weights: Dict of integration weight dicts from model
        
        Returns:
            Regularization loss
        """
        loss = torch.tensor(0.0)
        
        current_weights = []
        for stage_name, weights in integration_weights.items():
            weight_tensor = torch.tensor([
                weights.get('color', 0.5),
                weights.get('brightness', 0.5),
                weights.get('integrated', 0.0)
            ])
            current_weights.append(weight_tensor)
            
            # Sparsity loss - encourage weights to be close to 0 or 1
            if self.sparsity_weight > 0:
                sparsity = torch.sum(weight_tensor * (1 - weight_tensor))
                loss = loss + self.sparsity_weight * sparsity
        
        # Temporal smoothness
        if self.temporal_smoothness > 0 and self.previous_weights is not None:
            for curr, prev in zip(current_weights, self.previous_weights):
                smoothness = torch.norm(curr - prev)
                loss = loss + self.temporal_smoothness * smoothness
        
        self.previous_weights = current_weights
        
        return loss


class PathwayBalanceLoss(nn.Module):
    """Loss to ensure both pathways contribute equally to the final prediction."""
    
    def __init__(self, balance_weight: float = 0.1):
        super().__init__()
        self.balance_weight = balance_weight
    
    def forward(self, color_contribution: torch.Tensor,
                brightness_contribution: torch.Tensor) -> torch.Tensor:
        """
        Compute pathway balance loss.
        
        Args:
            color_contribution: Contribution from color pathway
            brightness_contribution: Contribution from brightness pathway
        
        Returns:
            Balance loss
        """
        # Compute average contribution magnitude
        color_mag = torch.mean(torch.abs(color_contribution))
        brightness_mag = torch.mean(torch.abs(brightness_contribution))
        
        # Penalize imbalance
        imbalance = torch.abs(color_mag - brightness_mag) / (color_mag + brightness_mag + 1e-8)
        
        return self.balance_weight * imbalance


class RobustnessLoss(nn.Module):
    """Loss to improve robustness to color/brightness perturbations."""
    
    def __init__(self,
                 primary_loss: nn.Module = None,
                 robustness_weight: float = 0.1,
                 perturbation_type: str = 'both'):
        """
        Args:
            primary_loss: Main task loss
            robustness_weight: Weight for robustness term
            perturbation_type: 'color', 'brightness', or 'both'
        """
        super().__init__()
        self.primary_loss = primary_loss or nn.CrossEntropyLoss()
        self.robustness_weight = robustness_weight
        self.perturbation_type = perturbation_type
    
    def forward(self, model: nn.Module, inputs: torch.Tensor, 
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss with robustness term.
        
        Args:
            model: The model being trained
            inputs: Input images
            targets: Ground truth labels
        
        Returns:
            Total loss with robustness term
        """
        # Standard forward pass
        outputs = model(inputs)
        standard_loss = self.primary_loss(outputs, targets)
        
        if self.robustness_weight == 0:
            return standard_loss
        
        # Create perturbed inputs
        perturbed_inputs = self._create_perturbations(inputs)
        
        # Forward pass with perturbations
        perturbed_outputs = model(perturbed_inputs)
        
        # Consistency loss - predictions should be similar
        consistency_loss = F.kl_div(
            F.log_softmax(perturbed_outputs, dim=1),
            F.softmax(outputs.detach(), dim=1),
            reduction='batchmean'
        )
        
        return standard_loss + self.robustness_weight * consistency_loss
    
    def _create_perturbations(self, inputs: torch.Tensor) -> torch.Tensor:
        """Create color or brightness perturbations."""
        perturbed = inputs.clone()
        
        if self.perturbation_type in ['color', 'both']:
            # Random hue shift
            hue_shift = torch.rand(1) * 0.2 - 0.1
            # Simple approximation - in practice use proper color space conversion
            perturbed = perturbed + hue_shift
        
        if self.perturbation_type in ['brightness', 'both']:
            # Random brightness adjustment
            brightness_factor = torch.rand(1) * 0.4 + 0.8  # 0.8 to 1.2
            perturbed = perturbed * brightness_factor
        
        return torch.clamp(perturbed, 0, 1)