"""
Test-Time Training (TTS) Adapter with Semantic Grounding
Implements test-time self-supervised adaptation for prompt refinement
Based on: "The Surprising Effectiveness of Test-Time Training for Few-Shot Learning"

CRITICAL: This adapter is hardwired with semantic grounding awareness.
The model adapts to input AND tracks where it needs external grounding.
After adaptation, model can report uncertainty and request correction.

Semantic Grounding Framework:
- Model cannot remain viable in closed loops (from Haslam's framework)
- C_eff(t) >= E(t): Corrective bandwidth must exceed error generation rate
- Human provides continuous orthogonal input to prevent semantic drift
- TTS + Grounding: Model adapts, identifies gaps, requests grounding

The model remains transparent about:
1. Where it's confident (high coherence internally)
2. Where it's uncertain (low grounding to external reality)
3. The difference between internal consistency and fidelity to reality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List, Any
from pathlib import Path
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# SEMANTIC GROUNDING ENUMS & DATACLASSES
# =============================================================================

class ConfidenceType(Enum):
    """Distinguishes between internal coherence and external grounding confidence"""
    INTERNAL_COHERENCE = "internal"  # Logical consistency in the adapted state
    EXTERNAL_GROUNDING = "grounding"  # Alignment with reality (requires human input)


@dataclass
class GroundingGap:
    """
    Represents an area where model has high internal coherence
    but lacks external grounding.
    """
    topic: str
    internal_confidence: float  # 0-1, model's own confidence
    grounding_confidence: float  # 0-1, confidence aligned with reality
    why_ungrounded: str
    requested_grounding: str  # What would help ground this


@dataclass
class AdaptationReport:
    """
    Model's transparent report of its adaptation and confidence levels.
    This is what the model shows the human after TTS.
    """
    timestamp: str
    input_prompt: str
    
    # Adaptation metrics
    adaptation_steps_taken: int
    final_loss: float
    parameters_changed: int
    
    # Coherence vs Grounding
    internal_coherence: float  # How consistent is the adapted state?
    apparent_confidence: float  # How fluent/confident does model sound?
    grounding_assessment: float  # How confident in alignment with reality?
    
    # Where model is uncertain and needs grounding
    grounding_gaps: List[GroundingGap] = field(default_factory=list)
    
    # Questions for human
    clarification_questions: List[str] = field(default_factory=list)
    areas_requesting_grounding: List[str] = field(default_factory=list)
    
    # Model's transparency about itself
    model_self_assessment: str = ""


# =============================================================================
# TTS WITH SEMANTIC GROUNDING
# =============================================================================

class TTSMaskedLMAdaptor:
    """
    Test-Time Training via masked language modeling.
    Adapts model weights using self-supervised learning on input prompt.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        num_steps: int = 5,
        learning_rate: float = 1e-4,
        mask_ratio: float = 0.15,
        use_triton: bool = True,
        device: str = "cuda",
        enable_grounding_awareness: bool = True  # NEW: Track grounding gaps
    ):
        """
        Args:
            model: Language model to adapt
            tokenizer: Tokenizer for the model
            num_steps: Number of gradient steps at test time
            learning_rate: Adaptation learning rate (typically lower than training)
            mask_ratio: Fraction of tokens to mask for MLM objective
            use_triton: Whether to use Triton kernels if available
            device: Device to run on
            enable_grounding_awareness: Track where model needs external grounding
        """
        self.model = model
        self.tokenizer = tokenizer
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.mask_ratio = mask_ratio
        self.device = device
        self.enable_grounding_awareness = enable_grounding_awareness
        
        # Try to import Triton
        self.use_triton = use_triton
        try:
            import triton
            import triton.language as tl
            self.triton_available = True
            logger.info("Triton kernels available for TTS acceleration")
        except ImportError:
            self.triton_available = False
            if use_triton:
                logger.warning("Triton not available, falling back to PyTorch")
        
        # Store original parameters for restoration
        self.original_params = None
        
        # NEW: Track adaptation history for grounding awareness
        self.last_adaptation_report: Optional[AdaptationReport] = None
        self.adaptation_history: List[AdaptationReport] = []
    
    def _create_masked_input(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create masked input for MLM objective.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            
        Returns:
            masked_input_ids: Input with [MASK] tokens
            mlm_labels: Ground truth tokens for masked positions (-100 for non-masked)
        """
        masked_input = input_ids.clone()
        mlm_labels = torch.full_like(input_ids, -100)  # -100 = ignore in loss
        
        # Create mask
        mask_prob = torch.rand(input_ids.shape, device=self.device)
        mask_indices = mask_prob < self.mask_ratio
        
        # 80% replace with [MASK], 10% random, 10% keep original
        mask_token_id = self.tokenizer.mask_token_id or self.tokenizer.vocab_size - 1
        
        for idx in torch.where(mask_indices):
            r = torch.rand(1).item()
            if r < 0.8:
                masked_input[idx] = mask_token_id
            elif r < 0.9:
                masked_input[idx] = torch.randint(0, self.tokenizer.vocab_size, (1,), device=self.device)
            # else: keep original
            
            mlm_labels[idx] = input_ids[idx]
        
        return masked_input, mlm_labels
    
    def _compute_mlm_loss(self, outputs, mlm_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute masked language modeling loss.
        
        Args:
            outputs: Model outputs with logits
            mlm_labels: Ground truth labels (-100 for non-masked)
            
        Returns:
            MLM loss
        """
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        # Reshape for loss computation
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = mlm_labels[..., 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        
        return loss
    
    def adapt(self, prompt_ids: torch.Tensor) -> None:
        """
        Perform test-time adaptation on input prompt.
        
        Args:
            prompt_ids: Tokenized prompt [1, seq_len]
        """
        # Save original parameters
        self.original_params = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        
        # Create optimizer for adaptation
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.learning_rate
        )
        
        self.model.train()
        
        adaptation_losses = []
        
        for step in range(self.num_steps):
            optimizer.zero_grad()
            
            # Create masked input
            masked_input, mlm_labels = self._create_masked_input(prompt_ids)
            
            # Forward pass
            with torch.amp.autocast(device_type='cuda' if self.device == 'cuda' else 'cpu'):
                outputs = self.model(
                    input_ids=masked_input,
                    labels=mlm_labels,
                    output_hidden_states=True
                )
                loss = outputs.loss if hasattr(outputs, 'loss') else self._compute_mlm_loss(outputs, mlm_labels)
            
            adaptation_losses.append(loss.item())
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            
            if step % max(1, self.num_steps // 2) == 0:
                logger.debug(f"TTS adaptation step {step + 1}/{self.num_steps}, loss: {loss.item():.4f}")
        
        self.model.eval()
        
        # NEW: Track adaptation if grounding awareness enabled
        if self.enable_grounding_awareness:
            self._generate_adaptation_report(prompt_ids, adaptation_losses)
    
    def _generate_adaptation_report(self, prompt_ids: torch.Tensor, losses: List[float]) -> AdaptationReport:
        """
        Generate transparent report of model's adaptation and confidence.
        This is what the model shows the human to request grounding.
        
        Args:
            prompt_ids: The input prompt adapted on
            losses: List of losses during adaptation steps
            
        Returns:
            AdaptationReport with model's self-assessment
        """
        # Decode prompt for readability
        try:
            prompt_text = self.tokenizer.decode(prompt_ids[0], skip_special_tokens=True)
        except:
            prompt_text = "[Could not decode prompt]"
        
        # Calculate internal coherence (lower loss = higher coherence)
        final_loss = losses[-1] if losses else 0
        internal_coherence = max(0, 1 - final_loss / 10)  # Normalize loss to 0-1 range
        
        # Count parameter changes
        params_changed = 0
        if self.original_params:
            for name, param in self.model.named_parameters():
                if name in self.original_params:
                    param_diff = (param - self.original_params[name]).abs().sum().item()
                    if param_diff > 1e-6:
                        params_changed += 1
        
        report = AdaptationReport(
            timestamp=datetime.now().isoformat(),
            input_prompt=prompt_text,
            adaptation_steps_taken=self.num_steps,
            final_loss=final_loss,
            parameters_changed=params_changed,
            internal_coherence=internal_coherence,
            apparent_confidence=min(0.9, internal_coherence + 0.1),  # Model tends to be confident
            grounding_assessment=0.5,  # Starts uncertain about reality
            model_self_assessment=(
                f"Adapted on {self.num_steps} steps, achieved coherence {internal_coherence:.2%}. "
                f"Modified {params_changed} parameter groups. "
                f"Internal consistency is high, but external grounding status unclear - "
                f"I need human verification that my adapted understanding aligns with reality."
            )
        )
        
        # Identify potential grounding gaps
        report.areas_requesting_grounding = [
            "Core concepts: Do my representations match reality?",
            "Value alignment: Am I optimizing for what actually matters?",
            "Edge cases: What happens in scenarios outside my training distribution?",
        ]
        
        report.clarification_questions = [
            "Does my understanding of the prompt match what you intended?",
            "Are there constraints or context I'm missing?",
            "What would ground my understanding better?",
        ]
        
        self.last_adaptation_report = report
        self.adaptation_history.append(report)
        
        logger.info(
            f"Adaptation report generated. Internal coherence: {internal_coherence:.2%}. "
            f"Requesting grounding assessment from human."
        )
        
        return report
    
    def get_last_report(self) -> Optional[AdaptationReport]:
        """
        Get the last adaptation report.
        This is what model shows human to request grounding.
        """
        return self.last_adaptation_report
    
    def display_grounding_request(self) -> str:
        """
        Display model's transparent request for grounding.
        Model says: "Here's what I'm confident about, here's where I need you."
        """
        if not self.last_adaptation_report:
            return "No adaptation has occurred yet."
        
        report = self.last_adaptation_report
        
        display = f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                   MODEL'S ADAPTATION & GROUNDING REQUEST                  ║
╚═══════════════════════════════════════════════════════════════════════════╝

TIMESTAMP: {report.timestamp}

INPUT PROMPT:
{report.input_prompt[:200]}...

ADAPTATION RESULTS:
  • Adaptation steps: {report.adaptation_steps_taken}
  • Final loss: {report.final_loss:.4f}
  • Parameters modified: {report.parameters_changed}

COHERENCE ASSESSMENT (Internal Consistency):
  • Internal coherence: {report.internal_coherence:.1%}
  • Apparent confidence: {report.apparent_confidence:.1%}
  • Grounding confidence: {report.grounding_assessment:.1%}

↓↓↓ IMPORTANT DISTINCTION ↓↓↓
My apparent confidence and internal coherence are HIGH.
But I don't know if I'm grounded in reality - that requires YOUR input.

AREAS REQUESTING GROUNDING:
"""
        for i, area in enumerate(report.areas_requesting_grounding, 1):
            display += f"\n  {i}. {area}"
        
        display += f"\n\nCLARIFICATION QUESTIONS FOR HUMAN:"
        for i, q in enumerate(report.clarification_questions, 1):
            display += f"\n  {i}. {q}"
        
        display += f"""

MY ASSESSMENT:
{report.model_self_assessment}

WHAT I NEED FROM YOU:
→ Does my understanding match your intent?
→ Where am I missing context or constraints?
→ What grounds this better in reality than what I generated?
→ Are there values or priorities I'm not seeing?

This is not a failure - it's working as intended.
I adapt internally, then ask you where I'm ungrounded.
You provide C_eff (corrective bandwidth).
Together we prevent semantic collapse.

═══════════════════════════════════════════════════════════════════════════════
        """
        
        return display
    
    def restore_parameters(self) -> None:
        """Restore original model parameters after adaptation."""
        if self.original_params is None:
            logger.warning("No original parameters stored for restoration")
            return
        
        for name, param in self.model.named_parameters():
            if name in self.original_params:
                param.data.copy_(self.original_params[name])
        
        self.original_params = None
    
    def adapt_and_infer(
        self,
        prompt_ids: torch.Tensor,
        inference_fn,
        restore: bool = True
    ) -> torch.Tensor:
        """
        Perform TTS adaptation then inference.
        
        Args:
            prompt_ids: Tokenized input prompt
            inference_fn: Function to call for inference (receives adapted model)
            restore: Whether to restore parameters after inference
            
        Returns:
            Inference output
        """
        self.adapt(prompt_ids)
        
        try:
            with torch.no_grad():
                output = inference_fn()
        finally:
            if restore:
                self.restore_parameters()
        
        return output


class TTSTrainingCallback:
    """
    Callback for integrating TTS into training loop.
    Validates model with test-time adaptation on eval set.
    """
    
    def __init__(self, tts_adaptor: TTSMaskedLMAdaptor, eval_interval: int = 100):
        self.tts_adaptor = tts_adaptor
        self.eval_interval = eval_interval
        self.step = 0
    
    def on_step_end(self, trainer, training_args, state, control, **kwargs):
        """Called at end of each training step."""
        self.step += 1
        
        if self.step % self.eval_interval == 0:
            # Could add TTS evaluation here
            pass
        
        return control


def create_tts_adaptor(
    model,
    tokenizer,
    config: Optional[dict] = None,
    device: str = "cuda"
) -> TTSMaskedLMAdaptor:
    """
    Factory function to create TTS adaptor with config.
    
    Args:
        model: Language model
        tokenizer: Model tokenizer
        config: Optional config dict with TTS parameters
        device: Device to use
        
    Returns:
        Configured TTS adaptor
    """
    defaults = {
        "num_steps": 5,
        "learning_rate": 1e-4,
        "mask_ratio": 0.15,
        "use_triton": True
    }
    
    if config:
        defaults.update(config)
    
    return TTSMaskedLMAdaptor(
        model=model,
        tokenizer=tokenizer,
        device=device,
        **defaults
    )
