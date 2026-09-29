"""Space V2 alignment stopping, adapted only for HF 5.2 hooks and cleanup.

Source: ResembleAI/Chatterbox-Multilingual-TTS Space revision
c612a94210c0af895d6b649850d15b7ccca36594, alignment_stream_analyzer.py
Original SHA256: 206e17cbd1ae713f9e814de35f05eb66e67f8140d3e80fb3204fa3d4ae150d92
The alignment/EOS math in step() is retained. Hooks require eager attention,
fail closed on missing weights, and are removed after every inference.
"""

# Copyright (c) 2025 Resemble AI
# Author: John Meade, Jeremy Hsu
# MIT License
import logging
import torch
from dataclasses import dataclass
from functools import wraps


logger = logging.getLogger(__name__)


LLAMA_ALIGNED_HEADS = [(12, 15), (13, 11), (9, 2)]


@dataclass
class AlignmentAnalysisResult:
    # was this frame detected as being part of a noisy beginning chunk with potential hallucinations?
    false_start: bool
    # was this frame detected as being part of a long tail with potential hallucinations?
    long_tail: bool
    # was this frame detected as repeating existing text content?
    repetition: bool
    # was the alignment position of this frame too far from the previous frame?
    discontinuity: bool
    # has inference reached the end of the text tokens? eg, this remains false if inference stops early
    complete: bool
    # approximate position in the text token sequence. Can be used for generating online timestamps.
    position: int


class AlignmentStreamAnalyzer:
    def __init__(self, tfmr, queue, text_tokens_slice, alignment_layer_idx=9, eos_idx=0):
        """
        Some transformer TTS models implicitly solve text-speech alignment in one or more of their self-attention
        activation maps. This module exploits this to perform online integrity checks which streaming.
        A hook is injected into the specified attention layer, and heuristics are used to determine alignment
        position, repetition, etc.

        NOTE: currently requires no queues.
        """
        # self.queue = queue
        self.text_tokens_slice = (i, j) = text_tokens_slice
        self.eos_idx = eos_idx
        self.alignment = torch.zeros(0, j-i)
        # self.alignment_bin = torch.zeros(0, j-i)
        self.curr_frame_pos = 0
        self.text_position = 0

        self.started = False
        self.started_at = None

        self.complete = False
        self.completed_at = None
        
        # Track generated tokens for repetition detection
        self.generated_tokens = []

        # Using `output_attentions=True` is incompatible with optimized attention kernels, so
        # using it for all layers slows things down too much. We can apply it to just one layer
        # by intercepting the kwargs and adding a forward hook (credit: jrm)
        self.last_aligned_attns = [None] * len(LLAMA_ALIGNED_HEADS)
        self._handles = []
        if tfmr.config._attn_implementation != "eager":
            raise RuntimeError("Space alignment requires eager attention")
        try:
            for i, (layer_idx, head_idx) in enumerate(LLAMA_ALIGNED_HEADS):
                self._add_attention_spy(tfmr, i, layer_idx, head_idx)
        except Exception:
            self.close()
            raise

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.last_aligned_attns = [None] * len(LLAMA_ALIGNED_HEADS)

    def _add_attention_spy(self, tfmr, buffer_idx, layer_idx, head_idx):
        """
        Adds a forward hook to a specific attention layer to collect outputs.
        """
        def attention_forward_hook(module, input, output):
            """
            Transformers 5.2 LlamaAttention.forward returns (attn_output, attn_weights).
            NOTE:
            - When `output_attentions=True`, `LlamaSdpaAttention.forward` calls `LlamaAttention.forward`.
            - `attn_output` has shape [B, H, T0, T0] for the 0th entry, and [B, H, 1, T0+i] for the rest i-th.
            """
            if (not isinstance(output, tuple) or len(output) < 2
                    or not torch.is_tensor(output[1]) or output[1].ndim != 4
                    or output[1].shape[0] < 1 or output[1].shape[1] <= head_idx):
                raise RuntimeError("Space alignment attention weights are unavailable")
            step_attention = output[1].detach().cpu()
            self.last_aligned_attns[buffer_idx] = step_attention[0, head_idx]

        target_layer = tfmr.layers[layer_idx].self_attn
        self._handles.append(target_layer.register_forward_hook(attention_forward_hook))

    def step(self, logits, next_token=None):
        """
        Emits an AlignmentAnalysisResult into the output queue, and potentially modifies the logits to force an EOS.
        """
        # extract approximate alignment matrix chunk (1 frame at a time after the first chunk)
        if any(value is None for value in self.last_aligned_attns):
            raise RuntimeError("Space alignment attention hooks did not run")
        aligned_attn = torch.stack(self.last_aligned_attns).mean(dim=0) # (N, N)
        # Each decoder step must provide fresh weights from all three heads.
        self.last_aligned_attns = [None] * len(LLAMA_ALIGNED_HEADS)
        i, j = self.text_tokens_slice
        if self.curr_frame_pos == 0:
            # first chunk has conditioning info, text tokens, and BOS token
            A_chunk = aligned_attn[j:, i:j].clone().cpu() # (T, S)
        else:
            # subsequent chunks have 1 frame due to KV-caching
            A_chunk = aligned_attn[:, i:j].clone().cpu() # (1, S)

        # TODO: monotonic masking; could have issue b/c spaces are often skipped.
        A_chunk[:, self.curr_frame_pos + 1:] = 0


        self.alignment = torch.cat((self.alignment, A_chunk), dim=0)

        A = self.alignment
        T, S = A.shape

        # update position
        cur_text_posn = A_chunk[-1].argmax()
        discontinuity = not(-4 < cur_text_posn - self.text_position < 7) # NOTE: very lenient!
        if not discontinuity:
            self.text_position = cur_text_posn

        # Hallucinations at the start of speech show up as activations at the bottom of the attention maps!
        # To mitigate this, we just wait until there are no activations far off-diagonal in the last 2 tokens,
        # and there are some strong activations in the first few tokens.
        false_start = (not self.started) and (A[-2:, -2:].max() > 0.1 or A[:, :4].max() < 0.5)
        self.started = not false_start
        if self.started and self.started_at is None:
            self.started_at = T

        # Is generation likely complete?
        self.complete = self.complete or self.text_position >= S - 3
        if self.complete and self.completed_at is None:
            self.completed_at = T

        # NOTE: EOS rarely assigned activations, and second-last token is often punctuation, so use last 3 tokens.
        # NOTE: due to the false-start behaviour, we need to make sure we skip activations for the first few tokens.
        last_text_token_duration = A[15:, -3:].sum()

        # Activations for the final token that last too long are likely hallucinations.
        long_tail = self.complete and (A[self.completed_at:, -3:].sum(dim=0).max() >= 5) # 200ms

        # If there are activations in previous tokens after generation has completed, assume this is a repetition error.
        # Space's prefix slice is empty for <=5 text tokens; there is no
        # earlier text to repeat. Avoid an empty max for these valid inputs.
        alignment_repetition = self.complete and S > 5 and (A[self.completed_at:, :-5].max(dim=1).values.sum() > 5)
        
        # Track generated tokens for repetition detection
        if next_token is not None:
            # Convert tensor to scalar if needed
            if isinstance(next_token, torch.Tensor):
                token_id = next_token.item() if next_token.numel() == 1 else next_token.view(-1)[0].item()
            else:
                token_id = next_token
            self.generated_tokens.append(token_id)
            
            # Keep only last 8 tokens to prevent memory issues
            if len(self.generated_tokens) > 8:
                self.generated_tokens = self.generated_tokens[-8:]
            
        # Check for excessive token repetition (3x same token in a row)
        token_repetition = (
            # self.complete and 
            len(self.generated_tokens) >= 3 and
            len(set(self.generated_tokens[-2:])) == 1
        )
        
        if token_repetition:
            repeated_token = self.generated_tokens[-1]
            logger.warning(f"🚨 Detected 2x repetition of token {repeated_token}")
            
        # Suppress EoS to prevent early termination
        if cur_text_posn < S - 3 and S > 5:  # Only suppress if text is longer than 5 tokens
            logits[..., self.eos_idx] = -2**15

        # If a bad ending is detected, force emit EOS by modifying logits
        # NOTE: this means logits may be inconsistent with latents!
        if long_tail or alignment_repetition or token_repetition:
            logger.warning(f"forcing EOS token, {long_tail=}, {alignment_repetition=}, {token_repetition=}")
            # (±2**15 is safe for all dtypes >= 16bit)
            logits = -(2**15) * torch.ones_like(logits)
            logits[..., self.eos_idx] = 2**15

        self.curr_frame_pos += 1
        return logits


def alignment_guard(inference):
    """Scope the Space analyzer to one request, including failed generations."""
    @wraps(inference)
    def guarded(self, *args, **kwargs):
        self._space_alignment = None
        try:
            return inference(self, *args, **kwargs)
        finally:
            if self._space_alignment is not None:
                self._space_alignment.close()
            self._space_alignment = None
    return guarded


def self_test():
    """Exercise real HF 5.2 eager heads + KV cache without downloading weights."""
    from transformers import LlamaConfig, LlamaModel
    import transformers

    assert transformers.__version__ == "5.2.0"
    torch.set_num_threads(1)
    model = LlamaModel(LlamaConfig(
        hidden_size=32, intermediate_size=48, num_hidden_layers=14,
        num_attention_heads=16, num_key_value_heads=16, vocab_size=32,
    )).eval()
    model.set_attn_implementation("eager")
    # HF 5.2 installs its own persistent output-capture hooks on first forward.
    # Establish that baseline before checking our per-request hook lifetime.
    with torch.inference_mode():
        model(inputs_embeds=torch.zeros(2, 7, 32), use_cache=True,
              output_attentions=True, output_hidden_states=True)
    baseline = [len(layer.self_attn._forward_hooks) for layer in model.layers]
    for _ in range(2):
        analyzer = AlignmentStreamAnalyzer(model, None, (0, 6), eos_idx=9)
        try:
            with torch.inference_mode():
                output = model(inputs_embeds=torch.zeros(2, 7, 32), use_cache=True,
                               output_attentions=True, output_hidden_states=True)
                assert all(a is not None and a.shape == (7, 7) for a in analyzer.last_aligned_attns)
                assert all(torch.count_nonzero(torch.triu(a, diagonal=1)) == 0
                           for a in analyzer.last_aligned_attns)
                assert output.past_key_values.get_seq_length() == 7
                analyzer.step(torch.zeros(1, 12), next_token=1)
                output = model(inputs_embeds=torch.zeros(2, 1, 32), use_cache=True,
                               past_key_values=output.past_key_values, output_attentions=True)
                assert all(a is not None and a.shape == (1, 8) for a in analyzer.last_aligned_attns)
                assert output.past_key_values.get_seq_length() == 8
                analyzer.step(torch.zeros(1, 12), next_token=2)
        finally:
            analyzer.close()
        assert [len(layer.self_attn._forward_hooks) for layer in model.layers] == baseline
    print("Space alignment eager/cache/hook smoke passed")


if __name__ == "__main__":
    self_test()
