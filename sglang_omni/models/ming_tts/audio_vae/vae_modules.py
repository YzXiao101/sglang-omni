# SPDX-License-Identifier: Apache-2.0
"""AudioVAE encoder and decoder modules for Ming-Omni-TTS.

Adapted from the official Ming-omni-tts ``audio_tokenizer`` package, with the
existing SGLang-Omni sliding-window cache workaround preserved.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2Model

from sglang_omni.models.ming_tts.audio_vae.istft import ISTFTHead


class StreamingLinearUpsample(nn.Module):
    def __init__(self, scale_factor: int = 4) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.upsampler = nn.Upsample(
            scale_factor=scale_factor,
            mode="linear",
            align_corners=False,
        )

    def forward(
        self,
        x: torch.Tensor | None,
        state: dict[str, Any] | None = None,
        is_last: bool = False,
    ) -> tuple[torch.Tensor | None, dict[str, Any] | None]:
        if state is None:
            state = {"prev_chunk": None, "history_last": None, "is_first": True}

        if x is None and not is_last:
            return None, state
        if x is None:
            raise RuntimeError("StreamingLinearUpsample final chunk requires input")

        if state["is_first"] and is_last:
            out = self.upsampler(x.transpose(1, 2)).transpose(1, 2)
            return out, None

        output_chunks = []
        if state["is_first"]:
            state["prev_chunk"] = x
            state["is_first"] = False
            if not is_last:
                return None, state

        if state["prev_chunk"] is not None:
            previous = state["prev_chunk"].transpose(1, 2)
            lookahead = x[:, :1, :].transpose(1, 2)
            if state["history_last"] is None:
                inp = torch.cat([previous, lookahead], dim=2)
                upsampled = self.upsampler(inp)
                out_prev = upsampled[:, :, : previous.size(2) * self.scale_factor]
            else:
                inp = torch.cat([state["history_last"], previous, lookahead], dim=2)
                upsampled = self.upsampler(inp)
                start = self.scale_factor
                end = start + previous.size(2) * self.scale_factor
                out_prev = upsampled[:, :, start:end]

            output_chunks.append(out_prev.transpose(1, 2))
            state["history_last"] = previous[:, :, -1:]
            state["prev_chunk"] = x

        if is_last:
            previous = state["prev_chunk"].transpose(1, 2)
            inp = torch.cat([state["history_last"], previous], dim=2)
            upsampled = self.upsampler(inp)
            output_chunks.append(upsampled[:, :, self.scale_factor :].transpose(1, 2))
            state = None

        output = torch.cat(output_chunks, dim=1) if output_chunks else None
        return output, state


class Encoder(nn.Module):
    def __init__(
        self,
        encoder_args: dict[str, Any],
        input_dim: int = 320,
        hop_size: int = 320,
        latent_dim: int = 64,
        patch_size: int = -1,
    ) -> None:
        super().__init__()
        config = Qwen2Config.from_dict(config_dict=encoder_args)
        # Transformers 5.x flash attention can fail on this AudioVAE Qwen2 path
        # with s_aux=None. Keep Phase I decode correctness-first.
        config._attn_implementation = "eager"
        self.encoder = Qwen2Model(config)
        self.input_dim = input_dim
        self.hop_size = hop_size
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(input_dim, config.hidden_size, bias=False)
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc3 = nn.Linear(config.hidden_size, latent_dim * 2)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.patch_size = patch_size
        if patch_size != -1:
            config.num_hidden_layers = 4
            self.aggregator = Qwen2Model(config)
            self.cls_embed = nn.Parameter(torch.rand(1, 1, config.hidden_size))
            self.cls_embed.data.normal_(0, 0.02)

    def get_frames(self, waveform: torch.Tensor) -> torch.Tensor:
        num_frames = (waveform.size(-1) + self.hop_size - 1) // self.hop_size
        expected_len = (num_frames - 1) * self.hop_size + self.input_dim
        waveform = F.pad(
            waveform,
            (0, expected_len - waveform.size(-1)),
            value=0.0,
        )
        return waveform.unfold(
            dimension=-1,
            size=self.input_dim,
            step=self.hop_size,
        )

    def pad_patch_insert_cls(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, dim = x.size()
        remainder = x.size(1) % self.patch_size
        pad_num = self.patch_size - remainder if remainder else 0
        x = F.pad(x, (0, 0, 0, pad_num), value=0.0)
        x = x.reshape(-1, self.patch_size, dim)
        x = torch.cat((x, self.cls_embed.expand(x.size(0), -1, -1)), dim=1)
        return x.reshape(batch_size, -1, dim)

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.get_frames(waveform)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.encoder(inputs_embeds=x).last_hidden_state

        if self.patch_size != -1:
            x = self.pad_patch_insert_cls(x)
            x = self.aggregator(inputs_embeds=x).last_hidden_state
            batch_size, _, dim = x.size()
            x = x.reshape(-1, self.patch_size + 1, dim)
            x = x[:, -1:, :].reshape(batch_size, -1, dim)

        return self.fc3(x), waveform.unsqueeze(1)


class Decoder(nn.Module):
    def __init__(
        self,
        decoder_args: dict[str, Any],
        output_dim: int = 320,
        latent_dim: int = 64,
        patch_size: int = -1,
    ) -> None:
        super().__init__()
        config = Qwen2Config.from_dict(config_dict=decoder_args)
        # Transformers 5.x flash attention can fail on this AudioVAE Qwen2 path
        # with s_aux=None. Keep Phase I decode correctness-first.
        config._attn_implementation = "eager"
        self.decoder = Qwen2Model(config)
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(latent_dim, config.hidden_size)
        self.hop_length = output_dim
        self.head = ISTFTHead(
            dim=config.hidden_size,
            n_fft=self.hop_length * 4,
            hop_length=self.hop_length,
            padding="same",
        )
        self.patch_size = patch_size
        if self.patch_size != -1:
            self.upsampling = StreamingLinearUpsample(scale_factor=patch_size)

    def forward(
        self,
        x: torch.Tensor,
        only_semantic_emb: bool = False,
        past_key_values: Any | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, None]:
        if only_semantic_emb:
            raise NotImplementedError(
                "Ming TTS AudioVAE semantic embedding path is currently unsupported; "
                "the 16.8B config sets semantic_module_kwargs=null."
            )
        waveform, _, _ = self.low_level_reconstruct(
            x,
            past_key_values=past_key_values,
            use_cache=use_cache,
            stream_state=(None, None, None),
            last_chunk=True,
        )
        return waveform, None

    def low_level_reconstruct(
        self,
        x: torch.Tensor,
        past_key_values: Any | None = None,
        use_cache: bool = False,
        stream_state: tuple[Any | None, Any | None, Any | None] | None = None,
        last_chunk: bool = False,
    ) -> tuple[
        torch.Tensor,
        tuple[Any | None, Any | None, Any | None],
        Any | None,
    ]:
        if stream_state is None:
            stream_state = (None, None, None)
        upsample_state, audio_buffer, window_buffer = stream_state
        batch_size, device, dtype = x.size(0), x.device, x.dtype

        x = self.fc1(x)
        if self.patch_size != -1:
            if use_cache:
                x, upsample_state = self.upsampling(
                    x,
                    state=upsample_state,
                    is_last=last_chunk,
                )
                if x is None:
                    stream_state = (upsample_state, audio_buffer, window_buffer)
                    empty = torch.empty(
                        batch_size,
                        1,
                        0,
                        device=device,
                        dtype=dtype,
                    )
                    return empty, stream_state, past_key_values
            else:
                x = self.upsampling.upsampler(x.transpose(1, 2)).transpose(1, 2)

        hidden_chunks = []
        if (
            use_cache
            and getattr(self.decoder.config, "sliding_window", None) is not None
        ):
            window = self.decoder.config.sliding_window
            target_len = window - 1
            if past_key_values is None:
                past_len = 0
            elif hasattr(past_key_values, "get_seq_length"):
                past_len = past_key_values.get_seq_length()
            elif isinstance(past_key_values, tuple) and past_key_values:
                past_len = past_key_values[0][0].shape[-2]
            else:
                past_len = 0

            curr_len = x.shape[1]
            if past_len < target_len and past_len + curr_len >= window:
                fill_len = target_len - past_len
                outputs = self.decoder(
                    inputs_embeds=x[:, :fill_len, :],
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )
                hidden_chunks.append(outputs.last_hidden_state)
                past_key_values = outputs.past_key_values
                x = x[:, fill_len:, :]

        outputs = self.decoder(
            inputs_embeds=x,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_chunks.append(outputs.last_hidden_state)
        past_key_values = outputs.past_key_values
        hidden = (
            torch.cat(hidden_chunks, dim=1)
            if len(hidden_chunks) > 1
            else hidden_chunks[0]
        )

        waveform, _, audio_buffer, window_buffer = self.head(
            hidden,
            streaming=use_cache,
            audio_buffer=audio_buffer,
            window_buffer=window_buffer,
            last_chunk=last_chunk,
        )
        return waveform, (upsample_state, audio_buffer, window_buffer), past_key_values


__all__ = ["Decoder", "Encoder", "StreamingLinearUpsample"]
