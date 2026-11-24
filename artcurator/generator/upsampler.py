"""Official BFL PUP weights + system prompt, using a CUDA-12-compatible runner.

This is NOT the upstream vLLM runtime. Greedy decoding and a local presence
penalty mirror the text-only official settings; numerical equivalence is not
claimed. No generic language model or hand-written caption replaces the PUP.
"""

import json


def caption_messages(folder, prompt):
    system = json.loads((folder / "system_prompts.json").read_text())["t2i"]
    if not isinstance(system, str) or not system.strip():
        raise ValueError("Official PUP t2i system prompt is missing.")
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]


class TransformersUpsampler:
    def __init__(self, folder):
        import torch
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        self.folder = folder
        self.processor = AutoProcessor.from_pretrained(folder, local_files_only=True)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            folder, dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa"
        ).to("cuda:0").eval()

    def __call__(self, prompt):
        import torch
        from transformers import LogitsProcessor, LogitsProcessorList

        messages = caption_messages(self.folder, prompt)
        text = self.processor.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
        inputs = self.processor.tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda:0")
        length = inputs["input_ids"].shape[1]
        if length + 3584 > 8192:
            raise ValueError("PUP prompt exceeds the official 8192-token context budget.")

        class PresencePenalty(LogitsProcessor):
            def __call__(self, ids, scores):
                # vLLM applies presence penalty to generated tokens, not the prompt.
                for row in range(ids.shape[0]):
                    seen = ids[row, length:].unique()
                    scores[row, seen] -= 1.5
                return scores

        tokenizer = self.processor.tokenizer
        stop = [tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
        if any(value is None or value == tokenizer.unk_token_id for value in stop):
            raise ValueError("Official PUP stop tokens are unavailable.")
        with torch.inference_mode():
            ids = self.model.generate(**inputs, do_sample=False, max_new_tokens=3584,
                                      eos_token_id=stop, pad_token_id=tokenizer.pad_token_id or stop[0],
                                      logits_processor=LogitsProcessorList([PresencePenalty()]))
        generated = ids[0, length:]
        if generated.shape[0] == 3584 and int(generated[-1]) not in stop:
            raise RuntimeError("PUP caption hit the token limit; refusing a truncated caption.")
        caption = tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not caption:
            raise RuntimeError("Official PUP produced an empty caption.")
        return caption
