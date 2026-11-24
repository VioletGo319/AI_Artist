import json
import sys
from types import SimpleNamespace

import pytest

from artcurator.generator.preflight import check_processors


@pytest.fixture
def processors(tmp_path, monkeypatch):
    (tmp_path / "pup").mkdir()
    (tmp_path / "pup" / "system_prompts.json").write_text(json.dumps({"t2i": "Official instructions"}))
    calls = []
    tokens = SimpleNamespace(length=400, stop=42)

    class Tokenizer:
        unk_token_id = -1

        def __call__(self, text, **kwargs):
            assert text == "rendered template"
            assert kwargs == {"return_tensors": "pt", "add_special_tokens": False}
            return {"input_ids": SimpleNamespace(shape=(1, tokens.length))}

        def convert_tokens_to_ids(self, token):
            assert token in ("<|im_end|>", "<|endoftext|>")
            return tokens.stop

    class Processor:
        tokenizer = Tokenizer()

        def apply_chat_template(self, messages, **kwargs):
            assert messages[0] == {"role": "system", "content": "Official instructions"}
            assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
            return "rendered template"

    def load(path, **kwargs):
        assert kwargs == {"local_files_only": True}
        calls.append(path)
        return Processor()

    auto = SimpleNamespace(from_pretrained=load)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoProcessor=auto))
    return tmp_path, calls, tokens, auto


def test_cached_processors_are_loaded_offline_and_template_is_exercised(processors):
    root, calls, _, _ = processors
    assert check_processors(root)["status"] == "passed"
    assert calls == [root / "flux-3-image-klein" / "text_encoder", root / "pup"]


def test_missing_processor_dependency_fails_before_gpu_allocation(processors):
    root, _, _, auto = processors
    def fail(*args, **kwargs):
        raise ImportError("Qwen3VLVideoProcessor requires Torchvision")
    auto.from_pretrained = fail
    with pytest.raises(ImportError, match="Torchvision"):
        check_processors(root)


@pytest.mark.parametrize("field,value,error", [("length", 5000, "context budget"), ("stop", None, "stop tokens")])
def test_bad_processor_configuration_fails_closed(processors, field, value, error):
    root, _, tokens, _ = processors
    setattr(tokens, field, value)
    with pytest.raises(RuntimeError, match=error):
        check_processors(root)
