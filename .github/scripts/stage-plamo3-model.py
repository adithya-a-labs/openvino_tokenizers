"""Create the OVMS model directory using the branch's tokenizer export."""
import json
import os
from pathlib import Path

import openvino as ov
from huggingface_hub import snapshot_download
from openvino.preprocess import PrePostProcessor
from openvino_tokenizers import convert_tokenizer
from transformers import AutoTokenizer

model_dir = Path(os.environ["MODEL_DIR"])
snapshot_download(
    "asahi-jp/plamo-3-translate-0.8b-int4-cw-ov",
    revision=os.environ["MODEL_REVISION"],
    local_dir=model_dir,
    allow_patterns=["openvino_model.xml", "openvino_model.bin", "config.json", "generation_config.json"],
)
model_path = model_dir / "openvino_model.xml"
model = ov.Core().read_model(model_path)
if model.output("logits").get_element_type() == ov.Type.bf16:
    # This GenAI build reads logits through a float pointer. Preserve the model
    # computation and convert its public result tensor only.
    adapter = PrePostProcessor(model)
    adapter.output(0).tensor().set_element_type(ov.Type.f32)
    adapted = adapter.build()
    adapted.output(0).tensor.set_names({"logits"})
    assert adapted.output("logits").get_element_type() == ov.Type.f32
    adapted_path = model_dir / "fp32_logits.xml"
    ov.save_model(adapted, adapted_path, compress_to_fp16=False)
    adapted_path.replace(model_path)
    adapted_path.with_suffix(".bin").replace(model_path.with_suffix(".bin"))
    print("Adapted the public logits from BF16 to FP32 for OVMS 2026.5")

reference_path = Path("pfn-tokenizer")
snapshot_download(
    "pfnet/plamo-3-translate-0.8b",
    revision=os.environ["PFN_REVISION"],
    local_dir=reference_path,
    allow_patterns=["tokenizer_config.json", "tokenization_plamo.py", "special_tokens_map.json",
                    "tokenizer.jsonl", "chat_template.jinja", "config.json"],
)
pfn = AutoTokenizer.from_pretrained(str(reference_path), trust_remote_code=True, local_files_only=True)
context_length = json.loads((model_dir / "config.json").read_text())["max_position_embeddings"]
tokenizer, detokenizer = convert_tokenizer(
    pfn, with_detokenizer=True, add_special_tokens=False, max_length=context_length,
)
tokenizer_path = model_dir / "openvino_tokenizer.xml"
ov.save_model(tokenizer, tokenizer_path)
ov.save_model(detokenizer, model_dir / "openvino_detokenizer.xml")
assert 'type="PlamoTokenizer"' in tokenizer_path.read_text(encoding="utf-8")
for name in ["tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json"]:
    (model_dir / name).write_bytes((reference_path / name).read_bytes())
config_path = model_dir / "tokenizer_config.json"
tokenizer_config = json.loads(config_path.read_text())
tokenizer_config["model_max_length"] = context_length
config_path.write_text(json.dumps(tokenizer_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("Generated stock PFN tokenizer and detokenizer IRs with", ov.get_version(), "and context", context_length)
