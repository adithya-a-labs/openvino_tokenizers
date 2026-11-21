"""Create the OVMS model directory using the branch's tokenizer export."""
import os
from pathlib import Path

import openvino as ov
from huggingface_hub import snapshot_download
from openvino_tokenizers import convert_tokenizer
from transformers import AutoTokenizer

model_dir = Path(os.environ["MODEL_DIR"])
snapshot_download(
    "asahi-jp/plamo-3-translate-0.8b-int4-cw-ov",
    revision=os.environ["MODEL_REVISION"],
    local_dir=model_dir,
    allow_patterns=["openvino_model.xml", "openvino_model.bin", "config.json", "generation_config.json"],
)
reference_path = Path("pfn-tokenizer")
snapshot_download(
    "pfnet/plamo-3-translate-0.8b",
    revision=os.environ["PFN_REVISION"],
    local_dir=reference_path,
    allow_patterns=["tokenizer_config.json", "tokenization_plamo.py", "special_tokens_map.json",
                    "tokenizer.jsonl", "chat_template.jinja", "config.json"],
)
pfn = AutoTokenizer.from_pretrained(str(reference_path), trust_remote_code=True, local_files_only=True)
tokenizer, detokenizer = convert_tokenizer(pfn, with_detokenizer=True, add_special_tokens=False)
tokenizer_path = model_dir / "openvino_tokenizer.xml"
ov.save_model(tokenizer, tokenizer_path)
ov.save_model(detokenizer, model_dir / "openvino_detokenizer.xml")
assert 'type="PlamoTokenizer"' in tokenizer_path.read_text(encoding="utf-8")
for name in ["tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json"]:
    (model_dir / name).write_bytes((reference_path / name).read_bytes())
print("Generated stock PFN tokenizer and detokenizer IRs with", ov.get_version())
