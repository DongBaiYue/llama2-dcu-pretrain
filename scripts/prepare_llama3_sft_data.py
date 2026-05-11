#!/usr/bin/env python3
"""将 erniekit SFT 数据转换为 Llama-3 chat template 格式

用法: python3 scripts/prepare_llama3_sft_data.py
输入: data/sft/school_math_0.25M/{train,dev}.json
输出: data/sft/school_math_0.25M/{train,dev}_llama3.json
"""

import json

BOS = "<|begin_of_text|>"
START_H = "<|start_header_id|>"
END_H = "<|end_header_id|>"
EOT = "<|eot_id|>"

for split in ["train", "dev"]:
    src = f"data/sft/school_math_0.25M/{split}.json"
    dst = f"data/sft/school_math_0.25M/{split}_llama3.json"
    out = []
    with open(src) as f:
        for line in f:
            item = json.loads(line.strip())
            prompt = f"{BOS}{START_H}user{END_H}\n\n{item['src']}{EOT}{START_H}assistant{END_H}\n\n{item['tgt']}{EOT}"
            out.append({"src": prompt, "tgt": item["tgt"]})
    with open(dst, "w") as f:
        for item in out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"{split}: {len(out)} samples -> {dst}")
