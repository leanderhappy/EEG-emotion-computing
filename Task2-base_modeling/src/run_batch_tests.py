import subprocess
from pathlib import Path


current_dir = Path(__file__).parent
script_path = current_dir / "base-modeling-merged.py"

for data_root in ["task1", "official"]:
    for model_type in ["svm", "rf", "mlp"]:
        for task in ["binary", "threeclass"]:
            for split in ["random", "loto"]:
                for regression in [False, True]:
                    cmd = [
                        "python", str(script_path),
                        "--data_root", data_root,
                        "--model_type", model_type,
                        "--task", task,
                        "--split", split,
                    ]

                    if regression:
                        cmd.append("--regression")
                    
                    print("Run:", " ".join(cmd))
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    print(result.stdout)  # 输出结果
                    print(result.returncode)  # 返回码
