"""Start (or resume) the model download on the current Colab session.

Kept in the repository so it survives /tmp cleanups. Usage:
  colab --auth=oauth2 exec -s <session> -f tools/colab_download_model.py
"""
import subprocess

cmd = ("mkdir -p /content/models && "
       "nohup bash -lc 'hf download pentacoxian-dev/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF "
       "Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf --local-dir /content/models > /content/download.log 2>&1' &")
subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
print(subprocess.run(
    ["bash", "-lc", "sleep 8; du -sh /content/models 2>/dev/null; "
                    "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"],
    capture_output=True, text=True).stdout.strip())
