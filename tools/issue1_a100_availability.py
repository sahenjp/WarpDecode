"""One bounded A100 allocation check; never falls back to another GPU."""
import modal

app = modal.App("warpdecode-issue1-a100-availability")


@app.function(gpu="A100-80GB", timeout=60)
def check():
    import subprocess
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        text=True,
    ).strip()


@app.local_entrypoint()
def main():
    print(check.remote())
