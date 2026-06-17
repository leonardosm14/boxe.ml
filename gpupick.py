"""Escolhe a GPU mais vazia e seta CUDA_VISIBLE_DEVICES. Chamar ANTES de importar
tensorflow. Na máquina compartilhada do lab rodar numa GPU cheia atrasa tudo e falha —
isso evita o conflito automaticamente."""
import os
import subprocess


def pick_and_set_gpu(verbose=True):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader,nounits"], text=True)
        gpus = []
        for line in out.strip().splitlines():
            idx, used, total = (int(x) for x in line.split(","))
            gpus.append((idx, used, total))
        best = min(gpus, key=lambda g: g[1])          # menor memória usada
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best[0])
        if verbose:
            print(f"--> GPU escolhida: {best[0]} (usado {best[1]}/{best[2]} MB) "
                  f"| ocupação: {[(i, u) for i, u, _ in gpus]}")
        return best[0]
    except Exception as e:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        if verbose:
            print(f"--> nvidia-smi falhou ({e}); usando GPU 0")
        return 0
