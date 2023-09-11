import torch
# import torchvision
import subprocess
import psutil

def command_line_call(_cmd: str, timeout: float = 3600.) -> int:
    with subprocess.Popen(_cmd, shell=True) as process:
        try:
            process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            for child in psutil.Process(process.pid).children(recursive=True):
              child.kill()

            process.kill()
            process.wait()
            raise
        except Exception as ex:
          print(f"exception in subprocess - {ex}")
          process.kill()
          raise
        retcode = process.poll()

    return retcode

def test_torch():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CUDA AVAILABLE: device = '{device}'")
    # PyTorch's versions:
    print("PyTorch Version: ", torch.__version__)
    # print("Torchvision Version: ", torchvision.__version__)
    cmd_ = "nvidia-smi"
    print(cmd_)
    command_line_call(cmd_)


if __name__ == '__main__':
  test_torch()
