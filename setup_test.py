import sys
import os
import platform
import shutil

# Terminal colors for CLI output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def colorize(text, color):
    if sys.stdout.isatty():
        return f"{color}{text}{Colors.ENDC}"
    return text

def detect_hardware():
    system = platform.system()
    machine = platform.machine()

    hardware_info = {
        "system": system,
        "machine": machine,
        "gpu_type": None,
        "recommended_jax": "pip install --upgrade jax"
    }

    if system == "Darwin":
        if machine == "arm64":
            hardware_info["gpu_type"] = "Apple Silicon GPU (Metal)"
            hardware_info["recommended_jax"] = "pip install --upgrade jax"
        else:
            hardware_info["gpu_type"] = "Intel Mac GPU"
            hardware_info["recommended_jax"] = "pip install --upgrade jax"
    elif system == "Linux":
        # Check for NVIDIA GPU
        if shutil.which("nvidia-smi") is not None:
            hardware_info["gpu_type"] = "NVIDIA GPU (CUDA detected)"
            hardware_info["recommended_jax"] = 'pip install --upgrade "jax[cuda12]"'
        elif os.path.exists("/proc/driver/nvidia"):
            hardware_info["gpu_type"] = "NVIDIA GPU (CUDA driver files detected)"
            hardware_info["recommended_jax"] = 'pip install --upgrade "jax[cuda12]"'
        # Check for AMD GPU / ROCm
        elif shutil.which("rocminfo") is not None or os.path.exists("/opt/rocm"):
            hardware_info["gpu_type"] = "AMD GPU (ROCm detected)"
            hardware_info["recommended_jax"] = 'pip install --upgrade "jax[rocm7-local]"'
        else:
            hardware_info["gpu_type"] = "CPU Only"
            hardware_info["recommended_jax"] = "pip install --upgrade jax"
    elif system == "Windows":
        if shutil.which("nvidia-smi") is not None:
            hardware_info["gpu_type"] = "NVIDIA GPU (CUDA detected on Windows)"
            # Official docs suggest running in WSL2 for GPU, or CPU only on native Windows.
            hardware_info["recommended_jax"] = "pip install --upgrade jax\n   (For GPU acceleration, JAX recommends using WSL2: https://docs.jax.dev/en/latest/installation.html)"
        else:
            hardware_info["gpu_type"] = "CPU Only"
            hardware_info["recommended_jax"] = "pip install --upgrade jax"

    return hardware_info

def check_requirements():
    reqs = ["matplotlib", "numpy", "jax"]

    print(colorize("=== Scipy Tutorial 2026 Environment Check ===", Colors.HEADER + Colors.BOLD))
    print(f"Python version: {sys.version.split()[0]}")

    hw = detect_hardware()
    print(f"Platform: {hw['system']} ({hw['machine']})")
    if hw['gpu_type']:
        print(f"Hardware detected: {colorize(hw['gpu_type'], Colors.OKCYAN)}")
    print("-" * 45)

    missing_libs = []
    installed_libs = {}

    for req in reqs:
        base_name = req
        try:
            mod = __import__(base_name)
            version = getattr(mod, "__version__", "unknown")
            installed_libs[base_name] = version
            print(f"[{colorize('OK', Colors.OKGREEN)}] {base_name:<15} (Version: {version})")
        except ImportError:
            missing_libs.append(base_name)
            print(f"[{colorize('MISSING', Colors.FAIL)}] {base_name:<15}")

    print("-" * 45)

    if missing_libs:
        print(colorize("\n[!] Environment check FAILED. Please install missing libraries:", Colors.WARNING + Colors.BOLD))

        # Build recommendation instructions
        cmd_suggestions = []
        for lib in missing_libs:
            if lib == "jax":
                cmd_suggestions.append(f"# To install JAX for your {hw['gpu_type'] if hw['gpu_type'] else 'system'}:\n  {hw['recommended_jax']}")
            else:
                cmd_suggestions.append(f"# To install {lib}:\n  pip install --upgrade {lib}")

        print("\n" + "\n\n".join(cmd_suggestions) + "\n")
        return False
    else:
        print(colorize("\n[+] All required packages are present! Testing JAX computation...", Colors.OKGREEN + Colors.BOLD))
        try:
            import jax
            import jax.numpy as jnp

            # Print active devices
            devices = jax.devices()
            print(f"JAX Backend Devices: {colorize(str(devices), Colors.OKBLUE)}")

            # Perform a simple check computation
            x = jnp.array([1.0, 2.0, 3.0])
            y = x * 2.0
            # Trigger JIT to verify compilation pipeline is functional
            @jax.jit
            def square(val):
                return val ** 2
            _ = square(x)

            print(colorize("[+] JAX computation check passed successfully!", Colors.OKGREEN + Colors.BOLD))
            return True
        except Exception as e:
            print(colorize(f"[!] JAX was imported but computation check failed: {e}", Colors.FAIL + Colors.BOLD))
            return False

if __name__ == "__main__":
    success = check_requirements()
    sys.exit(0 if success else 1)
