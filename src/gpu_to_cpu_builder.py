import sys
import subprocess
import argparse
import os
import tempfile
import re

cgeist = "/mnt/pvc/Polygeist/build/bin/cgeist"
polygeist_opt = "/mnt/pvc/Polygeist/build/bin/polygeist-opt"
mlir_opt = "/mnt/pvc/Enzyme-JAX/bazel-bin/enzymexlamlir-opt"
mlir_translate = "/mnt/pvc/Enzyme-JAX/install/execroot/__main__/bazel-out/k8-opt/bin/external/llvm-project/mlir/mlir-translate.runfiles/__main__/external/llvm-project/mlir/mlir-translate"
clang = "/mnt/pvc/Enzyme-JAX/install/execroot/__main__/bazel-out/k8-opt/bin/external/llvm-project/clang/clang.runfiles/llvm-project/clang/clang"

# STRICTLY LLVM's OpenMP runtime for __kmpc compatibility
openmp_flag = "-fopenmp=libomp"

def sanitize_cuda_source(input_path, output_path):
    """
    Intelligently rewrites CUDA API calls into standard C equivalents in the source code.
    This prevents Polygeist from generating double-pointer indirection and unkillable MLIR casts.
    """
    with open(input_path, 'r') as f:
        content = f.read()

    # 1. cudaMalloc((void**)&d_A, size) -> d_A = (decltype(d_A))malloc(size)
    content = re.sub(r'cudaMalloc\s*\(\s*(?:\(\s*void\s*\*\*\s*\))?\s*&?\s*([a-zA-Z0-9_]+)\s*,\s*(.*?)\s*\)', 
                     r'\1 = (decltype(\1))malloc(\2)', content)

    # 2. cudaMemcpy(d_C, d_A, size, cudaMemcpy...) -> memcpy(d_C, d_A, size)
    content = re.sub(r'cudaMemcpy\s*\(\s*([a-zA-Z0-9_]+)\s*,\s*([a-zA-Z0-9_]+)\s*,\s*(.*?)\s*,\s*[a-zA-Z0-9_]+\s*\)', 
                     r'memcpy(\1, \2, \3)', content)

    # 3. cudaFree(d_A) -> free(d_A)
    content = re.sub(r'cudaFree\s*\(\s*([a-zA-Z0-9_]+)\s*\)', r'free(\1)', content)

    # 4. Erase GPU Timing APIs
    content = re.sub(r'cudaEvent_t\s+[^;]+;', '', content)
    content = re.sub(r'cudaEventCreate\s*\([^)]+\);?', '', content)
    content = re.sub(r'cudaEventRecord\s*\([^)]+\);?', '', content)
    content = re.sub(r'cudaEventSynchronize\s*\([^)]+\);?', '', content)
    content = re.sub(r'cudaEventDestroy\s*\([^)]+\);?', '', content)
    content = re.sub(r'cudaDeviceSynchronize\s*\(\s*\);?', '', content)

    # 5. cudaEventElapsedTime(&time, ...) -> time = 0.0f
    content = re.sub(r'cudaEventElapsedTime\s*\(\s*&?([a-zA-Z0-9_]+)\s*,[^)]+\)', r'\1 = 0.0f', content)

    headers = "#include <stdlib.h>\n#include <string.h>\n"
    with open(output_path, 'w') as f:
        f.write(headers + content)

def extract_signature(func_decl):
    sig = func_decl.split("attributes")[0].strip()
    if not sig.endswith(")"):
        sig = sig.rsplit("{", 1)[0].strip()
    return sig

def split_and_prepare_mlir(input_file, host_file, kernel_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    host_lines = ["module {\n"]
    kernel_lines = ["module {\n"]

    start_idx = 0
    for i, line in enumerate(lines):
        if line.lstrip().startswith("func.func") or line.lstrip().startswith("llvm.func") or line.lstrip().startswith("llvm.mlir.global") or line.lstrip().startswith("memref.global"):
            start_idx = i
            break

    brace_count = 0
    current_block = []
    aliases = []

    for idx, line in enumerate(lines[start_idx:]):
        if line.strip() == "}" and brace_count == 0:
            aliases = lines[start_idx + idx + 1:]
            break

        current_block.append(line)
        brace_count += line.count("{") - line.count("}")

        if brace_count == 0 and current_block:
            block_text = "".join(current_block)

            if "polygeist.device_only_func" in block_text:
                if "__device_stub__" in block_text:
                    pass
                else:
                    sig = extract_signature(current_block[0])
                    if "private " not in sig:
                        sig = sig.replace("func.func ", "func.func private ").replace("llvm.func ", "llvm.func ")
                    host_lines.append(f"  {sig}\n")

                    kernel_lines.append(current_block[0].replace("private ", ""))
                    kernel_lines.append(f"    %c1 = arith.constant 1 : index\n")
                    kernel_lines.append(f"    %bx = arith.constant 16 : index\n")
                    kernel_lines.append(f"    %gx = arith.constant 64 : index\n")
                    kernel_lines.append(f"    gpu.launch blocks(%grid_id_x, %grid_id_y, %grid_id_z) in (%grid_x = %gx, %grid_y = %gx, %grid_z = %c1)\n")
                    kernel_lines.append(f"               threads(%thread_id_x, %thread_id_y, %thread_id_z) in (%block_x = %bx, %block_y = %bx, %block_z = %c1) {{\n")

                    skip_next = False
                    for b_line in current_block[1:]:
                        b_line = b_line.replace("nvvm.barrier0", "gpu.barrier")

                        if "scf.if" in b_line:
                            skip_next = True
                            continue
                        if skip_next and b_line.strip() == "}":
                            skip_next = False
                            continue

                        if b_line.strip() == "return" or b_line.strip() == "llvm.return":
                            kernel_lines.append("      gpu.terminator\n    }\n")
                        kernel_lines.append(b_line)
            else:
                skip_braces = 0
                for h_line in current_block:
                    if "gpu.launch " in h_line and "{" in h_line:
                        skip_braces += 1
                        continue
                    if "gpu.terminator" in h_line:
                        continue
                    if skip_braces > 0 and h_line.strip() == "}":
                        skip_braces -= 1
                        continue

                    if "__device_stub__" in h_line:
                        h_line = h_line.replace("__device_stub__", "")

                    if " call @" in h_line:
                        h_line = h_line.replace(" call @", " func.call @")
                    elif h_line.lstrip().startswith("call @"):
                        h_line = h_line.replace("call @", "func.call @", 1)
                        
                    if h_line.lstrip().startswith("return ") or h_line.strip() == "return":
                        h_line = h_line.replace("return", "func.return", 1)

                    host_lines.append(h_line)

            current_block = []

    host_lines.append("}\n")
    kernel_lines.append("}\n")

    host_lines.extend(aliases)
    kernel_lines.extend(aliases)

    with open(host_file, 'w') as f: f.writelines(host_lines)
    with open(kernel_file, 'w') as f: f.writelines(kernel_lines)
    print(f"[+] Split complete: Host and Kernel MLIR generated in temp directory.")

def run_cmd(cmd):
    print(f"[+] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[-] Error executing command:\n{result.stderr}")
        sys.exit(1)

def compile_pipeline(args):
    final_output_path = os.path.abspath(args.output)

    with tempfile.TemporaryDirectory(dir="/tmp", prefix="enzyme_build_") as tmpdir:
        print(f"[*] Created unique temporary workspace: {tmpdir}")

        input_mlir = args.input

        if args.input.endswith(".cu") or args.input.endswith(".cu.cc"):
            print("[+] Detected CUDA file! Sanitizing source via Regex...")
            sanitized_cu = os.path.join(tmpdir, "sanitized_input.cu")
            sanitize_cuda_source(args.input, sanitized_cu)
            
            input_mlir = os.path.join(tmpdir, "raw_polygeist.mlir")
            run_cmd([cgeist, sanitized_cu, "-O3", "-I.", "-I/usr/local/cuda/include", "--cuda-gpu-arch=sm_80", "-S", "-o", input_mlir])

        host_mlir = os.path.join(tmpdir, "host.mlir")
        kernel_mlir = os.path.join(tmpdir, "kernel.mlir")
        kernel_llvm_mlir = os.path.join(tmpdir, "kernel_llvm.mlir")
        host_llvm_mlir = os.path.join(tmpdir, "host_llvm.mlir")
        kernel_ll = os.path.join(tmpdir, "kernel.ll")
        host_ll = os.path.join(tmpdir, "host.ll")
        host_o = os.path.join(tmpdir, "host.o")
        kernel_o = os.path.join(tmpdir, "kernel.o")

        split_and_prepare_mlir(input_mlir, host_mlir, kernel_mlir)

        pass_opts = f"reg-bit-width={args.vector_width} unroll-factor={args.unroll_factor} max-threads={args.threads}"

        print("[+] Compiling Kernel MLIR...")
        kernel_opt_cmd = [
            mlir_opt, kernel_mlir,
            "--lower-affine",
            f"--cuda-to-hierarchical-parallel={pass_opts}",
            "--convert-scf-to-openmp", "--convert-scf-to-cf", "--convert-openmp-to-llvm",
            "--convert-vector-to-llvm", "--convert-index-to-llvm", "--convert-arith-to-llvm",
            "--finalize-memref-to-llvm", "--convert-func-to-llvm", "--reconcile-unrealized-casts"
        ]
        with open(kernel_llvm_mlir, "w") as f:
            subprocess.run(kernel_opt_cmd, stdout=f, check=True)

        print("[+] Lowering Polygeist dialects in Host MLIR...")
        host_intermediate_mlir = os.path.join(tmpdir, "host_intermediate.mlir")
        host_polygeist_cmd = [
            polygeist_opt, host_mlir,
            "--convert-polygeist-to-llvm"
        ]
        with open(host_intermediate_mlir, "w") as f:
            subprocess.run(host_polygeist_cmd, stdout=f, check=True)

        print("[+] Compiling Host MLIR to LLVM dialects...")
        host_opt_cmd = [
            mlir_opt, host_intermediate_mlir,
            "--canonicalize", "--cse", 
            "--lower-affine", "--convert-scf-to-cf", "--convert-index-to-llvm",
            "--convert-arith-to-llvm", "--finalize-memref-to-llvm",
            "--convert-func-to-llvm", "--reconcile-unrealized-casts",
            "--canonicalize", "--cse"
        ]
        with open(host_llvm_mlir, "w") as f:
            subprocess.run(host_opt_cmd, stdout=f, check=True)

        print("[+] Translating to LLVM Bitcode...")
        run_cmd([mlir_translate, "--mlir-to-llvmir", kernel_llvm_mlir, "-o", kernel_ll])
        run_cmd([mlir_translate, "--mlir-to-llvmir", host_llvm_mlir, "-o", host_ll])

        print(f"[+] Compiling Host/Kernel Object Files...")
        run_cmd([clang, "-g", "-c", "-O3", openmp_flag, host_ll, "-o", host_o])
        run_cmd([clang, "-g", "-c", "-O3", openmp_flag, kernel_ll, "-o", kernel_o])

        print(f"[+] Linking final executable -> {final_output_path}")
        run_cmd([
            clang,
            "-g",
            "-O3", 
            openmp_flag, 
            host_o, 
            kernel_o, 
            "-o", final_output_path, 
            "-L/usr/lib/llvm-20/lib/", "-lm"
        ])

    print(f"[*] Build Successful! You can now run ./{os.path.basename(final_output_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile Polygeist CUDA output to CPU Vectors")
    parser.add_argument("input", help="Raw MLIR file or CUDA (.cu) file from Polygeist")
    parser.add_argument("-o", "--output", default="matrix_math", help="Final executable output name")
    parser.add_argument("-t", "--threads", default="256", help="Maximum OpenMP threads")
    parser.add_argument("-v", "--vector-width", default="256", help="Vector register bit-width (e.g., 256 for AVX2)")
    parser.add_argument("-u", "--unroll-factor", default="4", help="Loop unroll factor")

    args = parser.parse_args()
    compile_pipeline(args)
