import sys
import subprocess
import argparse
import os
import tempfile
import re
import contextlib

# Toolchain Paths
cgeist = "/mnt/pvc/Polygeist/build/bin/cgeist"
polygeist_opt = "/mnt/pvc/Polygeist/build/bin/polygeist-opt"
enzyme_opt = "/mnt/pvc/Enzyme-JAX/bazel-bin/enzymexlamlir-opt"
mlir_translate = "/mnt/pvc/Enzyme-JAX/bazel-bin/external/llvm-project/mlir/mlir-translate"
clang = "clang"
host_compiler = "clang++"

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


def fix_dlti_dialect_only(mlir_path):
    """
    Strips the DLTI spec so Enzyme's newer MLIR parser doesn't crash on
    Polygeist's older data layouts.
    """
    with open(mlir_path, 'r') as f:
        content = f.read()

    content = re.sub(
        r'dlti\.dl_spec\s*=\s*#dlti\.dl_spec<.*?>,\s*(?=llvm\.data_layout)',
        '',
        content,
        flags=re.DOTALL
    )

    with open(mlir_path, 'w') as f:
        f.write(content)


def fix_mlir_for_enzyme(mlir_path):
    with open(mlir_path, 'r') as f:
        content = f.read()

    # Strip the DLTI Version Mismatch
    content = re.sub(
        r'dlti\.dl_spec\s*=\s*#dlti\.dl_spec<.*?>,\s*(?=llvm\.data_layout)',
        '',
        content,
        flags=re.DOTALL
    )

    # Translate custom Polygeist pointer casts to Standard MLIR
    
    # Fix memref -> pointer
    pattern_m2p = r'%([a-zA-Z0-9_]+)\s*=\s*"polygeist\.memref2pointer"\((%[a-zA-Z0-9_]+)\)\s*:\s*\((memref<[^>]+>)\)\s*->\s*(!llvm\.ptr[^\n]*)'
    replacement_m2p = (
        r'%idx_\1 = memref.extract_aligned_pointer_as_index \2 : \3 -> index\n'
        r'    %i64_\1 = arith.index_cast %idx_\1 : index to i64\n'
        r'    %\1 = llvm.inttoptr %i64_\1 : i64 to \4'
    )
    content = re.sub(pattern_m2p, replacement_m2p, content)

    # Fix pointer -> memref (The Double-Cast Fold Strategy)
    pattern_p2m = r'%([a-zA-Z0-9_]+)\s*=\s*"polygeist\.pointer2memref"\((%[a-zA-Z0-9_]+)\)\s*:\s*\((!llvm\.ptr[^)]*)\)\s*->\s*(memref<[^>]+>)'
    replacement_p2m = (
        r'%c0_i64_\1 = llvm.mlir.constant(0 : i64) : i64\n'
        r'    %c1_i64_\1 = llvm.mlir.constant(1 : i64) : i64\n'
        r'    %undef_\1 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>\n'
        r'    %s1_\1 = llvm.insertvalue \2, %undef_\1[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>\n'
        r'    %s2_\1 = llvm.insertvalue \2, %s1_\1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>\n'
        r'    %s3_\1 = llvm.insertvalue %c0_i64_\1, %s2_\1[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>\n'
        r'    %s4_\1 = llvm.insertvalue %c0_i64_\1, %s3_\1[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>\n'
        r'    %s5_\1 = llvm.insertvalue %c1_i64_\1, %s4_\1[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>\n'
        r'    %\1 = builtin.unrealized_conversion_cast %s5_\1 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to \4'
    )
    content = re.sub(pattern_p2m, replacement_p2m, content)

    # Context-Aware AST Stack Parser for Terminators
    lines = content.split('\n')
    out_lines = []
    scope_stack = []

    for line in lines:
        if "scf.yield" in line:
            if scope_stack and scope_stack[-1] == "scf.parallel":
                line = line.replace("scf.yield", "scf.reduce")

        if "}" in line:
            for _ in range(line.count("}")):
                if scope_stack:
                    scope_stack.pop()

        if "{" in line:
            for _ in range(line.count("{")):
                if "scf.parallel" in line:
                    scope_stack.append("scf.parallel")
                elif "scf.if" in line:
                    scope_stack.append("scf.if")
                elif "scf.for" in line:
                    scope_stack.append("scf.for")
                else:
                    scope_stack.append("other")

        out_lines.append(line)

    with open(mlir_path, 'w') as f:
        f.write('\n'.join(out_lines))

def run_cmd(cmd):
    print(f"[+] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
   

    print(f"[*] Compiler Output (stderr):\n{result.stderr}")
    if result.stderr and result.stderr.strip():
        print(f"[*] Compiler Output (stderr):\n{result.stderr}")
        
    if result.returncode != 0:
        print(f"[-] Error executing command!\n")
        sys.exit(1)

def compile_cuda_to_object(input_path, obj_path, tmpdir):
    """Pushes a .cu file through the Polygeist/Enzyme pipeline to get a .o file."""
    print(f"[+] Compiling CUDA to Object: {input_path}")
    
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    raw_mlir = os.path.join(tmpdir, f"{base_name}_1_raw.mlir")
    final_mlir = os.path.join(tmpdir, f"{base_name}_2_final.mlir")
    final_ll = os.path.join(tmpdir, f"{base_name}_final.ll")
    sanitized_cu = os.path.join(tmpdir, f"{base_name}_sanitized.cu")

    sanitize_cuda_source(input_path, sanitized_cu)
    
    # Phase 1: Frontend
    run_cmd([cgeist, sanitized_cu, "-O3", "-fopenmp", "-I.", "-I/usr/local/cuda/include",
             "--cuda-gpu-arch=sm_80", "--cuda-lower", "--cpuify=distribute.mincut", "-S", "-o", raw_mlir])

    # Phase 1.5: Patching
    fix_mlir_for_enzyme(raw_mlir)

    # Phase 2: Enzyme Backend
    run_cmd([
        enzyme_opt, raw_mlir, "-allow-unregistered-dialect",
        "--inline", "--lower-affine", "--convert-scf-to-openmp",   
        "--finalize-memref-to-llvm", "--convert-scf-to-cf",       
        "--convert-openmp-to-llvm", "--convert-arith-to-llvm",   
        "--convert-func-to-llvm", "--reconcile-unrealized-casts",
        "--canonicalize", "-o", final_mlir
    ])

    # Phase 3: Translate to LLVM IR
    run_cmd([mlir_translate, "--mlir-to-llvmir", final_mlir, "-o", final_ll])
    
    # Phase 4: Compile to Object File (-c flag means "Compile only, do not link")
    run_cmd([clang, "-c", "-O3", openmp_flag, final_ll, "-o", obj_path])

def compile_cpp_to_object(input_path, obj_path, tmpdir):
    """Bypasses MLIR and compiles standard C++ host code directly to a .o file."""
    print(f"[+] Compiling standard C++ to Object: {input_path}")

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    sanitized_cpp = os.path.join(tmpdir, f"{base_name}_sanitized.cpp")
    
    # 1. Sanitize the host code to remove CUDA memory/event calls
    sanitize_cuda_source(input_path, sanitized_cpp)

# Using standard clang++ for normal host code
    run_cmd([host_compiler, "-c", "-O3", openmp_flag, "-I.", "-I/usr/local/cuda/include", sanitized_cpp, "-o", obj_path])

def build_project(args):
    final_executable = os.path.abspath(args.output)
    
    with contextlib.nullcontext("debug_build") as tmpdir:
        os.makedirs(tmpdir, exist_ok=True)
        object_files = []

        # 1. Compile Phase
        for src_file in args.inputs:
            ext = os.path.splitext(src_file)[1].lower()
            obj_name = os.path.splitext(os.path.basename(src_file))[0] + ".o"
            obj_path = os.path.join(tmpdir, obj_name)

            if ext == ".cu":
                compile_cuda_to_object(src_file, obj_path, tmpdir)
            elif ext in [".cpp", ".c", ".cc"]:
                compile_cpp_to_object(src_file, obj_path, tmpdir)
            else:
                print(f"[-] Skipping unknown file type: {src_file}")
                continue
                
            object_files.append(obj_path)

        # 2. Link Phase
        print("\n[+] Phase 5: Linking Object Files...")
        link_cmd = [host_compiler, "-O3", openmp_flag] + object_files + ["-o", final_executable, "-L/usr/lib/llvm-20/lib/", "-lm"]
        run_cmd(link_cmd)

    print(f"\n[*] Build Successful! Run ./{os.path.basename(final_executable)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-file GPU-to-CPU Builder")
    # 'nargs='+' allows you to pass multiple files to the script
    parser.add_argument("inputs", nargs='+', help="Input source files (.cu, .cpp)")
    parser.add_argument("-o", "--output", default="a.out", help="Final executable name")
    args = parser.parse_args()
    
    build_project(args)
