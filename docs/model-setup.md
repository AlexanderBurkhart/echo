# Local judge setup

The offline example uses only Python's standard library. Full tests need the
native binding but no model. This page covers the separate development evaluation
with real inference. Commands run from the repository root; no command in Echo
fetches a model automatically.

## Reference model

The locally verified reference artifact is:

| Item | Value |
| --- | --- |
| Base model | [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) |
| GGUF producer | [bartowski/Qwen2.5-14B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF) |
| GGUF repository revision | `05244aa5d871c661c80082a15d3bce44714d068d` |
| File | `Qwen2.5-14B-Instruct-Q4_K_M.gguf` |
| Bytes | `8988110976` |
| SHA-256 | `e47ad95dad6ff848b431053b375adb5d39321290ea2c638682577dafca87c008` |

Obtain the file separately from the [pinned revision](https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/tree/05244aa5d871c661c80082a15d3bce44714d068d)
under its upstream terms. The checksum and size above were checked against its
[Git LFS pointer](https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/raw/05244aa5d871c661c80082a15d3bce44714d068d/Qwen2.5-14B-Instruct-Q4_K_M.gguf)
and the existing local file. No weights are distributed with Echo. The GGUF
repository identifies its license as Apache-2.0; Echo's license does not replace
model terms or attribution to Qwen and the quantization producer.

After placing it in `models/`, verify:

```bash
sha256sum models/Qwen2.5-14B-Instruct-Q4_K_M.gguf
```

The revision pins the quantized artifact. It does not independently establish the
exact base-model snapshot used by its producer. Different models are accepted by
the CLI but their behavior has not been validated here.

## Native installation

The observed environment uses Linux x86-64, Python 3.12.3, Loguru 0.7.3, and
llama-cpp-python 0.3.35 on an NVIDIA GeForce RTX 4080 SUPER (16 GB). The installed
binding reports CUDA support and architecture `890` (compute capability 8.9).
This describes the available installation, not a new inference or memory benchmark.

The [binding's versioned installation instructions](https://github.com/abetlen/llama-cpp-python/blob/v0.3.35/README.md#installation)
describe source builds, compilers, backend options, and wheels. Its
[pinned llama.cpp submodule](https://github.com/abetlen/llama-cpp-python/tree/v0.3.35/vendor)
is `4df29be4f4c3673f428170fda944a5b19f743bb8`. For a source build, provide a working
C/C++ compiler and the CMake build tooling used by the binding. CUDA additionally
needs a compatible NVIDIA driver and CUDA development toolkit; see the
[upstream build instructions at that revision](https://github.com/ggml-org/llama.cpp/blob/4df29be4f4c3673f428170fda944a5b19f743bb8/docs/build.md).

For a new environment targeting the reference GPU, a source installation is:

```bash
python3.12 -m venv .venv
CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89" \
  .venv/bin/python -m pip install --no-binary=llama-cpp-python -r requirements.txt
```

Use the architecture appropriate to your GPU, and keep the build log. This recipe
follows upstream build options but was not executed as a fresh installation in
this release pass. For CPU installation, use the README's plain requirements
command. CPU and other GPU setups have not been tested for model evaluation.
Do not reuse the reference CUDA description for a different build or wheel.

Native requirements cannot be captured by the two direct Python dependency pins.
The inspected host has driver `580.178.04` and `nvcc` 12.0.140, but that alone does
not prove which compiler/toolkit produced the installed libraries. Record your
actual build source, flags, compiler/toolkit, and wheel origin where applicable.

Check the installed backend without loading a model:

```bash
.venv/bin/python -c 'import llama_cpp; print(llama_cpp.__version__); print(llama_cpp.llama_print_system_info().decode())'
```

The runtime settings request all GPU layers (`n_gpu_layers=-1`), context 8192,
batch 512, temperature 0, and a maximum of 1024 completion tokens. Defaults are
in [runtime_settings.py](../user_simulator/judge/runtime_settings.py); thread count
is a CLI argument. Memory needs depend on model, context, and backend; no minimum
memory or throughput guarantee is established by the offline tests.

## Run development evaluation

Supply provenance that describes **your installed runtime**. For the verified
reference model, set `ECHO_RUNTIME_REVISION` to your recorded llama.cpp revision
and build details, then run:

```bash
.venv/bin/python -m experiments.smoke_test.main \
  --model models/Qwen2.5-14B-Instruct-Q4_K_M.gguf \
  --model-revision 'bartowski/Qwen2.5-14B-Instruct-GGUF@05244aa5d871c661c80082a15d3bce44714d068d' \
  --runtime-revision "${ECHO_RUNTIME_REVISION:?Set this to your actual runtime revision and build provenance}" \
  --threads 8
```

For an official 0.3.35 source build, include its llama.cpp commit above. For a
custom build, use the actual commit. The string is caller-supplied provenance,
not an automatic attestation. The judge separately records binding version,
SHA-256 hashes of the model and installed Python/native runtime files, model
metadata, host platform/machine, generation settings, and prompts.

Missing model files fail with an explanatory error; blank provenance and invalid
thread counts are rejected before loading. The VS Code launch configuration
prompts for the same path and revisions. No generated user directory is required:
the development fixtures include their synthetic user configurations.

The command evaluates the 20 development cases only. It saves a timestamped run
under `artifacts/smoke_test_real_judge/`, including `logs/run.log`, score records,
and development diagnostics under `logs/smoke_test/`. Exit status 1 means the
existing development gate failed: complete scores/reference coverage and pooled
criterion MAE at most 0.20 on the 0–4 scale are required. Rule selection, ordering,
and score arithmetic remain separate diagnostics. An exit status of zero does
not validate human calibration or held-out transfer.

The published historical holdout was already exposed and is not part of this
command. Its fixtures, thresholds, and release guard were not changed during
release preparation. No live-model run was performed for this release pass.
