# 로컬 checkpoint 형식과 변환

`reference/diffusers/checkpoint_conversion.py`의
`materialize_safetensors(path, cache_dir)`는 다운로드한 tensor checkpoint를
기존 safetensors 로더에 공급할 파일로 준비한다. 모델 계열과 checkpoint/LoRA/VAE 등
역할 판별, 필요한 구성 요소의 완전성 검증, 실제 생성은 이후 로더의 책임이다.
확장자가 같다고 서로 다른 모델 계열이 호환되는 것은 아니다.

| 입력 확장자 | 처리 |
| --- | --- |
| `.safetensors`, `.safetensor` | 원본 경로와 SHA-256 identity를 반환한다. 이 단계에서는 tensor header를 파싱하거나 다시 저장하지 않는다. |
| `.ckpt`, `.pt`, `.pth`, `.bin` | 설치된 Torch의 제한된 weights-only 로더로 읽고 dense tensor state dictionary를 safetensors로 변환한다. |
| `.gguf` | 변환하지 않는다. 해당 모델 계열과 GGUF 양자화를 지원하는 native loader 또는 ComfyUI workflow가 필요하다. |
| 그 외 확장자 | 명확한 오류로 거부한다. |

확장자는 대소문자를 구분하지 않는다. Legacy 확장자라고 임의의 Python 객체나
완성된 `torch.nn.Module`을 허용하지 않는다. 입력은 문자열 key와 일반
`torch.Tensor`/`torch.nn.Parameter` value로 이루어진 mapping이거나, 그러한
mapping을 `state_dict`에 담은 checkpoint여야 한다. Wrapper의 epoch/optimizer 등
훈련 metadata는 저장하지 않는다. 빈 state dictionary, scalar metadata가 섞인
flat dictionary, custom tensor subclass, sparse/quantized/meta/complex tensor는
거부한다. Boolean·integer·floating tensor는 설치된 safetensors가 지원하는 dtype에
한해 원래 dtype을 유지한다. 지원되지 않는 dtype을 다른 정밀도로 암묵 변환하지 않는다.

## 제한된 역직렬화

변환 시 Torch는 지연 import되며 다음 옵션을 항상 명시한다.

```python
torch.load(open_source_file, map_location="cpu", weights_only=True)
```

`weights_only=False` 재시도, 추가 pickle globals 등록, custom pickle module,
remote code, 네트워크 및 모델 다운로드는 사용하지 않는다. 설치된 Torch 버전이
확인 가능한 안정 버전 `2.10.0` 이상이 아니면 checkpoint를 읽기 전에 거부한다.
이는 `weights_only=True`에도 영향을 주었던
[PyTorch CVE-2026-24747 공식 공지](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p)의
수정 버전 기준이다. 버전 문자열은 안정 버전과 CPU/CUDA 등의 local build suffix를
허용하고 개발·RC 버전은 거부한다. 같은 검사를 metadata inspector에서 재사용할 수
있도록 `ensure_safe_torch_version(torch) -> None`도 제공한다.

이 정책은 임의 checkpoint의 모든 메모리·리소스 위험에 대한 sandbox를 제공하지
않는다. CPU에서 원본 tensor를 읽고 각 tensor를 독립적인 contiguous storage로
복사하므로 checkpoint 크기에 비례하는 CPU RAM과 cache 공간이 필요하다.
공유 storage의 key를 삭제하지 않고 모든 이름과 값을 보존한다. 생성 품질, 모델
라이선스, 필요한 VRAM 또는 베이스 모델 호환성은 변환 성공으로 입증되지 않는다.

API 옵션은 [Torch 공식 load 문서](https://docs.pytorch.org/docs/stable/generated/torch.load.html),
dense/contiguous 저장 조건은
[safetensors 공식 Torch API](https://huggingface.co/docs/safetensors/api/torch)를 기준으로
2026-09-04에 확인했다. 기존의
[CVE-2025-32434 공식 공지](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6)도
현재 최소 버전 요구에 포함된다.

## Cache와 provenance

호출자는 `build/` 아래의 cache directory를 넘긴다. 예를 들어 저장소 루트에서
Python으로 사용할 때 다음과 같다.

```python
from pathlib import Path
import sys

root = Path.cwd()
sys.path.insert(0, str(root / "reference" / "diffusers"))
from checkpoint_conversion import materialize_safetensors

prepared = materialize_safetensors(
    "/absolute/local/models/model.ckpt",
    root / "build" / "reference" / "checkpoint-cache",
)
loader_path = prepared["converted_path"]
```

반환값은 다음 필드를 포함한다.

| 필드 | 의미 |
| --- | --- |
| `converted_path` | 후속 로더에 공급할 절대 경로 |
| `converted` | legacy 파일에서 변환했는지 여부 |
| `cache_hit` | 기존 검증된 변환 산출물을 재사용했는지 여부 |
| `original` | 이번 호출의 원본 `path`, `resolved_file`, `sha256`, `size_bytes` |
| `output` | 공급 파일의 `path`, `resolved_file`, `sha256`, `size_bytes` |
| `manifest_path`, `manifest` | 변환 manifest의 경로와 내용. 원본 safetensors를 그대로 쓰면 `None` |

Legacy cache는 `<cache_dir>/v1-<original_sha256>/model.safetensors`와
`conversion.json`을 포함한다. 임시 directory에서 두 파일을 완성한 뒤 동일
파일시스템의 directory rename으로 함께 게시한다. 실패 시 임시 파일은 정리한다.
동일 content를 병렬 변환하면 먼저 게시된 산출물의 manifest와 SHA를 확인한 뒤
재사용한다. 기존 cache가 변조되었거나 불완전하면 이를 덮어쓰지 않고 오류를 낸다.

Manifest는 원본 identity, 변환 결과 SHA-256/크기, Torch/safetensors 버전,
`weights_only=True`와 CPU 정책, tensor 수/dtype별 수, `state_dict` unwrap 여부,
`generation_verified=False`를 기록한다. 원본은 수정하지 않으며 원본 경로와 열린
파일의 SHA를 읽기 전후 및 게시 전후에 다시 검사한다. Symlink 원본은 resolved
target이 바뀌면 거부하고, cache 산출물의 symlink 교체도 거부한다.

같은 content를 다른 파일명으로 호출하면 cache 경로가 같다. 이때 반환값의
`original`은 현재 입력을, manifest의 `original`은 최초 변환 입력을 가리킨다.
Cache 재사용은 SHA와 manifest를 검증하므로 Torch import를 필요로 하지 않는다.
SHA는 기록 이후 content 변경을 검사하는 identity이며 발행자의 서명은 아니다.

## 검증

기본 회귀 테스트는 Torch/safetensors 설치 없이 mock으로 정책·오류·변조·원자적
게시를 검사한다.

```sh
python3 tests/CheckpointConversionTests.py
```

설치된 저장소 `.venv`를 사용하는 실제 CPU smoke도 같은 테스트에 포함되어 있다.
다음처럼 명시적으로 실행하며 모델을 다운로드하지 않는다.

```sh
IILD_CHECKPOINT_REAL_RUNTIME_TESTS=1 \
  reference/diffusers/.venv/bin/python tests/CheckpointConversionTests.py
```

2026-09-04 실행 결과는 Torch `2.13.0`에서 22개 테스트 통과이다. 실제 zip `.ckpt`와
구형 non-zip `.pt`를 변환한 뒤 safetensors로 다시 읽어 key·값·dtype·contiguity와
원본 SHA 불변을 검사했다. 실행 시 marker를 쓰도록 구성된 `__reduce__` fixture는
제한된 unpickler에서 거부되었으며 marker가 생성되지 않았음을 검사했다. 증거는
`build/reference/checkpoint-conversion-smoke/tiny-conversion.json` 및
`malicious-checkpoint.json`에 저장된다. 이 tiny fixture 검증은 실제 배포 모델의
생성 검증을 대신하지 않는다.
