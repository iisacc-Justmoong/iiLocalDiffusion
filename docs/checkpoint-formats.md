# 로컬 checkpoint 형식과 변환

`reference/diffusers/checkpoint_conversion.py`의
`materialize_safetensors(path, cache_dir)`는 다운로드한 tensor checkpoint를
기존 safetensors 로더에 공급할 파일로 준비한다. 모델 계열과 checkpoint/LoRA/VAE 등
역할 판별, 필요한 구성 요소의 완전성 검증, 실제 생성은 이후 로더의 책임이다.
확장자가 같다고 서로 다른 모델 계열이 호환되는 것은 아니다.

| 확인된 컨테이너 | 처리 |
| --- | --- |
| safetensors 단일 파일 또는 분할 인덱스 | 단일 파일은 원본 경로와 identity를 반환한다. 분할 인덱스는 샤드와 이름 매핑을 검사해 제한된 버퍼로 결합한 파일을 캐시에 준비한다. |
| PyTorch ZIP 또는 pickle 내용 | 설치된 Torch의 제한된 weights-only 로더로 읽고 dense tensor state dictionary를 safetensors로 변환한다. |
| GGUF 내용 | Safetensors로 변환하지 않는다. 해당 모델 계열과 GGUF 양자화를 지원하는 네이티브 로더, 로컬 구성 기반 Diffusers 변환기 또는 ComfyUI workflow가 필요하다. |
| 판별할 수 없고 확장자 힌트도 없는 내용 또는 ONNX | 명확한 오류로 거부한다. |

파일 내용을 우선 판별하며 확장자는 보조 힌트다. Legacy 컨테이너는 임의의 Python 객체나
완성된 `torch.nn.Module`을 허용하지 않는다. 입력은 문자열 key와 일반
`torch.Tensor`/`torch.nn.Parameter` value로 이루어진 mapping이거나, 그러한
mapping을 `state_dict`에 담은 checkpoint여야 한다. Wrapper의 epoch/optimizer 등
훈련 metadata는 저장하지 않는다. 빈 state dictionary, scalar metadata가 섞인
flat dictionary, custom tensor subclass, sparse/meta/complex tensor와 양자화 텐서는 기본적으로 거부한다.
명시적 `quantized_tensors="dequantize"`는 Torch qint8·quint8·qint32의 지원되는 per-tensor·per-channel 양자화 텐서를 float32로 복원하며, 비양자화 텐서의 dtype은 유지한다. 범용 Diffusers CLI에서는 `--dequantize-pytorch`를 사용한다.
변환 정책과 양자화 통계는 별도 cache 및 manifest에 기록한다. packed 모듈·custom tensor와 지원되지 않는 양자화 형식은 거부한다.

## 제한된 역직렬화

변환 시 Torch는 지연 import되며 다음 옵션을 항상 명시한다.

```python
torch.load(open_source_file, map_location="cpu", weights_only=True)
torch.load(private_snapshot_path, map_location="cpu", weights_only=True, mmap=True)
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
않는다. POSIX에서 스토리지 항목이 비압축인 PyTorch ZIP은 변환 cache의 임시 스냅샷으로 복사하고
`weights_only=True` 메모리 매핑으로 읽는다. 원본을 직접 매핑하지 않으므로 외부에서
원본이 동기화·변경되어도 매핑된 파일이 잘려 프로세스가 중단되는 상황을 피한다.
스토리지 항목이 압축된 ZIP, 구형 pickle, Windows 또는 스냅샷 공간이 부족한 경우 기존 CPU weights-only 로더를 사용한다.
출력 텐서는 contiguous여야 하며 연속되지 않거나 서로 겹치는 텐서만 복사하여 모든 key와 값을
보존한다. 매핑 경로에도 임시 스냅샷과 변환 결과를 위한 디스크 공간이 필요하다.
비연속 또는 공유 텐서 복사, 구형 입력 및 운영체제 페이지 캐시로 인해 최대 메모리
사용량이 일정하게 제한되는 것은 아니다. 생성 품질, 모델 라이선스, 필요한 VRAM 또는
베이스 모델 호환성은 변환 성공으로 입증되지 않는다.

API 옵션은 [Torch 공식 load 문서](https://docs.pytorch.org/docs/stable/generated/torch.load.html),
dense/contiguous 저장 조건은
[safetensors 공식 Torch API](https://huggingface.co/docs/safetensors/api/torch)를 기준으로
2026-09-04에 확인했다. 기존의
[CVE-2025-32434 공식 공지](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6)도
현재 최소 버전 요구에 포함된다.

## Cache와 provenance

호출자는 리소스 저장 영역의 cache directory를 넘긴다. 예를 들어 저장소 루트에서
Python으로 사용할 때 다음과 같다.

```python
from pathlib import Path
import sys

root = Path.cwd()
sys.path.insert(0, str(root / "reference" / "diffusers"))
from checkpoint_conversion import materialize_safetensors

prepared = materialize_safetensors(
    "/absolute/local/models/model.ckpt",
    root / "resources" / ".state" / "checkpoint-conversion-cache",
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

Legacy cache는 전체 검증에서 `<cache_dir>/v1-<original_sha256>/model.safetensors`,
메타데이터 검증에서 `v1-metadata2-<identity>/model.safetensors`와 `conversion.json`을 포함한다. 임시 directory에서 두 파일을 완성한 뒤 동일
파일시스템의 directory rename으로 함께 게시한다. 실패 시 임시 파일은 정리한다.
동일 content를 병렬 변환하면 먼저 게시된 산출물의 manifest와 선택된 식별 정보를 확인한 뒤
재사용한다. 기존 cache가 변조되었거나 불완전하면 이를 덮어쓰지 않고 오류를 낸다.

새 변환 manifest는 원본 및 출력 식별 정보, Torch/safetensors 버전,
`weights_only=True`와 CPU 정책, `source_loading` 경로, tensor 수/dtype별 수, `state_dict` unwrap 여부,
`generation_verified=False`를 기록한다. 원본은 수정하지 않으며 전체 검증은 SHA를,
메타데이터 검증은 파일 크기와 장치·inode·수정 및 변경 시각 등의 식별 정보를 사용한다.
Symlink 원본의 target 변경과 cache 산출물의 symlink 교체도 거부한다.

전체 검증에서 같은 content를 다른 파일명으로 호출하면 cache 경로가 같다. 이때 반환값의
`original`은 현재 입력을, manifest의 `original`은 최초 변환 입력을 가리킨다.
Cache 재사용은 선택된 식별 정보와 manifest를 검증하므로 Torch import를 필요로 하지 않는다.
SHA는 기록 이후 content 변경을 검사하는 identity이며 발행자의 서명은 아니다. 출력 식별 정보가 없는 구형 메타데이터 cache는 그대로 보존하며 새 변환에 재사용하지 않는다.

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
당시 `build/reference/checkpoint-conversion-smoke/tiny-conversion.json` 및
`malicious-checkpoint.json`에 저장됐다. 현재 테스트는 결과 JSON을 표준 출력으로
내보내고, 임시 입력과 변환 결과는 `resources/test-workspaces/`에서 정리한다.
이 tiny fixture 검증은 실제 배포 모델의
생성 검증을 대신하지 않는다.

독립형 로컬 이미지 생성과 로컬 파이프라인 구성이 있는 범용 Diffusers 생성은 PyTorch 입력을 실행 시 `--cache-dir` 아래 `checkpoint-conversions/`에 변환한 뒤 실행 경로로 전달한다. 모델 파일의 원래 경로는 재실행 설정에, 원본과 변환본의 식별자는 결과 보고서에 남긴다. 파일은 검사와 변환 사이 및 출력 게시 전후에 다시 확인한다. `--print-config`와 독립형 `--validate-only`는 대용량 변환을 수행하지 않으며 `--inspect-model`을 안내한다. 변환에는 모델 크기에 비례한 CPU 메모리와 디스크가 필요하며, 형식 변환만으로 모델 아키텍처·구성 요소·양자화 또는 결과 품질의 호환성이 보장되지는 않는다.

이미지 출력 또는 그 JSON 보고서가 변환 전 체크포인트, 인덱스, 샤드나 변환된 가중치와 같은 경로를 가리키면 `--overwrite`를 지정해도 생성 전에 거부한다. 범용 파이프라인의 출력 디렉터리에는 해당 입력과 변환본을 둘 수 없다. 따라서 명시적으로 출력 교체를 허용한 경우에도 입력 모델과 변환 캐시는 보존된다.

## PyTorch 분할 체크포인트

PyTorch 분할 인덱스의 각 샤드는 안전한 weights-only 로더로 순차 검사·변환한다.
인덱스와 각 샤드의 텐서 이름이 정확히 일치해야 하며, 변환된 Safetensors 조각은
제한된 버퍼로 하나의 표준 파일로 결합한다. 인덱스·샤드·결과 식별자는 변환과
캐시 재사용 및 출력 게시 전에 확인한다. 원본은 수정하지 않으며 불완전하거나
변경된 인덱스와 잘못된 매핑은 실패한다. 이는 실제 모델 추론이나 품질 보장이 아니다.
병합 중에는 변환 조각과 결과를 위한 임시 디스크 공간이 모두 필요하다.
임시 조각은 검증 후 제거하며 재사용 가능한 병합본만 캐시에 남긴다. Safetensors 분할 인덱스도 같은 완전성·식별자 검사를 거쳐 원본을 수정하지 않고 제한된 버퍼로 병합할 수 있다.
