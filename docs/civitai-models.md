<a id="civitai-base-model-compatibility"></a>

# Civita 기본 모델 호환성

현재 [검증 보고서](civitai-validation.md)는 카탈로그 및 설치 클래스 적용 범위를 실제 생성된 아티팩트 및 나머지 공백과 분리합니다.

진입점은 `reference/generate.py`이다. 명시한 Civitai 기본 모델을 이름 있는 프리셋·로컬 Diffusers 파이프라인·로컬 ComfyUI API 워크플로우로 라우팅한다. 카탈로그에는 이름 105개를 담은 고정 Civitai 스냅샷에서 선택한 **기본 모델 이름104개**가 있으며, 숨기거나 비활성화한 로컬 범주도 포함한다. `local` 항목 79개에는 상위 공급 측 로컬 런타임 경로가 있다. `hosted` 항목 23개와 `unknown` 범주 2개에는 자동 로컬 경로가 추가되지 않는다.

기존 SD1/SDXL 체크포인트 파일은 이제 [스탠다일 이미지 런타임](local-image-generation.md)에서 기본값으로 설정되며, 이는 Diffusers 를 통해 번들된 오프라인 구성과 토크나이저를 로드합니다. 관리된 워크플로우 레시피는 `--backend comfyui-local` 를 요구합니다. 명시적인 프리셋, 파이프라인 또는 워크플로우는 해당 실행 경로를 유지합니다.
[다운로드 검사기](downloaded-models.md)는 번들/분할 중량 및 구성 요소 역할을 식별합니다. [레거시 변환](checkpoint-formats.md)는 소스 해시를 보존합니다.

이는 **런타임 -의존 호환성 카탈로그**입니다. 모든 모델 가중치가 유효한 결과를 생성했다는 증거가 아닙니다. 이 카탈로그를 추가하는 동안 새로운 풀 사이즈 모델 가중치가 다운로드되거나 생성 테스트되지 않았습니다. 유닛 테스트는 완전한 이름 커버리지, 아키텍처 구분, 작업 메타데이터 및 라우팅 계약을 확인합니다. 특정 모델은 여전히 호환 가능한 로컬 구성 요소, 지원된 런타임 버전, 충분한 메모리 및 실제 가중치와의 검증을 필요로 합니다. C++ 매니페스트/메타데이터 경계는 변경되지 않았습니다.

<a id="commands"></a>

## 명령

GPU 런타임를 가져오지 않고 정확한 이름과 경로 메타데이터를 나열합니다.

```bash
python reference/generate.py --list-base-models
```

선택한 Python 환경이 모델 가중치를 로드하거나 서버에 연결하지 않고 카탈로그화된 각 Diffusers 파이프라인 클래스를 가져올 수 있는지 검사합니다.

```bash
reference/diffusers/.venv/bin/python reference/generate.py --check-runtime
reference/diffusers/.venv/bin/python reference/generate.py \
  --check-runtime --base-model 'Krea 2'
```

감사 기록은 고정된 카탈로그 소스, 설치된 Diffusers / Torch 패키지 버전 및 요청된 각 행에 대한 상태를 기록합니다. `pipeline_available` 는 제안된 클래스가 설치된 빌트인 `DiffusionPipeline` 서브클래스임을 의미하며, 그 지연된 임포트가 성공했다는 것을 의미합니다. `runtime_missing` 는 누락된 패키지, 부재한 클래스, 댄미/파이프라인 내보내기 또는 선택적 의존성 임포트 실패를 포함합니다. 각 실패는 다른 행을 중단시키지 않고 진단을 포함합니다. `workflow_required` 는 ComfyUI 설치, 서버, 노드 및 모델 파일을 체크하지 않으며, `hosted` 와 `unknown` 는 명시적으로 유지됩니다. 이 3 상태는 선택된 행이 유일한 경우 ML 런타임을 가져오지 않습니다.

각 감사 결과 및 행 집합 `generation_verified=false` 와 `weights_verified=false` 입니다. 감사는 모델 생성자를 호출하거나, 가중치를 다운로드하거나, 디노이징을 실행하거나, GPU 를 확인하지 않습니다. 실제 모델 생성 경로를 사용하여 호환성 및 출력 검증을 수행하거나, ComfyUI `--validate-only` 를 사용하여 라이브 워크플로우/노드/모델 가용성을 확인합니다. Python 호출자는 동일한 보고를 위해 `runtime_compatibility.inspect_runtime(base_name=None)` 를 사용할 수 있습니다. 해당 모듈만 가져오면 Torch 나 Diffusers 를 가져오지 않습니다.

SDXL 파생 사전 설정과 일치하는 로컬 Illustrious 체크포인트를 사용합니다.

```bash
reference/diffusers/.venv/bin/python reference/generate.py \
  --base-model Illustrious \
  --model /absolute/path/Illustrious.safetensors
```

NoobAI는 예측 매개변수화를 결정하는 데 정보가 충분하지 않습니다. 실제 체크포인트가 v-pred 모델인 경우 v-pred 사전 설정을 명시적으로 사용합니다.

```bash
reference/diffusers/.venv/bin/python reference/generate.py --model-config /absolute/path/model-config \
  --base-model NoobAI --preset noobai-v-pred \
  --model /absolute/path/NoobAI-XL-Vpred.safetensors
```

FLUX.1 파생 가중치에는 정확한 FLUX.1 Krea 이름을 사용합니다. `Krea 2`는 별도의 아키텍처입니다.

```bash
reference/diffusers/.venv/bin/python reference/generate.py \
  --base-model 'Flux.1 Krea' \
  --model /absolute/path/flux1-krea-dev.safetensors
```

전체 로컬 Diffusers 디렉터리는 `model_index.json` 및 구성 요소 폴더의 구성을 유지합니다.

```bash
reference/diffusers/.venv/bin/python reference/generate.py \
  --backend diffusers --base-model 'SD 3.5' \
  --model /absolute/path/sd35-diffusers \
  --pipeline-inputs '{"prompt":"a ceramic teapot on a wooden table","num_inference_steps":28}' \
  --output-dir build/reference/sd35
```

ComfyUI 에 대해, 일치하는 로컬 모델, 로더 및 출력 노드를 가진 **API 형식** 워크플로우를 내보냅니다. 워크플로우는 모델 컴포넌트 및 작업별 입력의 권한입니다. `--workflow-inputs` 의 노드 ID 는 해당 워크플로우에 존재해야 하며, `6` 는 오직 예시일 뿐입니다. 큐잉 전에 바인딩을 확인합니다.

```bash
python reference/generate.py \
  --backend comfyui --base-model Anima \
  --workflow /absolute/path/anima-api.json \
  --workflow-inputs '{"6":{"text":"a ceramic teapot on a wooden table"}}' \
  --output-dir build/reference/anima --validate-only
```

로컬 ComfyUI 서버, 모델 및 출력 노드가 준비되면 `--validate-only` 없이 동일한 명령을 실행합니다. 카테고리 이름은 누락된 가중치를 제공하지 않으며 호스팅 제공자를 로컬 모델로 변경하지 않으며 모델 액세스 또는 사용 권한을 부여하지 않습니다.

<a id="architecture-and-task-distinctions"></a>

## 아키텍처 및 작업 구별

- **Illustrious, NoobAI 및 Pony**는 SDXL에서 파생되었습니다. 디노이저 아키텍처가 일치하더라도 어댑터에 대해 명명된 모델의 정체성을 유지하십시오. NoobAI v-pred 작성자는 `prediction_type="v_prediction"`와 `rescale_betas_zero_snr=True`를 명시적으로 요구합니다; 해당 배포된 스케줄러인 JSON는 여전히 `epsilon`라고 말할 수 있으므로, 카테고리 또는 원시 스케줄러 JSON만으로는 충분하지 않습니다.
- **Pony V7** 는 AuraFlow 에서 유래하며 SDXL 에서 유래하지 않습니다. **FLUX.1   Krea** 는 `FluxPipeline` 를 사용하며; **Krea   2** 는 `Krea2Pipeline` 를 사용합니다. 원본/기본 및 증류/터보 변형은 다른 가이드와 시간 단계 설정이 필요합니다.
- **SD   2.x** 는 에파손 또는 v-예측을 사용할 수 있습니다. LCM , 하이퍼, 라이트닝 및 터보는 일반적인 기본 모델 스케줄러 기본값이 아닙니다. SDXL 증류는 또한 디노이저 차원을 변경할 수 있습니다. 실제 로컬 구성과 체크포인트 지침을 사용하세요.
- **Anima와 MiniMax H3**는 모듈식 Diffusers 지원을 제공하며, 이는 기존의 `DiffusionPipeline`와 다릅니다. 렌즈, MageFlow 및 HiDream-O1도 자체 코드/계약이 필요합니다. 카탈로그는 이러한 패밀리에 대해 일반적인 이미지 파이프라인이 존재한다고 가정하기보다 명시적인 로컬 ComfyUI 워크플로를 선호합니다.
- **Wan 2.2 A14B**는 2개 전문가 아키텍처이다. 고노이즈 전용 또는 저노이즈 전용 파일은 완전한 파이프라인이 아니다. Wan의 이미지 조건·해상도·VAE 버전은 일치해야 한다. LTX-2.5의 가중치는 LTX-2/2.3와 다르다.
- **SVD / SVD XT**는 Civitai의 레거시 `type="image"` 메타데이터에도 불구하고 이미지에서 비디오를 생성합니다. **ACE 오디오/ MiniMax 음악 3** 오디오를 생성합니다. **Hunyuan3D/Pixal3D/Trellis.2**는 3D 자산을 생성합니다. 그들의 워크플로 출력은 적절한 미디어/메시 형식으로 내보내야 합니다.
- **시비타이 기본 모델 이름은 파일 형식**이 아닙니다. SafeTensor , PickleTensor , GGUF , Diffusers 디렉터리, Core ML 및 ONNX 는 다른 로더가 필요합니다. LoRA , VAE , 텍스트 인코더, UNet , 워크플로우 또는 업스케일러는 반드시 완전한 체크포인트는 아닙니다. 일반적인 로컬 Diffusers 라우트는 아무런 알림 없이 변환, GGUF pickle 실행 또는 임의의 커스텀 원격 코드 임포트를 수행하지 않으며, 해당 라우트 밖의 형식에 대해서는 일치하는 로컬 워크플로우/ 런타임 를 사용하십시오.

<a id="catalog-api-and-update-contract"></a>

## 카탈로그 API 및 업데이트 계약

`reference/diffusers/civitai_catalog.py` 는 Python 표준 라이브러리만 사용합니다. `lookup_base_model(name)` 는 독립적인 사전을 반환하고 `ValueError` 로 알 수 없는 이름을 거부하며, 대소문자와 최외각 공백만 정규화됩니다. `list_base_models()` 는 상위 공급 측 순서로 독립적인 레코드를 반환합니다. `CATALOG_SOURCE` 와 `UPSTREAM_SOURCES` 은 고정된 소스 정보를 노출합니다.

각 레코드에는 `name`, `family`, `local_status`, `preferred_backend`, `preset`, `pipeline_class`, `task`, `notes`, `sources`, `civitai_hidden` 및 `civitai_disabled` 이 포함됩니다. `pipeline_class` 은 호환되지 않는 `model_index.json` 를 덮어쓰기 위한 지시 사항이 아닌 제안된 작업별 클래스입니다. 쉼표로 구분된 `task` 값 이름 대안들; 이는 하나의 파이프라인이 모든 작업을 지원한다는 것을 의미하지 않습니다. `preset` 는 정의된 모델 계약이 있는 이름이 있는 프리셋에만 공급됩니다.

새로운 Civitai 스냅샷을 채택할 때 카탈로그 JSON 와 독립적인 이름 픽스처 을 함께 업데이트하세요. 각 새 이름에 대해 아키텍처, 로컬-가중치 사용 가능 여부 및 런타임 소스 증거를 검토하세요. Civitai 의 `hidden`, `disabled`, `generation` 또는 `selfHosted` 플래그에서 로컬 사용 가능 여부를 추론하지 마세요: 이들은 Civitai 의 서비스를 설명하며 사용자의 컴퓨터를 설명하지 않습니다. 알 수 없는 카테고리는 명시적으로 유지됩니다. 체크인된 카탈로그를 읽을 때 네트워크 액세스는 발생하지 않습니다.

<a id="source-snapshot"></a>

## 소스 스냅샷

- Civitai: [ec49115e55d7c85722ec6223ec342c36df59c9d4](https://github.com/civitai/civitai/blob/ec49115e55d7c85722ec6223ec342c36df59c9d4/packages/civitai-shared/src/basemodel.constants.ts), 2026-09-04T02:33:02Z, 2026-09-04에 검색됨. 파일 SHA-256: `032c3f038b5c4abf1e4a47a4622e9eeb8db084080e38c9f2830dcb8fa9f5fea3`.
- 디퓨저: [7643c4826609c47755e3da0e5b768e8070468f49](https://github.com/huggingface/diffusers/blob/7643c4826609c47755e3da0e5b768e8070468f49/src/diffusers/pipelines/__init__.py).
- comfyui: [e80c1570b6b44a2557d5d8e341e05782d18c9bbb](https://github.com/Comfy-Org/ComfyUI/blob/e80c1570b6b44a2557d5d8e341e05782d18c9bbb/comfy/supported_models.py).
- [유명한 XL 모델 카드](https://huggingface.co/OnomaAIResearch/Illustrious-XL-v1.0), [NoobAI v-pred 지침](https://huggingface.co/Laxhar/noobai-XL-Vpred-1.0/blob/main/README.md), [FLUX.1 Krea 모델 카드](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev).
- [Krea 2 파이프라인](https://huggingface.co/docs/diffusers/main/en/api/pipelines/krea2), [MiniMax H3 모듈식 파이프라인](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3), [렌즈 모델 카드](https://huggingface.co/microsoft/Lens), [MageFlow 참조](https://github.com/microsoft/Mage/tree/main/mage_flow).

<a id="complete-snapshot-matrix"></a>

## 완전한 스냅샷 매트릭스

`local` 는 상위 공급 측 로컬 구현이 런타임 요구 사항에 따라 존재한다는 것을 의미합니다. `hosted` 는 이 카탈로그에 검증된 로컬 경로가 없는 Civitai 공급자/카테고리를 설명하며, `unknown` 는 명시적인 아키텍처 식별이 필요합니다. 소스 증거와 변형 주의사항은 JSON 카탈로그의 각 행마다 유지됩니다.

|Civitai 기본 모델 이름|상태|선호 경로|작업|
|---|---|---|---|
|Anima|로컬| `comfyui` |텍스트를 이미지로 변환|
| AuraFlow |로컬| `AuraFlowPipeline` |텍스트를 이미지로 변환|
|크로마|로컬| `ChromaPipeline` |텍스트를 이미지로 변환|
| CogVideoX |로컬| `CogVideoXPipeline` |텍스트-비디오,이미지-비디오,비디오-비디오|
|Ernie|로컬| `ErnieImagePipeline` |텍스트를 이미지로 변환|
| Flux.1S |로컬| `flux1-schnell-compatible` |텍스트를 이미지로 변환|
| Flux.1D |로컬| `flux1-dev` |텍스트를 이미지로 변환|
| Flux.1Krea |로컬| `flux1-krea-dev` |텍스트를 이미지로 변환|
|Flux.1 Kontext|로컬| `FluxKontextPipeline` |이미지 대 이미지|
| Flux.2D |로컬| `Flux2Pipeline` |텍스트-이미지, 이미지-이미지|
|Flux.2 클라인 9B|로컬| `Flux2KleinPipeline` |텍스트-이미지, 이미지-이미지|
|Flux.2 Klein 9B 기반|로컬| `Flux2KleinPipeline` |텍스트-이미지, 이미지-이미지|
|Flux.2 클라인 4B|로컬| `Flux2KleinPipeline` |텍스트-이미지, 이미지-이미지|
|Flux.2 Klein 4B 기반|로컬| `Flux2KleinPipeline` |텍스트-이미지, 이미지-이미지|
|Flux 3 비디오|호스팅| `unavailable` |텍스트-비디오|
|Grok|호스팅| `unavailable` |텍스트-이미지, 텍스트-비디오, 이미지-비디오|
| HappyHorse |호스팅| `unavailable` |텍스트-비디오|
| HiDream |로컬| `HiDreamImagePipeline` |텍스트를 이미지로 변환|
| HiDream-O1 |로컬| `comfyui` |텍스트-이미지, 이미지-이미지|
|훈위안 1|로컬| `HunyuanDiTPipeline` |텍스트를 이미지로 변환|
|훈위안 비디오|로컬| `HunyuanVideoPipeline` |텍스트-비디오,이미지-비디오|
|표의 문자 4.0|로컬| `Ideogram4Pipeline` |텍스트를 이미지로 변환|
|Boogu|로컬| `comfyui` |텍스트-이미지, 이미지-이미지|
|일러스트리어스|로컬| `illustrious` |텍스트를 이미지로 변환|
|Imagen4|호스팅| `unavailable` |텍스트를 이미지로 변환|
|컬러스|로컬| `KolorsPipeline` |텍스트를 이미지로 변환|
| Krea 2 |로컬| `Krea2Pipeline` |텍스트를 이미지로 변환|
| LTXV |로컬| `LTXPipeline` |텍스트-비디오,이미지-비디오|
| LTXV2 |로컬| `comfyui` |텍스트-비디오,이미지-비디오|
| LTXV 2.3 |로컬| `comfyui` |텍스트-비디오,이미지-비디오|
| LTXV 2.5 |로컬| `comfyui` |텍스트-비디오,이미지-비디오|
|렌즈|로컬| `comfyui` |텍스트를 이미지로 변환|
|Lumina|로컬| `Lumina2Pipeline` |텍스트를 이미지로 변환|
| MageFlow |로컬| `comfyui` |텍스트-이미지, 이미지-이미지|
| MAI |호스팅| `unavailable` |텍스트-이미지, 이미지-이미지|
|떡|로컬| `MochiPipeline` |텍스트-비디오|
|나노 바나나|호스팅| `unavailable` |텍스트-이미지, 이미지-이미지|
| NoobAI |로컬| `noobai` |텍스트를 이미지로 변환|
| ODOR |알 수 없음| `unavailable` |알 수 없음|
| OpenAI |호스팅| `unavailable` |텍스트를 이미지로 변환|
|업스케일러|로컬| `comfyui` |이미지 업스케일|
|기타|알 수 없음| `unavailable` |알 수 없음|
| PixArt a |로컬| `PixArtAlphaPipeline` |텍스트를 이미지로 변환|
| PixArt E |로컬| `PixArtSigmaPipeline` |텍스트를 이미지로 변환|
|놀이터 v2|로컬| `StableDiffusionXLPipeline` |텍스트를 이미지로 변환|
|포니|로컬| `pony` |텍스트를 이미지로 변환|
|포니 V7|로컬| `AuraFlowPipeline` |텍스트를 이미지로 변환|
|Qwen|로컬| `QwenImagePipeline` |텍스트를 이미지로 변환|
|Qwen 2|호스팅| `unavailable` |텍스트-이미지, 이미지-이미지|
|Qwen 3|호스팅| `unavailable` |텍스트-이미지, 이미지-이미지|
|안정적인 캐스케이드|로컬| `StableCascadeCombinedPipeline` |텍스트를 이미지로 변환|
| SD 1.4 |로컬| `StableDiffusionPipeline` |텍스트를 이미지로 변환|
| SD 1.5 |로컬| `sd15-compatible` |텍스트를 이미지로 변환|
| SD 1.5 LCM |로컬| `LatentConsistencyModelPipeline` |텍스트를 이미지로 변환|
|SD 1.5 하이퍼|로컬| `StableDiffusionPipeline` |텍스트를 이미지로 변환|
| SD 2.0 |로컬| `StableDiffusionPipeline` |텍스트를 이미지로 변환|
| SD 2.0 768 |로컬| `StableDiffusionPipeline` |텍스트를 이미지로 변환|
| SD 2.1 |로컬| `StableDiffusionPipeline` |텍스트를 이미지로 변환|
| SD 2.1 768 |로컬| `StableDiffusionPipeline` |텍스트를 이미지로 변환|
|SD 2.1|로컬| `StableUnCLIPImg2ImgPipeline` |이미지 대 이미지|
| SD 3 |로컬| `StableDiffusion3Pipeline` |텍스트를 이미지로 변환|
| SD 3.5 |로컬| `StableDiffusion3Pipeline` |텍스트를 이미지로 변환|
|SD 3.5 대형|로컬| `StableDiffusion3Pipeline` |텍스트를 이미지로 변환|
|SD 3.5 대형 터보|로컬| `StableDiffusion3Pipeline` |텍스트를 이미지로 변환|
|SD 3.5 중간|로컬| `StableDiffusion3Pipeline` |텍스트를 이미지로 변환|
| SDXL 0.9 |로컬| `StableDiffusionXLPipeline` |텍스트를 이미지로 변환|
| SDXL 1.0 |로컬| `sdxl` |텍스트를 이미지로 변환|
| SDXL 1.0 LCM |로컬| `StableDiffusionXLPipeline` |텍스트를 이미지로 변환|
|SDXL 라이트닝|로컬| `StableDiffusionXLPipeline` |텍스트를 이미지로 변환|
|SDXL 하이퍼|로컬| `StableDiffusionXLPipeline` |텍스트를 이미지로 변환|
|SDXL 터보|로컬| `StableDiffusionXLPipeline` |텍스트를 이미지로 변환|
|SDXL 증류된|로컬| `StableDiffusionXLPipeline` |텍스트를 이미지로 변환|
|Reve|호스팅| `unavailable` |텍스트-이미지, 이미지-이미지|
|뮤즈 이미지|호스팅| `unavailable` |텍스트-이미지, 이미지-이미지|
|Seedream|호스팅| `unavailable` |텍스트를 이미지로 변환|
| SVD |로컬| `StableVideoDiffusionPipeline` |이미지-비디오|
| SVD XT |로컬| `StableVideoDiffusionPipeline` |이미지-비디오|
|소라 2|호스팅| `unavailable` |텍스트-비디오|
|Veo 3|호스팅| `unavailable` |텍스트-비디오|
|완 비디오|로컬| `WanPipeline` |텍스트-비디오|
|완 비디오 1.3B t2v|로컬| `WanPipeline` |텍스트-비디오|
|Wan 비디오 14B t2v|로컬| `WanPipeline` |텍스트-비디오|
|Wan 비디오 14B i2v 480p|로컬| `WanImageToVideoPipeline` |이미지-비디오|
|Wan 비디오 14B i2v 720p|로컬| `WanImageToVideoPipeline` |이미지-비디오|
|완 비디오 2.2 TI2V-5B|로컬| `WanPipeline` |텍스트-비디오,이미지-비디오|
|완 비디오 2.2 I2V-A14B|로컬| `WanImageToVideoPipeline` |이미지-비디오|
|완 비디오 2.2 T2V-A14B|로컬| `WanPipeline` |텍스트-비디오|
|완 비디오 2.5 T2V|호스팅| `unavailable` |텍스트-비디오|
|완 비디오 2.5 I2V|호스팅| `unavailable` |이미지-비디오|
|Wan 이미지 2.7|호스팅| `unavailable` |텍스트를 이미지로 변환|
|완 비디오 2.7|호스팅| `unavailable` |텍스트-비디오|
|완 비디오 3.0|호스팅| `unavailable` |텍스트-비디오|
| ZImageTurbo |로컬| `ZImagePipeline` |텍스트를 이미지로 변환|
| ZImageBase |로컬| `ZImagePipeline` |텍스트를 이미지로 변환|
|Vidu Q1|호스팅| `unavailable` |텍스트-비디오|
| MiniMax H3 |로컬| `comfyui` |텍스트-비디오,이미지-비디오,비디오-비디오|
|Kling|호스팅| `unavailable` |텍스트-비디오|
|ACE 오디오|로컬| `comfyui` |텍스트-오디오|
|MiniMax 음악 3|로컬| `comfyui` |텍스트-오디오|
| PolyGen |호스팅| `unavailable` |텍스트-3d, 이미지-3d|
|Tripo|호스팅| `unavailable` |이미지를3D로|
|Hunyuan3D|로컬| `comfyui` |이미지를3D로|
|Pixal3D|로컬| `comfyui` |이미지를3D로|
|Trellis.2|로컬| `comfyui` |이미지를3D로|
