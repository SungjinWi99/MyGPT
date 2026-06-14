# 한국어 사전학습 데이터 준비

현재 파이프라인은 다음 두 소스를 하나의 정규화된 사전학습 데이터셋으로 만든다.

1. 한국어 Wikimedia 공식 덤프
2. Open Korean Historical Corpus 중 `Modern Korean`

NIKL 말뭉치는 접근 및 수동 준비 부담 때문에 현재 빌드에서 제외했다. 관련
어댑터는 향후 재검토를 위해 코드에만 유지한다.

출력은 Google Drive의 버전 디렉터리에 저장된다.

```text
/content/drive/MyDrive/KTB/MyGPT/datasets/
  raw/
    wikimedia/
    historical/
      <hugging-face-revision>/
  pretrain/
    v1/
      parquet/
        train/
        validation/
      manifest.json
      profile.json
      samples.jsonl
```

## 처리 정책

- 텍스트는 Unicode NFC와 LF 줄바꿈으로 정규화한다.
- 정규화 후 SHA-256이 같은 문서는 선택한 소스 전체에서 한 번만 남긴다.
- 주제, 안전성, 문서 길이 기반 필터는 아직 적용하지 않는다.
- Open Korean Historical Corpus는 `language == "Modern Korean"`만 남긴다.
- 역사 말뭉치는 기본적으로 `copyright == "Public Domain"`도 요구한다.
- 문서 해시를 이용해 99.5% 학습, 0.5% 검증으로 결정론적으로 분할한다.
- `skt/kogpt2-base-v2`의 실제 Hub 커밋을 조회해 매니페스트에 고정한다.
- 정제된 원문은 Zstandard 압축 Parquet으로 저장한다.
- 데이터 버전은 불변이다. 이미 `v1`이 있으면 덮어쓰지 않는다.

`profile.json`에는 소스별 문서 수, 제거 사유, UTF-8 용량, KoGPT2 토큰 수,
문서 길이 백분위수, 문자 비율이 기록된다. `samples.jsonl`에는 수동 검토용
결정론적 샘플이 저장된다.

## Colab 설치

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
git clone https://github.com/SungjinWi99/MyGPT.git
cd MyGPT
pip install -r requirements.txt
```

이미 clone한 저장소라면 `git pull`만 실행한다.

## 소스별 스모크 테스트

정식 빌드 전에 각 어댑터를 별도 버전으로 확인한다. 출력 버전 이름은 다시
사용하지 않는다.

### Wikimedia

최초 실행 시 약 1.3GB 공식 덤프를 `raw/wikimedia`에 내려받고 체크섬을
검증한다.

```bash
python -m src.dataset_pipeline.prepare_pretrain \
  --output-dir /content/drive/MyDrive/KTB/MyGPT/datasets/pretrain/smoke-wikimedia-v1 \
  --work-dir /content/mygpt_dataset_work \
  --raw-cache-dir /content/drive/MyDrive/KTB/MyGPT/datasets/raw \
  --sources wikimedia \
  --max-accepted-per-source 1000
```

### Open Korean Historical Corpus

스모크 테스트에서는 한 파일만 내려받는다. 정식 빌드에서는
`--historical-allow-pattern`을 제거하면 전체 JSONL 스냅샷을 받는다.

```bash
python -m src.dataset_pipeline.prepare_pretrain \
  --output-dir /content/drive/MyDrive/KTB/MyGPT/datasets/pretrain/smoke-historical-v1 \
  --work-dir /content/mygpt_dataset_work \
  --raw-cache-dir /content/drive/MyDrive/KTB/MyGPT/datasets/raw \
  --sources historical \
  --historical-allow-pattern gaksa_modern.jsonl \
  --max-accepted-per-source 1000
```

각 실행 후 `profile.json`과 `samples.jsonl`을 먼저 검토한다. 정식 빌드를
시작하기 전에 두 스모크 결과를 Codex와 함께 확인한다.

## 정식 통합 빌드

스모크 결과를 승인한 뒤 실행한다.

```bash
chmod +x scripts/prepare_pretrain_colab.sh
./scripts/prepare_pretrain_colab.sh
```

스크립트 기본값:

```text
DATASET_ROOT=/content/drive/MyDrive/KTB/MyGPT/datasets
OUTPUT_VERSION=v1
WORK_DIR=/content/mygpt_dataset_work
RAW_CACHE_DIR=$DATASET_ROOT/raw
```

필요하면 출력 버전을 환경 변수로 변경할 수 있다.

```bash
OUTPUT_VERSION=v2 ./scripts/prepare_pretrain_colab.sh
```

## 실패한 빌드

성공한 빌드만 최종 버전 디렉터리로 승격된다. 실행이 중단되면 다음과 같은
미완료 디렉터리가 남을 수 있다.

```text
/content/drive/MyDrive/KTB/MyGPT/datasets/pretrain/.v1.building
/content/mygpt_dataset_work/.<build-id>.building
```

원인을 확인한 뒤 이 미완료 디렉터리를 삭제하고 같은 명령을 다시 실행한다.
원본 다운로드 캐시는 Drive에 남아 있으므로 Wikimedia와 Hugging Face 파일은
이어받거나 재사용된다.
