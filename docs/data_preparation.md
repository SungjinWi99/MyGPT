# 한국어 사전학습 데이터 준비

현재 파이프라인은 다음 두 소스를 하나의 정규화된 사전학습 데이터셋으로 만든다.

1. 한국어 Wikimedia 공식 덤프
2. `HAERAE-HUB/KOREAN-WEBTEXT`

Open Korean Historical Corpus는 현대 한국어 스모크 후보가 한자·옛한글
중심이었고 레코드의 `copyright`가 비어 있어 현재 빌드에서 제외했다. NIKL도
현재 제외 상태이며, 두 어댑터는 향후 재검토를 위해 코드에 유지한다.

출력은 Google Drive의 버전 디렉터리에 저장된다.

```text
/content/drive/MyDrive/KTB/MyGPT/datasets/
  raw/
    wikimedia/
    webtext/
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
- KOREAN-WEBTEXT는 상류 데이터셋의 문장·문서 필터 및 중복 제거가 이미
  적용된 Parquet을 입력으로 사용한다.
- KOREAN-WEBTEXT 데이터 카드에는 명시적 라이선스가 없으므로 매니페스트에
  `Not declared in the dataset card`로 기록하고 결과물 재배포는 보류한다.
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

최초 실행 시 약 1.29GB 공식 `pages-articles.xml.bz2` 덤프를
`raw/wikimedia`에 내려받고 Wikimedia가 공개한 SHA1 체크섬을 검증한다.
`multistream` 결합 파일은 공식 `sha1sums.txt`에 체크섬이 없어 기본 입력으로
사용하지 않는다.

```bash
python -m src.dataset_pipeline.prepare_pretrain \
  --output-dir /content/drive/MyDrive/KTB/MyGPT/datasets/pretrain/smoke-wikimedia-v1 \
  --work-dir /content/mygpt_dataset_work \
  --raw-cache-dir /content/drive/MyDrive/KTB/MyGPT/datasets/raw \
  --sources wikimedia \
  --max-accepted-per-source 1000
```

### KOREAN-WEBTEXT

전체 데이터는 18개 Parquet 샤드, 약 4.47GB 다운로드, 데이터 카드 기준
128만 문서와 약 22억 토큰이다. 스모크 테스트에서는 첫 샤드만 내려받고
1,000문서만 처리한다.

```bash
python -m src.dataset_pipeline.prepare_pretrain \
  --output-dir /content/drive/MyDrive/KTB/MyGPT/datasets/pretrain/smoke-webtext-v1 \
  --work-dir /content/mygpt_dataset_work \
  --raw-cache-dir /content/drive/MyDrive/KTB/MyGPT/datasets/raw \
  --sources webtext \
  --webtext-allow-pattern 'data/train-00000-of-00018.parquet' \
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

## 데이터 규모와 학습 예산

정식 통합 빌드는 원본을 재사용할 수 있도록 Wikimedia와 KOREAN-WEBTEXT 전체를
정규화된 Parquet으로 보존한다. 이것이 8~12시간 동안 22억 토큰을 전부
학습한다는 뜻은 아니다. A100 단일 GPU 학습용 토큰 예산, 소스 혼합 비율,
시퀀스 패킹은 데이터 프로파일을 검토한 뒤 별도 학습 입력 단계에서 결정한다.
