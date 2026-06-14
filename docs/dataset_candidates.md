# Korean Dataset Candidate Review

Review date: 2026-06-13

## Selection Update

On 2026-06-14, the current pretraining build was narrowed to:

1. the Korean Wikimedia dump
2. Open Korean Historical Corpus records whose language is `Modern Korean`

The implementation also keeps the earlier public-domain filter for the
historical corpus. NIKL is excluded from the current build; its adapter remains
available for a later decision.

## Purpose

This document records the first dataset review for the MyGPT pretraining and
single-turn SFT pipeline. It is a shortlist, not a final dataset manifest.
Exact token counts, filtering rules, and mixing ratios will be decided after
source adapters produce comparable profiling reports.

The review follows the current project constraints:

- originally authored Korean first
- no translated or synthetic data in v1
- personal research use may include non-commercial or restricted data
- encyclopedia, news, and public or institutional documents before general web
- exact duplicate removal only until source-specific profiling is complete
- existing `skt/kogpt2-base-v2` tokenizer for token counting and training

License notes below are engineering screening notes, not legal advice. The
dataset manifest must preserve each source's exact terms and revision.

## Recommended V1 Shortlist

### Pretraining

| Priority | Source | Initial decision | Role |
| --- | --- | --- | --- |
| 1 | Korean Wikimedia dump | Include | Modern encyclopedic Korean baseline |
| 2 | Open Korean Historical Corpus | Conditional include | Public-domain news and institutional text after language/source filtering |
| 3 | National Institute of Korean Language corpora | Defer | Excluded from the current build; reconsider only after a later access decision |
| 4 | KOREAN-WEBTEXT | Defer | Fallback only if preferred sources remain below the usable-token target |

The current profiling build uses Wikimedia and selected Open Korean Historical
Corpus subsets only.

### Single-Turn SFT

| Priority | Source | Initial decision | Role |
| --- | --- | --- | --- |
| 1 | KLUE MRC | Include | Original Korean context-question-answer examples |
| 2 | KorQuAD 1.0 | Hold for license review | Additional original Korean extractive QA |
| 3 | NIKL summarization | Conditional include | Korean summarization instructions after official access review |
| 4 | KoAlpaca v1.1a | Exclude from v1 | License and answer-generation provenance are not clear enough |

KLUE MRC alone is useful but small for broad instruction following. The SFT v1
search therefore remains open for more human-authored Korean instruction,
explanation, and summarization data with explicit provenance.

## Pretraining Candidates

### 1. Korean Wikimedia Dump

Recommended source:

- <https://dumps.wikimedia.org/kowiki/20260601/>
- target file:
  `kowiki-20260601-pages-articles-multistream.xml.bz2`
- compressed size reported by Wikimedia: approximately 1.3 GB
- checksums are published alongside the dump
- content license: CC BY-SA and GFDL terms applicable to Wikipedia content

The direct Wikimedia dump is preferred over a third-party cleaned copy because
the snapshot date, file name, checksum, and upstream terms can be recorded
unambiguously. The 2026-06-01 article dump was complete when reviewed, although
unrelated full-history dump jobs on that date were still in progress.

For a processed-size reference, Hugging Face's
[`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia)
configuration `20231101.ko` reports:

- 647,897 rows
- 782,677,061 bytes of Parquet
- 6,823,640,837 bytes in memory
- fields: article id, URL, title, and text

This older Hugging Face build is suitable for adapter development and
comparison, but the production v1 build should pin and process the newer direct
dump. Exact KoGPT2 token count must be measured after extraction and cleaning.

Observed risks:

- templates, tables, lists, references, and malformed wiki markup require
  deterministic cleanup
- redirects and non-article namespaces must be excluded
- article-level records must remain intact until train/validation splitting
- attribution and share-alike obligations must remain visible in the manifest

### 2. Open Korean Historical Corpus

Source:

- <https://huggingface.co/datasets/seyoungsong/Open-Korean-Historical-Corpus>
- paper: <https://arxiv.org/abs/2510.24541>
- aggregate license: CC BY-NC 4.0
- reported scale: 17.7 million documents and 5.1 billion tokens
- reported coverage: 19 archives, seventh century through 2025

The dataset contains Modern Korean, older Korean varieties, North Korean,
Classical Chinese, and Japanese text. It cannot be included wholesale under the
"Korean original text first" policy. Its record schema is useful for selective
loading because it exposes `language`, `script`, `source`, `corpus`,
`copyright`, `year`, and `url`.

Initial eligibility rule for profiling:

```text
language == "Modern Korean"
copyright == "Public Domain"
text is non-empty
```

The build must also report counts by `corpus`, `source`, `year`, and `script`
before any subset is accepted. Newspaper archives dominate the repository file
layout, so an unrestricted load could overwhelm the mixture and train the model
mostly on historical orthography. Modern Hangul-heavy subsets should be
measured separately from Hanja or old-Hangul material.

This source is promising for reaching the 500-million-token target without
general web text, but the final source list and weights remain conditional on
profiling.

### 3. National Institute of Korean Language Corpora

Official service:

- <https://kli.korean.go.kr/>
- usage policy: <https://kli.korean.go.kr/boards/termsInfo.do>
- Hugging Face loader only:
  <https://huggingface.co/datasets/KETI-AIR/nikl>

The Hugging Face repository contains a loader script, not the corpus files. The
script requires the user to download approved resources manually from the
official NIKL service. Its Apache-2.0 header applies to the loader code and must
not be interpreted as the data license.

Potentially relevant official corpora include written text, newspapers, and
summarization. They are strong domain candidates, but the service terms restrict
use to the approved scope and prohibit unapproved copying, distribution, and
retention beyond the approved period. Therefore:

- the user must request and download the data directly
- credentials or downloaded files must not be committed
- the adapter should accept a local or Google Drive source directory
- restricted source text must not be uploaded to Hugging Face or W&B
- a combined dataset containing NIKL text remains private and restricted

NIKL is excluded from the current dataset build. The adapter remains in the
repository so a future dataset version can activate it after the user approves
access and the exact corpus-specific conditions are recorded.

### 4. Third-Party Wikipedia Builds

Reviewed:

- <https://huggingface.co/datasets/shopkeeper/kowiki-cleaned-050126>
- <https://huggingface.co/datasets/heegyu/kowikitext>

`shopkeeper/kowiki-cleaned-050126` is recent and reports 731,362 rows with about
936 MB of Parquet, but its dataset card does not clearly document the upstream
snapshot and license. `heegyu/kowikitext` declares CC BY-SA 3.0 and is easier to
trace, but it is based on a 2022 snapshot and splits articles into repeated
sections.

Both are useful for comparison tests, but neither should replace the direct
Wikimedia build in the canonical v1 dataset.

### 5. KOREAN-WEBTEXT

Source:

- <https://huggingface.co/datasets/HAERAE-HUB/KOREAN-WEBTEXT>

The dataset is large enough to be useful, but inspected samples included
shopping descriptions, religious text, and low-quality or aggressive blog
content. Its source mix includes general internet crawls, which conflicts with
the current preference to exhaust encyclopedia, news, and institutional sources
first.

Decision: defer. Reconsider only if the accepted preferred-source corpus remains
below 500 million usable KoGPT2 tokens after filtering and exact deduplication.

## SFT Candidates

### 1. KLUE MRC

Source:

- <https://huggingface.co/datasets/klue/klue>
- configuration: `mrc`
- license: CC BY-SA 4.0
- language/source metadata: monolingual Korean, original, expert-generated
- rows: 17,554 train and 5,841 validation

KLUE MRC is the strongest immediate SFT candidate. Its context, question, and
answer fields can be converted into explicit single-turn records without
translation or synthetic generation.

The adapter must preserve:

- `guid` as the source identifier
- source and news category metadata
- `is_impossible` and `question_type`
- original context and all answer spans

Answerable and impossible questions should be profiled separately before
deciding whether impossible examples belong in SFT v1.

### 2. KorQuAD 1.0

Source:

- <https://huggingface.co/datasets/KorQuAD/squad_kor_v1>
- official page: <https://korquad.github.io/category/1.0_KOR.html>
- scale: 60,407 train and 5,774 development examples
- official license: CC BY-ND 2.0 KR

KorQuAD is original Korean, crowdsourced QA over Korean Wikipedia. Its quality
and size are attractive, but the no-derivatives license creates uncertainty
around schema conversion, combined-dataset redistribution, and release of
derived artifacts.

Decision: do not include it in the automated combined SFT build until the
intended private training transformation and artifact handling have been
reviewed against the license. It may still be useful as an untouched evaluation
resource.

### 3. KoAlpaca v1.1a

Source:

- <https://huggingface.co/datasets/beomi/KoAlpaca-v1.1a>
- scale: 21,155 rows

The dataset card does not declare a data license, and the answer provenance is
not documented clearly enough to establish that responses are human-authored.
The examples link to Naver Knowledge iN material and contain expanded
instruction-style answers, so both source rights and possible synthetic
generation need clarification.

Decision: exclude from SFT v1. The current training example in this repository
should be treated as a legacy smoke-test input, not the approved production
dataset.

## Required Profiling Before Final Selection

Each source adapter must emit the same report before mixing:

- pinned source revision, snapshot date, URL, and checksum where available
- license and source-specific usage restrictions
- raw and accepted document counts
- raw and accepted UTF-8 bytes
- KoGPT2 token count and tokens per document distribution
- empty, malformed, and exact-duplicate removal counts
- Korean-character, Hanja, Latin, digit, and replacement-character ratios
- document length percentiles
- counts by source, corpus, year, script, and task-specific labels
- a deterministic random sample for manual inspection
- train/validation leakage check after the document-level split

No final mixing ratio should be set before this report exists. The next
implementation milestone is a profiling-only pipeline for the recommended
shortlist, followed by a review of the generated statistics and samples.
