# ViSiG-CR 프로그램 저작권 등록용 소스 복제물

프로그램/패키지 버전: 2.0.0
현재 검색 구현 버전: 3.2
검색 정책 계약: question_conditioned_contrast_v2
관계 직렬화 계약: visig_cr_relation_v2.0

## 프로그램 기능

흉부 X선 영상과 질문을 입력받아 영상 유사 사례를 검색하고, 질문에
관련된 소견 및 속성값을 기준으로 대조 사례를 구성하여 고정된 영상
언어 모델의 질의응답에 활용한다. 데이터 전처리, 온톨로지 사건 추출,
영상 임베딩 및 검색 인덱스 구축, presence 인덱스, 사례 검색, 답변 생성,
평가 기능을 포함한다. 현재 주요 검색 방식은 question-conditioned
Contrastive Retrieval이며 저장된 그래프 간선은 주요 검색에 사용하지 않는다.

## 복제물 구성과 원본 보존

- visig_cr/: 핵심 패키지 Python 소스 46개
- run_pipeline.py: 메모리 구축, 검색, 질의응답, 평가 실행 진입점
- prepare_datasets.py: 데이터 전처리 실행 진입점
- model.sh: 로컬 모델 서비스 실행 스크립트
- setup.py, requirements.txt: 설치 메타데이터와 기존 의존성 범위
- config.yaml, ontology.yaml: 기본 설정과 온톨로지
- COPY_MANIFEST.json, SHA256SUMS: 파일 목록, 복제본 조정 내역과 검증값

기존 파일은 읽기만 했다. 복제본의 config.yaml 내 project.version만
2.0에서 2.0.0으로 표기 통일했다. 나머지 포함 원본 파일의 바이트는
그대로 보존했다. 이 README와 검증 문서는 복제물용으로 새로 작성했다.
검색 구현 버전 3.2, 정책/스키마 식별자와 의존성 범위는 변경하지 않았다.

데이터, 환자 영상/보고서, 모델 가중치, 생성된 메모리/인덱스, 실험 결과,
로그, 캐시, 비교 모델 패키지와 비교 실행기, 연구 문서, 테스트, 예제,
다운로드 및 선택적 변환 보조 스크립트는 포함하지 않는다.

## 실행 환경과 기본 확인

이 복제물은 소스 제출물이며 데이터와 모델을 포함한 실행 환경 배포본은
아니다. Python 3.11 이상과 requirements.txt의 의존성이 필요하다.
실제 데이터 및 모델 가중치는 사용자가 별도로 준비하고 config.yaml의
경로를 실행 환경에 맞춰 설정한다. 기본 경로는 기존 Linux 컨테이너의
/workspace/visig_cr와 /workspace/Datasets/06.CXR를 그대로 사용한다.

모델 서비스 환경은 애플리케이션 환경과 별도로 구성한다. 기존 실행
문서의 서비스 기준 버전은 vllm==0.11.0이며 PyTorch/CUDA 조합은 원래
방식대로 하드웨어에 맞춰 준비한다. 의존성을 임의의 정확한 버전으로
고정하지 않았다. 선택적인 LLaVA-Med 변환 스크립트와 비교 실험 환경은
이 소스 복제물의 구성 범위에 포함하지 않는다.

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python run_pipeline.py --help
python prepare_datasets.py --help
bash model.sh help
```

COPY_MANIFEST.json의 source_sha256은 읽은 원본, archive_sha256은 ZIP에
담긴 파일을 의미한다. config.yaml은 명시한 버전 표기 조정으로 두 값이
다르다. SHA256SUMS는 해당 파일 자체를 제외한 모든 복제물 파일을 검증한다.
