# ViSiG-CR

흉부 X선 영상과 질문을 입력받아 질문에 맞는 대조 사례를 검색하고, 영상 언어 모델의 답변 생성을 지원하는 연구용 프로그램입니다. 판독문의 구조화 소견과 영상 특징을 활용하며, 별도의 모델 미세조정 없이 실행합니다.

## 설치

Linux, Python 3.11 이상, NVIDIA GPU와 모델 서버용 vLLM 환경이 필요합니다.

```bash
git clone https://github.com/UihyeonMaeng/ViSiG-CR.git
cd ViSiG-CR
python -m pip install -r requirements.txt
python -m pip install -e .
```

- 데이터셋과 사전학습 모델 가중치는 별도로 준비합니다.
- `config.yaml`의 데이터·모델·출력 경로와 서버 설정을 실행 환경에 맞게 수정합니다.


## 실행 방법

저장소 루트에서 아래 순서로 실행합니다. 데이터 준비는 MIMIC-CXR 기준 예시입니다.

### 1. 데이터 준비

```bash
python prepare_datasets.py mimic --config config.yaml
```

### 2. 소견 추출 및 검색 메모리 구축

```bash
bash model.sh start-ontology-extractor
python run_pipeline.py --config config.yaml extract-ontology
bash model.sh stop

python run_pipeline.py --config config.yaml build-visual-index
python run_pipeline.py --config config.yaml build-graph
python run_pipeline.py --config config.yaml build-presence-index
```

### 3. 대조 사례 검색 및 답변 생성

`/path/to/query.png`를 실제 질의 영상 경로로 바꿉니다. 검색과 답변 생성은 모델 서버를 교대로 실행합니다.

```bash
bash model.sh start-attribute-selector8
python run_pipeline.py --config config.yaml retrieve --image /path/to/query.png --question "Is there a pleural effusion?" --output work/evidence.json
bash model.sh stop

bash model.sh start-reader8
python run_pipeline.py --config config.yaml reader --package work/evidence.json --output work/answer.json
bash model.sh stop
```


## LICENSE

Copyright [yyyy] [Uihyeon Maeng]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
