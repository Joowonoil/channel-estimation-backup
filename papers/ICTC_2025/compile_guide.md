# ICTC 2025 논문 컴파일 가이드

## LaTeX 설치 방법

### Windows에서 MiKTeX 설치
1. https://miktex.org/download 에서 MiKTeX 다운로드
2. 설치 후 MiKTeX Console을 열고 필요한 패키지 업데이트
3. 또는 명령어로 설치: `winget install MiKTeX.MiKTeX`

### 필요한 패키지들
논문 컴파일에 필요한 추가 패키지들:
- IEEEtran (IEEE 논문 템플릿)
- graphicx (그림 삽입)
- booktabs (표 형식)
- subcaption (서브 캡션)
- amsmath, amssymb (수학 기호)

## 컴파일 순서

```bash
# 논문 디렉토리로 이동
cd papers/ICTC_2025

# LaTeX 컴파일 (3번 실행 권장)
pdflatex ICTC_2025_main.tex
bibtex ICTC_2025_main
pdflatex ICTC_2025_main.tex
pdflatex ICTC_2025_main.tex
```

## 현재 논문 상태

### ✅ 완성된 파일들
- `ICTC_2025_main.tex` - 메인 논문 파일
- `references.bib` - 참고문헌
- `cross_domain_section.tex` - Cross-domain 실험 섹션
- `figures/` - 모든 그림 파일들 (PDF, PNG)
- `tables/` - LaTeX 표 파일들
- `data/` - 실험 결과 데이터

### 📊 포함된 그림들
1. `performance_comparison.pdf` - 성능 비교 막대 그래프
2. `architecture_comparison.pdf` - 아키텍처 비교 다이어그램
3. `efficiency_scatter.pdf` - 파라미터 효율성 산점도
4. `convergence_analysis.pdf` - 수렴 분석 그래프
5. `resource_comparison.pdf` - 자원 사용량 비교

### 📋 포함된 표들
1. `performance_table.tex` - 성능 비교 표
2. `efficiency_table.tex` - 자원 효율성 표
3. `cross_domain_table.tex` - Cross-domain 실험 결과 표

## 실험 결과 업데이트 방법

실험이 완료되면 다음 파일들의 수치를 업데이트하세요:

### 1. 성능 데이터 업데이트
```bash
cd data
python extract_results.py  # 실제 실험 결과로 업데이트
```

### 2. 그림 재생성
```bash
cd figures
python generate_figures.py  # 업데이트된 데이터로 그림 재생성
```

### 3. 논문 텍스트 수정
- 초록의 성능 수치
- 결론의 핵심 결과
- 표와 그림의 수치

## 논문 구조

1. **Abstract** - LoRA vs Adapter 핵심 결과 요약
2. **Introduction** - 연구 동기 및 기여점
3. **Related Work** - DNN 채널 추정, PEFT 방법론
4. **Methodology** - 시스템 모델, 아키텍처, 학습 설정
5. **Experimental Setup** - 데이터셋, 구현 상세, 평가 지표
6. **Results and Analysis** - 성능 비교, 자원 효율성, 절제 연구
7. **Cross-Domain Transfer Learning** - 예비 실험 결과
8. **Industry Deployment Considerations** - 실용적 배포 고려사항
9. **Conclusion** - 핵심 발견 및 향후 연구

## 주요 기여점

1. **최초 비교 연구**: 무선 채널 추정에서 Adapter vs LoRA 최초 비교
2. **파라미터 효율성**: LoRA가 79% 적은 파라미터로 우수한 성능
3. **산업적 가치**: 실제 5G 배포를 위한 실용적 가이드라인
4. **Cross-domain 적용**: 다양한 환경 간 전이학습 가능성 제시

## 검토 체크리스트

- [ ] 모든 그림이 올바르게 참조됨
- [ ] 표의 수치가 일관됨
- [ ] 수식이 올바르게 렌더링됨
- [ ] 참고문헌이 적절히 인용됨
- [ ] 실험 결과 수치 업데이트 완료
- [ ] 영문 문법 및 표현 검토
- [ ] ICTC 형식 요구사항 준수