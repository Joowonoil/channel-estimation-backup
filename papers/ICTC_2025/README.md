# ICTC 2025 논문: Parameter-Efficient Transfer Learning for DNN-based Channel Estimation

## 논문 정보
- **제목**: Parameter-Efficient Transfer Learning for DNN-based Channel Estimation: Adapter vs LoRA Comparison
- **저자**: Joo Won Lee
- **소속**: Department of Electronic and Electrical Engineering, Sungkyunkwan University
- **이메일**: joowonoil@skku.edu
- **학회**: ICTC 2025 (International Conference on Information and Communication Technology Convergence)

## 파일 구조
```
ICTC_2025/
├── ICTC_2025_main.tex          # 메인 LaTeX 파일
├── references.bib              # 참고문헌
├── README.md                   # 이 파일
├── figures/                    # 그림 파일들
│   ├── architecture_comparison.pdf
│   ├── performance_comparison.pdf
│   └── convergence_analysis.pdf
├── data/                       # 실험 데이터
│   ├── performance_results.csv
│   └── parameter_efficiency.csv
└── tables/                     # 표 데이터
    ├── performance_table.tex
    └── efficiency_table.tex
```

## 컴파일 방법
```bash
cd papers/ICTC_2025
pdflatex ICTC_2025_main.tex
bibtex ICTC_2025_main
pdflatex ICTC_2025_main.tex
pdflatex ICTC_2025_main.tex
```

## 주요 기여점
1. 무선 채널 추정 분야 최초의 Adapter vs LoRA 종합 비교
2. LoRA가 76.8% 파라미터 감소와 함께 우수한 성능 달성
3. 산업 배포를 위한 실용적 가이드라인 제시
4. Cross-domain transfer learning의 예비 결과 제시

## 실험 결과 요약
- **LoRA**: 26K 파라미터로 평균 2.5dB 성능 향상
- **Adapter**: 131K 파라미터로 평균 1.8dB 성능 향상
- **효율성**: LoRA가 메모리 17% 감소, 추론 17% 속도 향상