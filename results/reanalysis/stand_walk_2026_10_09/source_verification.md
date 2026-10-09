# MI mu/beta 및 복합사지 주장 독립 검증

검증일: 2026-10-09. 역할: 제안자의 분석을 전달받지 않은 별도 skeptic 하위 에이전트. 적용: claim-source-verification. 제공된 네 원문과 주장 연결만 검토했으며, 사용자 EEG 데이터나 구현을 직접 분석하지 않았다. 아래의 데이터 적용 한계는 제공된 과제 설명(Stand 대 양측 교대 Walk, Cz/CPz 고정 8-13 Hz)을 전제로 한 추론이다.

CONFIRMED는 명시한 접근 수준과 범위에서 확인했다는 뜻이다. REJECTED는 제안된 출처가 그 주장을 지지하지 않는다는 뜻이며, 주장 자체가 모든 조건에서 거짓이라는 뜻은 아니다. 부재 주장은 읽은 자료 범위 안에서만 판단했다.

## 주장별 판정

| ID | 검증한 주장 | 판정 | 근거/제한 |
| --- | --- | --- | --- |
| C01 | 1997: 손 MI 중 반대측 Rolandic mu/beta ERD 관찰 | CONFIRMED | 출판사 초록에서 확인. |
| C02 | 1997 전체 본문에는 MI 고유성/필요충분성 검증이 없다 | BLOCKED | 전문 미확보. 초록 밖의 분석 부재를 단정할 수 없음. |
| C03 | 1997은 mu 차이를 MI의 고유·충분·필수 판별지표로 입증한다 | REJECTED | 접근한 초록이 이 판별 성능·범용성을 지지하지 않음. |
| C04 | 2006: 발/혀 MI에서 손영역 mu ERS가 나타날 수 있다 | CONFIRMED | 초록 및 Results/Discussion 확인. |
| C05 | 2006은 Cz/CPz 고정 8-13 Hz 복합보행을 직접 검증했다 | REJECTED | 해당 센서·주파수·과제 조합의 직접 평가가 아님. |
| C06 | Ma 2023의 복합사지 분류는 LLRA 대 RLLA이다 | CONFIRMED | Methods II.A/II.E 및 Fig. 1에서 확인. |
| C07 | Ma 2023은 Stand 대 양측 교대 Walk 분리를 직접 보였다 | REJECTED | 평가한 두 클래스가 다름. |
| C08 | Ma 2023이 양측 교대의 비대칭이 반드시 상쇄됨을 입증한다 | REJECTED | 해당 교대 과제를 평가하지 않았음. |
| C09 | Ma 2023이 현재 구현에서 복합사지 효과가 도움이 될 수 없음을 입증한다 | REJECTED | 현재 구현을 평가하지 않았음. |
| C10 | 2013: MI를 억제한 nogo에서도 초기 beta ERD가 있다 | CONFIRMED | Results/Fig. 1/Fig. 2에서 확인. |
| C11 | 2013은 모든 mu/beta ERD가 cue만의 효과임을 입증한다 | REJECTED | 초기 beta와 지속 MI 관련 반응을 구분함. |
| C12 | 2013: 시각 cue에 유발된 준비 반응이 초기 beta ERD에 기여할 수 있다 | CONFIRMED | Discussion의 저자 해석으로 확인. |

총 세부 연결 12개: CONFIRMED 5, REJECTED 6, BLOCKED 1. REJECTED 사유: does_not_support_claim 3개(C03/C08/C11), wrong_task_or_evaluation_scope 3개(C05/C07/C09). 원래 1997 제안문 묶음은 C01과 C02가 섞여 있으므로 PARTIAL이며, 전체 확인으로 처리하지 않는다.

## S1. Pfurtscheller & Neuper 1997

- 논문: *Motor imagery activates primary sensorimotor area in humans*. Neuroscience Letters 239(2-3), 65-68. DOI: `10.1016/S0304-3940(97)00889-6`.
- 원문: [출판사 초록](https://www.sciencedirect.com/science/article/pii/S0304394097008896).
- 접근 수준: 초록과 서지 확인. 전문은 기관 접근/구매 필요로 확보하지 못했다.
- 짧은 발췌: “The Rolandic rhythms displayed an event-related desynchronization (ERD) only over the contralateral hemisphere.”
- 정확한 위치: Abstract, 반대측 ERD를 기술하는 네 번째 문장. 초록은 세 피험자, 시각 지시에 따른 좌/우 손 MI를 기술한다.
- 판정: 반대측 ERD 관찰은 CONFIRMED. 고유·필수·충분 판별성의 입증으로 쓰는 것은 REJECTED. 논문 전체에 그러한 검증이 없다는 부재 주장은 BLOCKED.
- 적용 한계: 손 MI의 관찰 결과를 보행 MI, Cz/CPz의 고정 대역, 개별 시행의 MI 수행 여부로 옮길 수 있다는 직접 근거가 아니다. 이는 접근 자료의 지지 범위를 제한하는 검증자 판단이다.

## S2. Pfurtscheller et al. 2006

- 논문: *Mu rhythm (de)synchronization and EEG single-trial classification of different motor imagery tasks*. NeuroImage 31, 153-159. DOI: `10.1016/j.neuroimage.2005.12.003`; [PubMed PMID 16443377](https://pubmed.ncbi.nlm.nih.gov/16443377/) 서지도 재검색으로 확인했다.
- 원문: [저자 호스팅 PDF](https://pub.ista.ac.at/~schloegl/publications/pfurtscheller2006.pdf).
- 접근 수준: 7쪽 PDF 본문 접근. Methods, Results, Discussion, Conclusion의 관련 내용 확인; 표의 개별 수치를 재분석하지 않았다.
- 짧은 발췌: “whereas an enhancement of the hand area mu rhythm was observed during foot or tongue motor imagery in the majority of the subjects.”
- 정확한 위치: p.153 Abstract. 방법은 pp.153-154, 주요 결과/해석은 pp.156-158, Table 1 및 Fig. 2.
- 확인 사실: 9명, 60 EEG 전극, 좌/우 손·양발·혀의 운동감각 MI. C3/Cz/C4의 반응 대역을 선택했고, 반응성 mu 성분은 9-14 Hz에 걸쳤다. 다수에서 손영역 ERS와 발 MI의 midcentral ERD를 함께 보고한다.
- 판정: 손영역 ERS 가능성은 CONFIRMED. 이를 발 MI의 일반적인 mu 증가나 mu ERD 부재로 바꾸면 지지 범위를 벗어난다. Cz/CPz 고정 8-13 Hz의 양측 교대 보행 효과를 직접 평가한 연구라는 주장은 REJECTED.
- 적용 한계: 전극 위치, 대역 선택, 동작 종류가 다르다. 현재 데이터의 ERD 존재 여부나 분류 가능성은 별도 실증이 필요하다는 검증자 추론이다.

## S3. Ma et al. 2023

- 논문: *A New Compound-Limbs Paradigm: Integrating Upper-Limb Swing Improves Lower-Limb Stepping Intention Decoding From EEG*. IEEE TNSRE 31, 3823-3834. DOI: `10.1109/TNSRE.2023.3315717`; [PubMed PMID 37713229](https://pubmed.ncbi.nlm.nih.gov/37713229/).
- 원문: [ResearchGate 본문 사본](https://www.researchgate.net/publication/373966293_A_New_Compound-limbs_Paradigm_Integrating_upper-limb_swing_improves_lower-limb_stepping_intention_decoding_from_EEG).
- 접근 수준: 본문 텍스트의 Methods/Results/Discussion/Conclusion 확인. IEEE 출판본 PDF의 모든 표/그림을 별도로 대조한 것은 아니다.
- 짧은 발췌 1: “LLRA: left leg with right arm; RLLA: right leg with left arm.”
- 짧은 발췌 2: “participants executed or imagined movement only once”
- 정확한 위치: p.3825 Fig. 1 caption 및 II.A; 좌우 분류 정의는 p.3827 II.E 마지막 문단. 결과는 pp.3830-3831 III.D, 범위와 가설은 pp.3832-3833 IV.
- 확인 사실: 앉은 자세에서 단발 동작/상상을 평가하며, 복합사지 두 클래스는 LLRA/RLLA이다. MI의 평균 성능 개선을 보고하지만 모든 분석 조합에서 유의하지 않았다.
- 판정: mirrored 좌우 과제 확인은 CONFIRMED. Stand/양측교대 Walk 직접 분리, 필연적 상쇄, 현재 구현의 이득 불가능이라는 귀속은 모두 REJECTED.
- 적용 한계: 저자의 활성 중첩 설명도 가설이다. 측성 차이가 시간 평균에서 약해질 가능성은 현재 과제에서 검증할 가설일 뿐이며, 전체 파워·공간·시간 정보가 반드시 없어짐을 뜻하지 않는다. 이것은 원문 결과와 구분한 검증자 판단이다.

## S4. Pfurtscheller et al. 2013

- 논문: *Brisk heart rate and EEG changes during execution and withholding of cue-paced foot motor imagery*. Frontiers in Human Neuroscience 7:379. DOI: `10.3389/fnhum.2013.00379`.
- 원문: [출판사 본문](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2013.00379/full). 제공된 [PMC 페이지](https://pmc.ncbi.nlm.nih.gov/articles/PMC3726939/)는 접근 시 reCAPTCHA가 표시되어 출판사 본문으로 확인했다.
- 접근 수준: 출판사 본문의 Methods/Results/Discussion/Conclusion 확인.
- 짧은 발췌 1: “the early beta ERD during MInogo”
- 짧은 발췌 2: “triggered by the visual stimulus and modulated by the cognitive task.”
- 정확한 위치: 발췌 1은 Results의 Fig. 1 caption; 발췌 2는 Discussion > EEG changes during execution (go) and withholding (nogo) of MI 마지막 문장.
- 확인 사실: 16명에서 시각 go/nogo 신호에 따른 양발 dorsiflexion MI를 평가했다. 초기 beta ERD는 양 조건에 있고 MIgo에서 더 컸다. 저자는 초기 반응, 더 오래 지속되는 MI 관련 ERD, 후속 beta ERS를 구분한다.
- 판정: nogo 초기 beta ERD와 cue 관련 준비 반응 해석은 CONFIRMED. 모든 mu/beta 반응을 cue만으로 설명한다는 주장은 REJECTED.
- 적용 한계: 이는 초기 beta 결과이지 고정 8-13 Hz mu의 직접 검증이 아니다. 실제 nogo의 정신 상태도 확정되지 않았다. 현재 데이터에서 MI 고유성을 주장하려면 cue 시점과 이후 반응을 구분해야 한다는 검증자 추론이다.

## 미해결 사항과 재사용 경고

- 1997 전문을 확보하기 전에는 C02를 완료 처리하지 않는다. 안전하게 인용할 수 있는 범위는 출판사 초록에 명시된 손 MI의 반대측 ERD 관찰이다.
- 2006의 손영역 ERS를 Cz/CPz 발영역 ERS로 바꾸지 않는다. 같은 원문에 midcentral ERD도 보고되어 있다.
- Ma 2023의 좌우 복합동작 판별 성과를 Stand/Walk 성과로 바꾸거나, 과제 차이를 현재 구현의 실패 증명으로 바꾸지 않는다.
- 2013의 초기 beta 결과를 모든 mu ERD의 부정으로 확대하지 않는다. 이 네 문헌만으로 사용자 시행에서 MI가 실제 수행되었는지 확정할 수 없다.
