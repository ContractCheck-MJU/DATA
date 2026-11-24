# 팀 환경 연동 체크리스트

- [ ] **설치 위치 확인.** [연동 가이드](BACKEND_API.md)에서 GitHub 또는 모델 ZIP 설치 절차를 선택하고 `pip check`를 실행한다. 저장소 경로를 선택했다면 별도 전달받은 fold 1 체크포인트를 배치한다.
- [ ] **Python 준비 상태.** 서버 실행 후 `GET /dev/v1/legacy/health/ready`가 200이고 `fold=1`, `model_version=cv_20261116_seed42/fold_1`, `research_only=true`인지 확인한다.
- [ ] **더미 요청.** `POST /dev/v1/legacy/clauses/classify`에 가이드의 더미 ID와 텍스트를 보내고 응답의 ID를 대조한다.
- [ ] **Java 단독 호출.** [Spring 예제](../spring-client/README.md)의 `ClientProbe self-test`로 연결 실패·타임아웃·오류 매핑을 검사한다. 서버를 실행한 상태에서 `ClientProbe live`로 준비 상태, 더미 추론, 토큰 한도 초과를 확인한다.
- [ ] **팀 앱 경유.** 팀 Spring Boot 설정에 Python 주소를 등록하고 개발용 DTO로 응답을 받는다. 422·503·연결 실패를 구분하고, 원문·오류 본문은 로그에서 제외한다.
- [ ] **결과 표시.** 약라벨 0/1을 시정 후/전 요약문 클래스로 표시하고 연구용 표시와 모델 버전을 유지한다. 팀 화면과 DB에 법률 위험 등급이나 확정 판정으로 저장되는 곳이 없는지 확인한다.
- [ ] **서로 다른 호스트.** 컨테이너나 다른 PC에서 호출할 때는 Spring의 `CONTRACTCHECK_AI_BASE_URL`, Python 바인딩 주소, 개발망 접근 권한을 함께 확인한다. 기본 바인딩은 루프백이며 API 자체에는 인증·TLS가 없다.
