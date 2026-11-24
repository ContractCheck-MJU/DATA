# Java Spring HTTP 클라이언트 예제

Java17 이상, Spring Framework6.1 이상(RestClient 사용)을 기준으로 한다. 팀의 Spring Boot/Java 버전을 먼저 확인한다. 현재 프로젝트 버전에 RestClient가 없다면 팀의 기존 HTTP 클라이언트로 같은 요청/응답 규약을 구현한다. 이 예제를 위해 전체 백엔드 버전을 변경할 필요는 없다.

## 팀 프로젝트에 적용

1. `ContractCheckResearchClient.java`의 package를 팀 패키지로 맞춰 Spring component scan 범위에 둔다. `@Service` 단일 생성자가 환경 설정을 받는다.
2. `application-example.yml`의 `contractcheck.ai` 설정을 팀 설정에 추가한다. base URL은 환경 변수 `CONTRACTCHECK_AI_BASE_URL`로 바꿀 수 있다. 같은 PC의 Python은 `http://127.0.0.1:8000`, 컨테이너/다른 PC는 합의한 개발망 주소를 사용한다.
3. 기존 백엔드의 Spring Boot BOM과 의존성 관리를 유지한다. 이 폴더의 pom은 **독립 예제 검증용**이며 현재 팀 프로젝트 pom을 대체하지 않는다. Spring MVC/JSON 변환기 의존성이 필요하다.
4. 서비스에서 아래처럼 호출한다. 외부 사용자용 API 경로·응답·DB 저장 형식은 팀에서 정한다.

```java
// 생성자 주입받은 researchClient 사용
var ready = researchClient.ready();
var result = researchClient.classify("integration-dummy-001", "개발 연동 확인용 입력입니다.");
// result.weakLabel(): 0/1 약라벨. 정상/위험이나 법률 위험 등급으로 변환하지 않는다.
// result.beforeSummaryModelScore(): 항상 클래스1 softmax 점수.
// result.researchOnly(): true. 개발 결과 DTO에 이 표시와 버전/fold를 보존한다.
```

예제는 fold1/예상 버전, `research_only=true`, 응답 clause ID, 클래스 명칭, 점수 범위와 토큰 수를 검증한다. 성공 결과를 운영 `risk_results`의 확정 판정으로 저장하는 코드는 없다. 실패를 정상0으로 대체하지 않으며 자동 재시도하지 않는다.

## 연결 실패·타임아웃 처리

`ResearchCallException.failure()`로 분기한다. `remoteStatus()`는 실제 원격 HTTP 오류일 때만 값이 있다. 아래 백엔드 응답 매핑은 제안이며 팀의 최종 API와 맞춘다.

|failure|의미|백엔드 처리 제안|
|---|---|---|
|REQUEST_REJECTED|빈 로컬 입력 또는 원격422(문자/토큰 제한 포함)|조항 분리·요청 수정. 사용자용422 등으로 명확히 안내|
|MODEL_UNAVAILABLE|원격503, 연구 모델 미설정|분석 이용 불가로 표시하고 설정 확인|
|TIMEOUT|연결/응답 대기 중 timeout|504 등으로 표시. 정상 라벨을 반환하지 않음|
|CONNECTION_OR_IO|주소/포트/서버 미실행/연결 오류|502/503 등으로 표시하고 주소·ready 확인|
|REMOTE_HTTP|그 밖의 원격4xx/5xx|502 등으로 구분하고 서버 상태 확인|
|INVALID_RESPONSE|연구 표시/버전/ID/스키마/점수 이상|502 등으로 처리하고 통합 설정 점검|

연결3초·응답30초는 개발 초기값이다. 시작 중 모델 로딩은 ready가 될 때까지 기다리고, CPU 처리량에 맞춰 협의한다. 한 요청의 timeout은 서버 작업 취소 보장과 다르다. 원격 detail/예외 메시지에는 원문이 들어갈 수 있어 입력·응답 본문을 그대로 로그에 남기지 않는다. 예제 예외는 원문 응답/원인 메시지를 외부로 노출하지 않는다.

## 독립 예제 실행

JDK와 Maven이 있는 환경에서 이 폴더로 이동한다. 최초 실행에는 Maven Central 접근이 필요하다.

```powershell
mvn -B compile dependency:build-classpath '-Dmdep.outputFile=target/classpath.txt'
$cp = 'target/classes;' + (Get-Content target/classpath.txt -Raw).Trim()
java -cp $cp example.contractcheck.ClientProbe self-test
# Python 모델 서버가 실행된 뒤
java -cp $cp example.contractcheck.ClientProbe live http://127.0.0.1:8000
```

macOS/Linux에서는 classpath 구분자를 `;` 대신 `:`로 사용한다. Python ZIP의 설치는 Windows Python3.12 x64를 기준으로 검증돼 있으며 이번에 macOS/Linux에서 재검증하지 않았다.

이번 로컬 검증: JBR25.0.3에서 Java17 대상으로 컴파일, Spring6.2.19/Jackson2.23.0. 연결 실패·timeout·422/503/500·위험한 응답 거부 등 모의검사12개 PASS, 실제 원본 ZIP의 FastAPI에 ready·더미 추론·토큰초과 처리3개 PASS. 팀 전체 Spring Boot 앱/DB 배포를 검증한 것은 아니다. 이 폴더의 테스트 문장은 통신용 fixture이며 실제 계약/학습 데이터가 아니다.

공식 API 근거: [Spring RestClient](https://docs.spring.io/spring-framework/reference/6.2/integration/rest-clients.html), [JdkClientHttpRequestFactory 및 timeout](https://docs.spring.io/spring-framework/docs/6.1.21/javadoc-api/org/springframework/http/client/JdkClientHttpRequestFactory.html).
