package example.contractcheck;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpTimeoutException;
import java.net.SocketTimeoutException;
import java.time.Duration;
import java.util.Map;
import java.util.concurrent.TimeoutException;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.MediaType;
import org.springframework.http.client.JdkClientHttpRequestFactory;
import org.springframework.stereotype.Service;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestClientResponseException;

/** Research-only transport adapter. It makes no legal-risk decision or DB write. */
@Service
public class ContractCheckResearchClient {
    private static final ParameterizedTypeReference<Map<String, Object>> BODY = new ParameterizedTypeReference<>() {};
    private final RestClient client;
    private final String expectedVersion;

    public ContractCheckResearchClient(
            @Value("${contractcheck.ai.base-url:http://127.0.0.1:8000}") String baseUrl,
            @Value("${contractcheck.ai.connect-timeout-ms:3000}") long connectMs,
            @Value("${contractcheck.ai.response-timeout-ms:30000}") long responseMs,
            @Value("${contractcheck.ai.expected-model-version:cv_20261116_seed42/fold_1}") String expectedVersion) {
        URI base = URI.create(baseUrl);
        if (!("http".equals(base.getScheme()) || "https".equals(base.getScheme())) || base.getHost() == null
                || base.getUserInfo() != null || base.getQuery() != null || base.getFragment() != null
                || !(base.getPath().isEmpty() || base.getPath().equals("/")) || connectMs <= 0 || responseMs <= 0
                || expectedVersion == null || expectedVersion.isBlank()) {
            throw new IllegalArgumentException("Set an origin base URL and positive timeouts");
        }
        HttpClient transport = HttpClient.newBuilder().connectTimeout(Duration.ofMillis(connectMs)).build();
        JdkClientHttpRequestFactory factory = new JdkClientHttpRequestFactory(transport);
        factory.setReadTimeout(Duration.ofMillis(responseMs));
        this.client = RestClient.builder().baseUrl(baseUrl).requestFactory(factory).build();
        this.expectedVersion = expectedVersion;
    }

    public Ready ready() {
        try {
            Map<String, Object> body = client.get().uri("/dev/v1/legacy/health/ready").retrieve().body(BODY);
            verifyResearchModel(body);
            if (!"ready".equals(body.get("status")) || integer(body, "max_tokens") != 192) throw invalid();
            return new Ready(string(body, "model_version"), integer(body, "fold"), 192, true);
        } catch (RestClientResponseException e) { throw httpFailure(e.getStatusCode().value());
        } catch (ResourceAccessException e) { throw ioFailure(e);
        } catch (RestClientException e) { throw invalid(); }
    }

    public ResearchResult classify(String clauseId, String text) {
        if (clauseId == null || text == null || clauseId.isBlank() || text.isBlank()) {
            throw new ResearchCallException(Failure.REQUEST_REJECTED, null);
        }
        String id = clauseId.strip();
        try {
            Map<String, Object> body = client.post().uri("/dev/v1/legacy/clauses/classify")
                    .contentType(MediaType.APPLICATION_JSON).accept(MediaType.APPLICATION_JSON)
                    .body(Map.of("clause_id", id, "text", text.strip())).retrieve().body(BODY);
            verifyResearchModel(body);
            int label = integer(body, "weak_label");
            if (label != 0 && label != 1) throw invalid();
            String name = label == 0 ? "AFTER_CORRECTION_SUMMARY_WEAK" : "BEFORE_CORRECTION_SUMMARY_WEAK";
            if (!id.equals(string(body, "clause_id")) || !name.equals(string(body, "weak_label_name"))) throw invalid();
            Object rawScore = body.get("before_summary_model_score");
            if (!(rawScore instanceof Number)) throw invalid();
            double score = ((Number) rawScore).doubleValue();
            int tokens = integer(body, "input_tokens");
            if (!Double.isFinite(score) || score < 0 || score > 1 || tokens < 1 || tokens > 192) throw invalid();
            return new ResearchResult(id, string(body, "model_version"), integer(body, "fold"), label, name,
                    score, tokens, true, string(body, "warning"));
        } catch (RestClientResponseException e) { throw httpFailure(e.getStatusCode().value());
        } catch (ResourceAccessException e) { throw ioFailure(e);
        } catch (RestClientException e) { throw invalid(); }
    }

    private void verifyResearchModel(Map<String, Object> body) {
        if (body == null || !Boolean.TRUE.equals(body.get("research_only"))
                || !expectedVersion.equals(string(body, "model_version")) || integer(body, "fold") != 1) throw invalid();
    }
    private static String string(Map<String, Object> body, String key) {
        Object value = body.get(key);
        if (!(value instanceof String) || ((String) value).isBlank()) throw invalid();
        return (String) value;
    }
    private static int integer(Map<String, Object> body, String key) {
        Object value = body.get(key);
        if (!(value instanceof Number)) throw invalid();
        Number n = (Number) value;
        if (!Double.isFinite(n.doubleValue()) || n.doubleValue() != n.intValue()) throw invalid();
        return n.intValue();
    }
    private static ResearchCallException httpFailure(int status) {
        return new ResearchCallException(status == 422 ? Failure.REQUEST_REJECTED
                : status == 503 ? Failure.MODEL_UNAVAILABLE : Failure.REMOTE_HTTP, status);
    }
    private static ResearchCallException ioFailure(Throwable error) {
        Throwable t = error;
        for (int i = 0; t != null && i < 32; i++, t = t.getCause()) {
            if (t instanceof HttpTimeoutException || t instanceof SocketTimeoutException || t instanceof TimeoutException) {
                return new ResearchCallException(Failure.TIMEOUT, null);
            }
        }
        return new ResearchCallException(Failure.CONNECTION_OR_IO, null);
    }
    private static ResearchCallException invalid() { return new ResearchCallException(Failure.INVALID_RESPONSE, null); }

    public enum Failure { REQUEST_REJECTED, MODEL_UNAVAILABLE, REMOTE_HTTP, TIMEOUT, CONNECTION_OR_IO, INVALID_RESPONSE }
    /** Deliberately omits the raw remote body/cause: validation errors may echo input text. */
    public static final class ResearchCallException extends RuntimeException {
        private final Failure failure;
        private final Integer remoteStatus;
        public ResearchCallException(Failure failure, Integer remoteStatus) {
            super("ContractCheck " + failure + (remoteStatus == null ? "" : " HTTP " + remoteStatus));
            this.failure = failure; this.remoteStatus = remoteStatus;
        }
        public Failure failure() { return failure; }
        public Integer remoteStatus() { return remoteStatus; }
    }
    public record Ready(String modelVersion, int fold, int maxTokens, boolean researchOnly) {}
    // weakLabel must not be copied into operational isRisky/riskLevel fields.
    public record ResearchResult(String clauseId, String modelVersion, int fold, int weakLabel,
            String weakLabelName, double beforeSummaryModelScore, int inputTokens, boolean researchOnly, String warning) {}
}
