package example.contractcheck;

import com.sun.net.httpserver.HttpServer;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import example.contractcheck.ContractCheckResearchClient.Failure;
import example.contractcheck.ContractCheckResearchClient.ResearchCallException;

/** Dummy transport tests only. Never reads any real contract or review dataset. */
public class ClientProbe {
    private static final String VERSION = "cv_20261116_seed42/fold_1";
    private static final String ID = "integration-dummy-001";
    private static final String VALID = "{\"clause_id\":\"" + ID + "\",\"model_version\":\"" + VERSION
            + "\",\"fold\":1,\"weak_label\":0,\"weak_label_name\":\"AFTER_CORRECTION_SUMMARY_WEAK\","
            + "\"before_summary_model_score\":0.25,\"input_tokens\":9,\"research_only\":true,\"warning\":\"transport fixture only\"}";
    private static void require(boolean ok, String message) { if (!ok) throw new AssertionError(message); }
    private static void expect(Failure kind, Runnable action) {
        try { action.run(); throw new AssertionError("Expected " + kind); }
        catch (ResearchCallException e) {
            require(e.failure() == kind, "Unexpected failure " + e.failure());
            require(!e.getMessage().contains("PRIVATE_FIXTURE"), "Remote body leaked");
        }
    }
    public static void main(String[] args) throws Exception {
        if (args.length == 2 && args[0].equals("live")) { live(args[1]); return; }
        if (args.length == 1 && args[0].equals("self-test")) { selfTest(); return; }
        throw new IllegalArgumentException("Use self-test OR live http://127.0.0.1:8000");
    }
    private static void live(String url) {
        ContractCheckResearchClient c = new ContractCheckResearchClient(url, 3000, 30000, VERSION);
        require(c.ready().researchOnly(), "Missing research flag");
        var result = c.classify(ID, "개발 연동 확인용 입력입니다.");
        require(result.researchOnly() && result.clauseId().equals(ID), "Response identity mismatch");
        expect(Failure.REQUEST_REJECTED, () -> c.classify(ID, "개발 확인 입력 ".repeat(300)));
        System.out.println("{\"status\":\"PASS\",\"live_checks\":3,\"fold\":" + result.fold()
                + ",\"weak_label\":" + result.weakLabel() + ",\"class_1_score\":" + result.beforeSummaryModelScore()
                + ",\"research_only\":true,\"actual_contracts_used\":false}");
    }
    private static void selfTest() throws Exception {
        AtomicInteger code = new AtomicInteger(200);
        AtomicInteger delay = new AtomicInteger(0);
        AtomicReference<String> body = new AtomicReference<>(VALID);
        ExecutorService executor = Executors.newCachedThreadPool();
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.setExecutor(executor);
        server.createContext("/", exchange -> {
            try {
                if (delay.get() > 0) Thread.sleep(delay.get());
                byte[] data = body.get().getBytes(StandardCharsets.UTF_8);
                exchange.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
                exchange.sendResponseHeaders(code.get(), data.length); exchange.getResponseBody().write(data);
            } catch (InterruptedException e) { Thread.currentThread().interrupt();
            } catch (java.io.IOException ignored) { /* expected if the timeout fixture closes its connection */
            } finally { exchange.close(); }
        });
        server.start();
        String url = "http://127.0.0.1:" + server.getAddress().getPort();
        List<String> passed = new ArrayList<>();
        try {
            ContractCheckResearchClient c = new ContractCheckResearchClient(url, 1000, 1000, VERSION);
            require(c.classify(ID, "개발 확인").weakLabel() == 0, "Success DTO"); passed.add("success");
            expect(Failure.REQUEST_REJECTED, () -> c.classify(ID, " ")); passed.add("blank_local");
            body.set("{\"detail\":\"PRIVATE_FIXTURE\"}");
            for (int status : new int[] {422, 503, 500}) {
                code.set(status);
                Failure expected = status == 422 ? Failure.REQUEST_REJECTED : status == 503 ? Failure.MODEL_UNAVAILABLE : Failure.REMOTE_HTTP;
                expect(expected, () -> c.classify(ID, "개발 확인")); passed.add("http_" + status);
            }
            code.set(200);
            for (String invalid : new String[] {
                    VALID.replace("\"research_only\":true", "\"research_only\":false"),
                    VALID.replace(ID, "different-id"), VALID.replace("\"weak_label\":0", "\"weak_label\":3"),
                    VALID.replace("0.25", "1.25"), "{}"}) {
                body.set(invalid); expect(Failure.INVALID_RESPONSE, () -> c.classify(ID, "개발 확인")); passed.add("invalid_response");
            }
            body.set(VALID); delay.set(350);
            ContractCheckResearchClient impatient = new ContractCheckResearchClient(url, 1000, 50, VERSION);
            expect(Failure.TIMEOUT, () -> impatient.classify(ID, "개발 확인")); passed.add("timeout");
        } finally { server.stop(0); executor.shutdownNow(); }
        ContractCheckResearchClient offline = new ContractCheckResearchClient(url, 300, 300, VERSION);
        expect(Failure.CONNECTION_OR_IO, () -> offline.classify(ID, "개발 확인")); passed.add("connection_failure");
        System.out.println("{\"status\":\"PASS\",\"self_checks\":" + passed.size()
                + ",\"actual_contracts_used\":false,\"training_run\":false}");
    }
}
