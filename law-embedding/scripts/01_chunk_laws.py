#!/usr/bin/env python3
"""Conservative statute-only chunking; no embeddings, DB writes, or legal judgments."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

MODEL_ID = "BAAI/bge-m3"
TOKENIZER_REVISION = "3069def033ce91d907258f9a830e442610dbfe0b"
RULE_VERSION = "law-semantic-conservative-v2"
TARGET_TOKENS = 512
CIRCLES = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_input(filename):
    records = [json.loads(line) for line in Path(filename).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    laws = [record for record in records if record.get("document_type") == "statute"]
    manifest_path = Path(filename).with_suffix(".manifest.json")
    if not manifest_path.exists():
        raise ValueError("입력 원문 검증용 manifest 파일이 필요합니다: " + str(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != manifest["file_sha256"]:
        raise ValueError("입력 파일 SHA-256 불일치")
    if len(laws) != manifest["statute_count"] or sorted(d["id"] for d in laws) != sorted(manifest["document_ids"]):
        raise ValueError("대상 법령 수/ID 목록 불일치")
    if not laws or len({d["id"] for d in laws}) != len(laws):
        raise ValueError("법령이 없거나 문서 ID가 중복됩니다")
    for d in laws:
        if not isinstance(d.get("source_text"), str) or not d["source_text"].strip():
            raise ValueError("원문 누락: " + d["id"])
    return laws


def paragraph_spans(text):
    starts = [0]
    for match in re.finditer(r"(?m)^[ \t]*([" + CIRCLES + r"])", text):
        if match.start(1) > 0:
            starts.append(match.start(1))
    starts = sorted(set(starts))
    spans = list(zip(starts, starts[1:] + [len(text)]))
    labels = {}
    for i, (a, b) in enumerate(spans):
        fragment = text[a:b]
        match = re.search("[" + CIRCLES + "]", fragment.split("\n", 1)[0]) if i == 0 else re.match("[" + CIRCLES + "]", fragment)
        if match:
            labels[CIRCLES.index(match.group()) + 1] = (a, b)
    return spans, labels


def merge_spans(spans):
    result = []
    for a, b in sorted(set(spans)):
        if not result or a > result[-1][1]:
            result.append([a, b])
        else:
            result[-1][1] = max(result[-1][1], b)
    return [tuple(pair) for pair in result]


def item_atoms(text, start, end):
    hits = [start + m.start() for m in re.finditer(r"(?m)^[ \t]*\d+(?:의\d+)?\.(?:[ \t]+)", text[start:end])]
    hits = sorted(set(p for p in hits if start < p < end))
    if not hits:
        return [(start, end)], None
    lead = (start, hits[0]) if text[start:hits[0]].strip() else None
    starts = [start] + hits
    return list(zip(starts, starts[1:] + [end])), lead


def reference_context(text, start, end, labels):
    # 제N항은 다른 법/조문을 가리킬 수 있어 자동 연결하지 않는다.
    # 현 조문 안의 '전항'만 보수적으로 직전 항 문맥 후보로 기록한다.
    fragment = text[start:end]
    current = [n for n, (a, b) in labels.items() if a <= start < b]
    if "전항" in fragment and current and current[0] > 1:
        previous = labels.get(current[0] - 1)
        if previous and not (previous[0] >= start and previous[1] <= end):
            return [previous]
    return []


def reference_mentions(text, start, end):
    result = []
    for match in re.finditer(r"(?:제\d+조(?:의\d+)?(?:제\d+항)?|제\d+항|전항)", text[start:end]):
        a, b = start + match.start(), start + match.end()
        result.append({"mention": match.group(), "start": a, "end": b, "surrounding_text": text[max(0, a - 35):min(len(text), b + 35)], "resolution_status": "not_automatically_resolved_review_required"})
    return result


def run(input_file, output_dir, target=TARGET_TOKENS):
    from transformers import AutoTokenizer
    laws = load_input(input_file)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=TOKENIZER_REVISION, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("fast tokenizer가 필요합니다")
    model_limit = int(tokenizer.model_max_length)
    if model_limit > 100000 or not 0 < target <= model_limit:
        raise ValueError("모델 최대 길이 또는 목표 길이를 확인하세요")

    def count(value):
        return len(tokenizer(value, add_special_tokens=True, truncation=False, return_attention_mask=False)["input_ids"])

    config_hash = sha(json.dumps({"target": target, "model": MODEL_ID, "revision": TOKENIZER_REVISION, "rules": RULE_VERSION}, sort_keys=True))[:12]
    chunks, reviews, docstats = [], [], []
    for document in laws:
        text = document["source_text"]
        meta = document.get("metadata", {})
        prefix = "[법령] " + document["title"] + "\n"
        paragraphs, labels = paragraph_spans(text)

        def embedding_text(a, b, contexts):
            context = ""
            for ca, cb in contexts:
                if ca >= a and cb <= b:
                    continue
                context += "[같은 조문 문맥·원문]\n" + text[ca:cb] + "\n"
            return prefix + context + "[검색 대상 원문]\n" + text[a:b]

        units = []
        if count(embedding_text(0, len(text), [])) <= target:
            units = [(0, len(text), [], "whole_article")]
        else:
            for pa, pb in paragraphs:
                pc = reference_context(text, pa, pb, labels)
                if count(embedding_text(pa, pb, pc)) <= target:
                    units.append((pa, pb, pc, "paragraph"))
                    continue
                atoms, lead = item_atoms(text, pa, pb)
                if len(atoms) == 1:
                    units.append((pa, pb, pc, "oversize_semantic_unit_preserved"))
                    continue
                current = None
                for aa, bb in atoms:
                    if current is None:
                        current = (aa, bb)
                        continue
                    ta, tb = current[0], bb
                    tc = reference_context(text, ta, tb, labels)
                    if lead and not (lead[0] >= ta and lead[1] <= tb):
                        tc = merge_spans(tc + [lead])
                    if count(embedding_text(ta, tb, tc)) <= target or (lead and current == lead):
                        current = (ta, tb)
                    else:
                        ca, cb = current
                        cc = reference_context(text, ca, cb, labels)
                        if lead and not (lead[0] >= ca and lead[1] <= cb):
                            cc = merge_spans(cc + [lead])
                        units.append((ca, cb, cc, "numbered_items_with_parent_context"))
                        current = (aa, bb)
                if current:
                    ca, cb = current
                    cc = reference_context(text, ca, cb, labels)
                    if lead and not (lead[0] >= ca and lead[1] <= cb):
                        cc = merge_spans(cc + [lead])
                    units.append((ca, cb, cc, "numbered_items_with_parent_context"))
        if not units or units[0][0] != 0 or units[-1][1] != len(text):
            raise ValueError("원문 시작/끝 범위 누락: " + document["id"])
        if not all(units[i][1] == units[i + 1][0] for i in range(len(units) - 1)):
            raise ValueError("원문 범위 누락/중복: " + document["id"])
        source_hash = sha(text)
        for number, (a, b, contexts, method) in enumerate(units, 1):
            contexts = merge_spans(contexts)
            embedded_input = embedding_text(a, b, contexts)
            tokens = count(embedded_input)
            body = text[a:b]
            flags = []
            if tokens > target:
                flags.append("target_exceeded_semantic_context_preserved")
            if tokens > model_limit:
                flags.append("model_limit_exceeded_do_not_embed")
            if contexts:
                flags.append("same_article_context_added_reference_scope_review")
            if re.search(r"제\d+항|전항", body):
                flags.append("paragraph_reference_scope_review")
            for phrase, flag in [("다만", "proviso_included_review_scope"), ("준용", "cross_article_application_reference_review"), ("별표", "annex_reference_review")]:
                if phrase in body:
                    flags.append(flag)
            key = document["id"] + "@" + str(meta.get("source_version_seq", "unversioned")) + ":" + source_hash[:12] + ":" + config_hash + ":" + str(a) + "-" + str(b)
            record = {
                "chunk_key": key, "document_key": document["id"], "chunk_no": number,
                "text": body, "embedding_text": embedded_input,
                "source_span": {"start": a, "end": b, "unit": "unicode_code_points", "interval": "start_inclusive_end_exclusive"},
                "context_spans": [{"start": ca, "end": cb, "text": text[ca:cb], "role": "same_article_context_candidate"} for ca, cb in contexts],
                "source_text_sha256": source_hash, "token_count": tokens,
                "body_token_count": count(body), "token_count_includes_special_tokens": True,
                "target_tokens": target, "model_max_tokens": model_limit,
                "tokenizer_model": MODEL_ID, "tokenizer_revision": TOKENIZER_REVISION,
                "tokenizer_is_fast": True, "rule_version": RULE_VERSION, "chunk_config_hash": config_hash,
                "split_method": method, "review_flags": flags,
                "reference_mentions": reference_mentions(text, a, b),
                "source_url": document["source_url"], "law_name": meta.get("law_name"),
                "article": meta.get("article"), "article_title": meta.get("article_title"),
                "source_version_seq": meta.get("source_version_seq"),
                "source_effective_date": document.get("source_effective_date"),
                "provision_effective_date": meta.get("provision_effective_date"),
                "instrument_type": meta.get("instrument_type"),
                "collected_date": document.get("collected_date"),
                "application_caution": meta.get("application_caution"),
                "scope_note": "원문 청킹 자료. 위험/안전 정답·개별 효력판정 아님. 참조조문 자동해결 미실시.",
                "db_id": None, "db_document_id": None, "embedding": None,
                "embedding_status": "not_generated"
            }
            if record["text"] != text[a:b] or not all(c["text"] == text[c["start"]:c["end"]] for c in record["context_spans"]):
                raise ValueError("원문 부분문자열 검증 실패")
            if count(record["embedding_text"]) != record["token_count"]:
                raise ValueError("토큰 수 재검증 실패")
            chunks.append(record)
            reviews.append({"chunk_key": key, "document_key": document["id"], "title": document["title"], "chunk_no": number, "token_count": tokens, "target_tokens": target, "start": a, "end": b, "method": method, "review_flags": ";".join(flags), "source_url": document["source_url"]})
        docstats.append({"document_key": document["id"], "title": document["title"], "source_characters": len(text), "source_text_sha256": source_hash, "chunks": len(units), "whole_article_tokens": count(embedding_text(0, len(text), [])), "coverage_complete": True})
    if len({c["chunk_key"] for c in chunks}) != len(chunks):
        raise ValueError("청크 키 중복")
    for d in laws:
        if "".join(c["text"] for c in chunks if c["document_key"] == d["id"]) != d["source_text"]:
            raise ValueError("원문 재조립 검증 실패: " + d["id"])
    audit = {
        "created_at": datetime.now(timezone.utc).isoformat(), "source_documents": len(laws),
        "source_filter": "document_type=statute only", "chunks": len(chunks),
        "target_tokens": target, "model_max_tokens": model_limit,
        "max_token_count": max(c["token_count"] for c in chunks),
        "target_exceeded": sum(c["token_count"] > target for c in chunks),
        "model_limit_exceeded": sum(c["token_count"] > model_limit for c in chunks),
        "review_flagged_chunks": sum(bool(c["review_flags"]) for c in chunks),
        "source_coverage": "PASS", "source_substrings": "PASS", "context_substrings": "PASS",
        "token_count_rechecked": "PASS", "tokenizer_model": MODEL_ID,
        "tokenizer_revision": TOKENIZER_REVISION, "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_is_fast": tokenizer.is_fast,
        "transformers_version": importlib.metadata.version("transformers"),
        "tokenizers_version": importlib.metadata.version("tokenizers"),
        "rule_version": RULE_VERSION, "chunk_config_hash": config_hash, "embedding_generated": False, "database_written": False,
        "legal_applicability_verified": False,
        "cross_reference_resolution": "not_performed; candidates recorded for review",
        "notes": ["512는 목표. 문맥 보존 때문에 넘는 단위는 검수 대상으로 남김.", "embedding_text에는 제목·문맥이 포함될 수 있으나 text는 원문 부분문자열만.", "같은 조문 항 참조 문맥은 후보이며 인용된 다른 법의 항인지 검수 필요.", "유형 매핑표는 정답 라벨로 사용하지 않음.", "DB BIGINT ID는 적재 시 생성. document_key는 내부 문자열키."]
    }
    (outdir / "law_chunks.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks), encoding="utf-8")
    (outdir / "chunking_validation.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "chunk_stats_by_document.json").write_text(json.dumps(docstats, ensure_ascii=False, indent=2), encoding="utf-8")
    with (outdir / "chunk_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reviews[0]))
        writer.writeheader()
        writer.writerows(reviews)
    reading = ["법령 청킹 검수용 원문", json.dumps(audit, ensure_ascii=False, indent=2), ""]
    for c in chunks:
        reading += ["=" * 50, c["chunk_key"], str(c["token_count"]) + " tokens / flags: " + ", ".join(c["review_flags"]), "[원문]", c["text"], "[별도 문맥 원문]"]
        reading += [item["text"] for item in c["context_spans"]]
        reading.append("")
    (outdir / "chunk_review.txt").write_text("\n".join(reading), encoding="utf-8-sig")
    archive = outdir.parent / "chunking_output.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for name in ["law_chunks.jsonl", "chunking_validation.json", "chunk_stats_by_document.json", "chunk_review.csv", "chunk_review.txt"]:
            package.write(outdir / name, name)
    with zipfile.ZipFile(archive) as package:
        if package.testzip() is not None:
            raise ValueError("결과 ZIP 검증 실패")
    if len((outdir / "law_chunks.jsonl").read_text(encoding="utf-8").splitlines()) != len(chunks):
        raise ValueError("저장된 청크 수 불일치")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print("결과 ZIP:", archive.resolve())
    if audit["model_limit_exceeded"]:
        raise RuntimeError("모델 최대 길이를 초과한 청크가 있습니다. 임베딩 전에 검수하세요.")
    return audit, archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="output")
    parser.add_argument("--target", type=int, default=TARGET_TOKENS)
    args = parser.parse_args()
    run(args.input, args.output, args.target)
