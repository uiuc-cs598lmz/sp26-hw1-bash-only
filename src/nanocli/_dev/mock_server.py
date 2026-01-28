import argparse
import hashlib
import http.client
import json
import re
import socket
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request, stream_with_context
from flask.typing import ResponseReturnValue

ResponsePayload = str | dict[str, Any]

TRIGGER_TEXT_PING = "mock:text"
TRIGGER_BASH_SINGLE_PWD = "mock:single"
TRIGGER_BASH_SINGLE_WHOAMI = "mock:single1"
TRIGGER_BASH_MULTI_INSPECT = "mock:multi"

PATTERN_RESPONSES: dict[str, list[str]] = {
    TRIGGER_TEXT_PING: ["pong"],
    TRIGGER_BASH_SINGLE_PWD: [
        "I'll check the workding directory:\n```bash\npwd\n```",
        "pwd complete.",
    ],
    TRIGGER_BASH_SINGLE_WHOAMI: [
        "Let's run `whoami`:\n```bash\nwhoami\n```",
        "whoami complete.",
    ],
    TRIGGER_BASH_MULTI_INSPECT: [
        "Let's check the current directory:\n```bash\npwd\n```",
        "I will run `ls` to confirm:\n```bash\nls\n```",
        "inspect complete.",
    ],
}

SHELL_OBS_RE = re.compile(r"^Command:\n```bash\n.*?\n```\nOutput:\n", re.DOTALL)


class _ServerState:
    def __init__(
        self,
        responses: Iterable[ResponsePayload],
        default_response: ResponsePayload | None = None,
    ) -> None:
        self.responses: deque[ResponsePayload] = deque(responses)
        self.default_response = default_response
        self.requests: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def record_request(self, payload: dict[str, Any]) -> None:
        with self.lock:
            self.requests.append(payload)

    def next_response(self, request_payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        pattern = _pattern_response(request_payload)
        if pattern is not None:
            return 200, pattern
        with self.lock:
            if self.responses:
                response = self.responses.popleft()
            elif self.default_response is not None:
                response = self.default_response
            else:
                response = _fallback_response_text(request_payload)
        if isinstance(response, dict):
            return 200, response
        return 200, _build_response(request_payload, content=str(response))


def _build_output_message(content: str, *, item_id: str | None = None, status: str = "completed") -> dict[str, Any]:
    return {
        "id": item_id or f"msg-mock-{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "status": status,
        "content": [
            {
                "type": "output_text",
                "text": content,
                "annotations": [],
            }
        ],
    }


def _build_response(
    request_payload: dict[str, Any],
    *,
    content: str | None,
    response_id: str | None = None,
    created_at: float | None = None,
    status: str = "completed",
) -> dict[str, Any]:
    created = time.time() if created_at is None else created_at
    model = request_payload.get("model", "mock-model")
    output: list[dict[str, Any]] = []
    if content is not None:
        output.append(_build_output_message(content, status="completed"))
    response: dict[str, Any] = {
        "id": response_id or f"resp-mock-{uuid.uuid4().hex}",
        "object": "response",
        "created_at": created,
        "model": model,
        "output": output,
        "parallel_tool_calls": False,
        "tool_choice": "none",
        "tools": [],
        "status": status,
        "prompt_cache_key": request_payload.get("prompt_cache_key"),
        "prompt_cache_retention": request_payload.get("prompt_cache_retention"),
    }
    if status == "completed":
        response["completed_at"] = created
    return response


def _input_items_from_payload(request_payload: dict[str, Any]) -> list[object]:
    raw_input = request_payload.get("input")
    if isinstance(raw_input, str):
        return [{"role": "user", "content": raw_input}]
    if isinstance(raw_input, list):
        return raw_input
    messages = request_payload.get("messages")
    if isinstance(messages, list):
        return messages
    return []


def _fallback_response_text(request_payload: dict[str, Any]) -> str:
    messages = _input_items_from_payload(request_payload)
    text = _last_non_shell_user_text(messages)
    if text:
        return _chatty_response(text)
    return "Mock response."


def _extract_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") not in {"text", "input_text", "output_text", "reasoning_text"}:
                continue
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return ""


def _is_shell_observation(text: str) -> bool:
    if SHELL_OBS_RE.match(text):
        return True
    return "<stdout>" in text and "<returncode>" in text


def _last_non_shell_user_text(messages: list[object]) -> str:
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            continue
        text = _extract_text(message.get("content")).strip()
        if text and not _is_shell_observation(text):
            return text
    return ""


def _find_last_trigger(messages: list[object]) -> tuple[int, str] | None:
    trigger_set = set(PATTERN_RESPONSES.keys())
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            continue
        text = _extract_text(message.get("content")).strip()
        if not text or _is_shell_observation(text):
            continue
        if text in trigger_set:
            return index, text
        return None
    return None


def _count_assistant_after(messages: list[object], start_index: int) -> int:
    count = 0
    for message in messages[start_index + 1 :]:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            count += 1
    return count


def _chatty_response(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return "Mock response."
    lower = cleaned.lower()

    if lower.startswith("mock:"):
        known = ", ".join(sorted(PATTERN_RESPONSES.keys()))
        return f"Unknown mock trigger. Try one of: {known}."

    if _matches_any(lower, ("hi", "hello", "hey", "yo", "sup")):
        return _pick_variant(
            cleaned,
            [
                "Hey! I am a mock server. Ask me something.",
                "Hello! I am a mock server and ready to chat.",
                "Hi! I am a mock server. What should we try?",
            ],
        )

    if _contains_any(lower, ("thanks", "thank you", "thx")):
        return _pick_variant(cleaned, ["You are welcome.", "Anytime.", "Glad to help."])

    if _contains_any(lower, ("bye", "goodbye", "see ya", "later")):
        return _pick_variant(cleaned, ["Later!", "Bye for now.", "See you soon."])

    if lower in {"yes", "yeah", "yep", "sure", "ok", "okay"}:
        return _pick_variant(cleaned, ["Got it.", "Sounds good.", "Okay."])

    if lower in {"no", "nope", "nah"}:
        return _pick_variant(cleaned, ["Understood.", "Okay, no problem.", "All right."])

    if _contains_any(lower, ("help", "what can you do", "commands")):
        return (
            "I am a mock chat server. I can answer simple text and emit bash blocks "
            "when you use the mock:bash:* triggers."
        )

    if _contains_any(lower, ("error", "bug", "failed", "traceback")):
        return "That sounds annoying. If you paste the exact error text, I can respond with a mock diagnosis."

    if _looks_like_question(lower):
        topic = _truncate_text(cleaned, 160).rstrip("?")
        return (
            "Good question. I do not have real data here, but a quick approach is to "
            f"start small and verify assumptions: {topic}."
        )

    if len(cleaned.split()) <= 3:
        return f"Noted: {_truncate_text(cleaned, 120)}."

    return f"Got it. You said: {_truncate_text(cleaned, 160)}."


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _pick_variant(key: str, options: list[str]) -> str:
    if not options:
        return ""
    digest = hashlib.sha256(key.encode("utf-8")).digest()[0]
    return options[digest % len(options)]


def _matches_any(text: str, prefixes: tuple[str, ...]) -> bool:
    for prefix in prefixes:
        if text == prefix or text.startswith(prefix + " "):
            return True
    return False


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _looks_like_question(text: str) -> bool:
    if "?" in text:
        return True
    words = text.split()
    if not words:
        return False
    return words[0] in {
        "what",
        "why",
        "how",
        "when",
        "where",
        "who",
        "which",
        "can",
        "could",
        "should",
        "do",
        "does",
        "is",
        "are",
        "am",
        "will",
        "would",
    }


def _pattern_response(request_payload: dict[str, Any]) -> dict[str, Any] | None:
    messages = _input_items_from_payload(request_payload)
    if not messages:
        return None
    match = _find_last_trigger(messages)
    if match is None:
        return None
    trigger_index, trigger = match
    responses = PATTERN_RESPONSES.get(trigger)
    if not responses:
        return None
    step = _count_assistant_after(messages, trigger_index)
    if step >= len(responses):
        content = responses[-1]
    else:
        content = responses[step]
    return _build_response(request_payload, content=content)


def _response_output_text(response_payload: dict[str, Any]) -> str:
    output = response_payload.get("output")
    if not isinstance(output, list):
        return ""
    texts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") != "output_text":
                continue
            text = part.get("text")
            if isinstance(text, str):
                texts.append(text)
    return "".join(texts)


def _sse_event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def _stream_response_events(request_payload: dict[str, Any], content: str) -> Iterator[str]:
    response_id = f"resp-mock-{uuid.uuid4().hex}"
    message_id = f"msg-mock-{uuid.uuid4().hex}"
    created_at = time.time()
    created_response = _build_response(
        request_payload,
        content=None,
        response_id=response_id,
        created_at=created_at,
        status="in_progress",
    )
    created_response["output"] = []
    created_response.pop("completed_at", None)

    yield _sse_event(
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": created_response,
        }
    )

    output_item = {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "status": "in_progress",
        "content": [],
    }
    yield _sse_event(
        {
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 0,
            "item": output_item,
        }
    )

    content_part = {"type": "output_text", "text": "", "annotations": []}
    yield _sse_event(
        {
            "type": "response.content_part.added",
            "sequence_number": 2,
            "output_index": 0,
            "item_id": message_id,
            "content_index": 0,
            "part": content_part,
        }
    )

    yield _sse_event(
        {
            "type": "response.output_text.delta",
            "sequence_number": 3,
            "output_index": 0,
            "content_index": 0,
            "item_id": message_id,
            "delta": content,
            "logprobs": [],
        }
    )

    yield _sse_event(
        {
            "type": "response.output_text.done",
            "sequence_number": 4,
            "output_index": 0,
            "content_index": 0,
            "item_id": message_id,
            "text": content,
            "logprobs": [],
        }
    )

    completed_response = _build_response(
        request_payload,
        content=content,
        response_id=response_id,
        created_at=created_at,
        status="completed",
    )
    completed_response["output"] = [_build_output_message(content, item_id=message_id, status="completed")]

    yield _sse_event(
        {
            "type": "response.completed",
            "sequence_number": 5,
            "response": completed_response,
        }
    )
    yield "data: [DONE]\n\n"


def _create_app(state: _ServerState) -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health() -> ResponseReturnValue:
        return jsonify({"status": "ok"}), 200

    @app.post("/v1/responses")
    def responses() -> ResponseReturnValue:
        payload = request.get_json(silent=True)
        if payload is None:
            if request.data:
                return (
                    jsonify(
                        {
                            "error": {
                                "message": "Invalid JSON",
                                "type": "invalid_request_error",
                                "code": "invalid_json",
                            }
                        }
                    ),
                    400,
                )
            payload = {}
        state.record_request(payload)
        status, response = state.next_response(payload)
        if payload.get("stream"):
            content = _response_output_text(response)
            return Response(
                stream_with_context(_stream_response_events(payload, content)),
                status=status,
                mimetype="text/event-stream",
            )
        return jsonify(response), status

    @app.post("/__shutdown__")
    def shutdown() -> ResponseReturnValue:
        shutdown_fn = request.environ.get("werkzeug.server.shutdown")
        if shutdown_fn is None:
            return (
                jsonify(
                    {
                        "error": {
                            "message": "Shutdown not supported.",
                            "type": "mock_error",
                            "code": "mock_shutdown_unavailable",
                        }
                    }
                ),
                500,
            )
        shutdown_fn()
        return jsonify({"status": "shutting down"}), 200

    return app


class MockResponsesServer:
    def __init__(
        self,
        responses: Iterable[ResponsePayload] | None = None,
        *,
        default_response: ResponsePayload | None = None,
    ) -> None:
        self._state = _ServerState(responses or [], default_response=default_response)
        self._app = _create_app(self._state)
        self._thread: threading.Thread | None = None
        self.base_url: str | None = None
        self._host: str | None = None
        self._port: int | None = None

    @property
    def requests(self) -> list[dict[str, Any]]:
        return self._state.requests

    @property
    def app(self) -> Flask:
        return self._app

    def push(self, response: ResponsePayload) -> None:
        with self._state.lock:
            self._state.responses.append(response)

    def _run_app(self, host: str, port: int) -> None:
        self._app.run(host=host, port=port, threaded=True, use_reloader=False)

    def start(self, host: str = "127.0.0.1", port: int = 0) -> None:
        if self._thread is not None:
            return
        if port == 0:
            port = _find_free_port(host)
        self._host = host
        self._port = port
        self.base_url = f"http://{host}:{port}/v1"
        self._thread = threading.Thread(
            target=self._run_app,
            args=(host, port),
            name="mock-responses-server",
            daemon=True,
        )
        self._thread.start()
        if not _wait_for_server(host, port, timeout=1.0):
            raise RuntimeError("Mock server failed to start.")

    def run(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        self._app.run(host=host, port=port, threaded=True, use_reloader=False)

    def stop(self) -> None:
        if self._thread is None:
            return
        self._request_shutdown()
        self._thread.join(timeout=1)
        self._thread = None
        self._host = None
        self._port = None

    def _request_shutdown(self) -> None:
        if self._host is None or self._port is None:
            return
        try:
            conn = http.client.HTTPConnection(self._host, self._port, timeout=0.5)
            conn.request("POST", "/__shutdown__")
            resp = conn.getresponse()
            resp.read()
            conn.close()
        except OSError:
            return

    def __enter__(self) -> "MockResponsesServer":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop()


MockChatCompletionServer = MockResponsesServer


def _load_responses(path: Path) -> list[ResponsePayload]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return [data]


def _find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def _wait_for_server(host: str, port: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            conn = http.client.HTTPConnection(host, port, timeout=0.2)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            resp.read()
            conn.close()
            if resp.status == 200:
                return True
        except OSError:
            time.sleep(0.02)
    return False


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="nanocli-mock")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    parser.add_argument(
        "--response",
        action="append",
        default=[],
        help="Assistant response content. Can be repeated.",
    )
    parser.add_argument(
        "--responses-file",
        type=Path,
        default=None,
        help="Path to a JSON list of response items.",
    )
    parser.add_argument(
        "--default-response",
        default=None,
        help="Fallback assistant response when the queue is empty.",
    )
    args = parser.parse_args(argv)

    responses: list[ResponsePayload] = list(args.response)
    if args.responses_file is not None:
        if not args.responses_file.exists():
            parser.error(f"Responses file not found: {args.responses_file}")
        responses.extend(_load_responses(args.responses_file))

    default_response: ResponsePayload | None = args.default_response

    server = MockResponsesServer(
        responses,
        default_response=default_response,
    )
    print(f"Mock server listening on http://{args.host}:{args.port}/v1")
    server.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
