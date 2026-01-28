"""Core unit tests for the nanocli agent loop.

These tests focus on bash parsing, shell execution, turn control flow, and
Responses API request shapes (via the mock server).
"""

from pathlib import Path

from openai import OpenAI

from nanocli._dev.mock_server import MockResponsesServer
from nanocli.core import BashOutput, BashRequest, Message, Session, parse_bash_block, run_bash_command, run_turn


def test_parse_bash_blocks_strict() -> None:
    """Extracts the first fenced bash block and ignores other languages."""
    text = "pre```bash\nls\n```mid```python\nprint('x')\n```post```sh\npwd\n```"
    assert parse_bash_block(text) == "ls"


def test_parse_bash_blocks_missing() -> None:
    """Returns None when no fenced bash block exists."""
    assert parse_bash_block("no code fences here") is None


def test_parse_bash_blocks_empty() -> None:
    """Treats empty bash blocks as absent."""
    assert parse_bash_block("```bash\n   \n```") is None


def test_parse_bash_blocks_first_match() -> None:
    """Uses the first bash block when multiple are present."""
    text = "```bash\nls\n```\ntext\n```bash\npwd\n```"
    assert parse_bash_block(text) == "ls"


def test_run_shell_command_empty(tmp_path: Path) -> None:
    """Returns a BashOutput object even for an empty command string."""
    output = run_bash_command("", tmp_path)
    assert isinstance(output, BashOutput)


def test_run_shell_command_error_exit(tmp_path: Path) -> None:
    """Non-zero exit status is propagated in BashOutput.returncode."""
    output = run_bash_command("false", tmp_path)
    assert output.returncode != 0


def test_run_shell_command_stderr(tmp_path: Path) -> None:
    """Captures stderr when a command fails."""
    output = run_bash_command("ls /definitely-missing-path", tmp_path)
    assert output.returncode != 0
    assert output.stderr


def test_run_shell_command_bad_cwd(tmp_path: Path) -> None:
    """Raises when the working directory does not exist."""
    bad_root = tmp_path / "missing"
    try:
        run_bash_command("echo hi", bad_root)
    except Exception:
        pass
    else:
        raise AssertionError("should raise when cwd doesn't exist")


def test_run_turn_executes_bash_blocks(tmp_path: Path) -> None:
    """Yields a BashRequest when the model returns a bash block."""
    responses = ["```bash\nls\n```", "done"]
    session = Session(
        model="gpt-test",
        instructions="",
        cwd=tmp_path,
        client=OpenAI(api_key="test", base_url="http://example.invalid"),
    )

    with MockResponsesServer(responses) as server:
        session.client = OpenAI(api_key="test", base_url=server.base_url, timeout=5.0)
        events = list(run_turn(session=session, user_text="run ls"))

    assert [type(event) for event in events] == [Message, BashRequest]
    assert server.requests[0]["store"] is False
    assert server.requests[0]["stream"] is True
    assert server.requests[0]["include"] == ["reasoning.encrypted_content"]


def test_run_turn_no_bash_blocks(tmp_path: Path) -> None:
    """Emits only a Message when no bash block is present."""
    responses = ["hello"]
    session = Session(
        model="gpt-test",
        instructions="",
        client=OpenAI(api_key="test", base_url="http://example.invalid"),
        cwd=tmp_path,
    )

    with MockResponsesServer(responses) as server:
        session.client = OpenAI(api_key="test", base_url=server.base_url, timeout=5.0)
        events = list(run_turn(session=session, user_text="hi"))

    assert [type(event) for event in events] == [Message]
    assert len(session.context) == 2


def test_run_turn_mock_text(tmp_path: Path) -> None:
    """Mock text trigger returns a single assistant message and no tool calls."""
    session = Session(
        model="gpt-test",
        instructions="",
        client=OpenAI(api_key="test", base_url="http://example.invalid"),
        cwd=tmp_path,
    )

    with MockResponsesServer() as server:
        session.client = OpenAI(api_key="test", base_url=server.base_url, timeout=5.0)
        events = list(run_turn(session=session, user_text="mock:text"))

    assert [type(event) for event in events] == [Message]
    first = events[0]
    assert isinstance(first, Message)
    assert first.content.strip() == "pong"
    assert len(server.requests) == 1


def test_run_turn_mock_single_pwd(tmp_path: Path) -> None:
    """Mock single trigger executes pwd and returns the completion text."""
    session = Session(
        model="gpt-test",
        instructions="",
        yolo=True,
        client=OpenAI(api_key="test", base_url="http://example.invalid"),
        cwd=tmp_path,
    )

    with MockResponsesServer() as server:
        session.client = OpenAI(api_key="test", base_url=server.base_url, timeout=5.0)
        events = list(run_turn(session=session, user_text="mock:single"))

    messages = [event for event in events if isinstance(event, Message)]
    outputs = [event for event in events if isinstance(event, BashOutput)]
    assert len(messages) == 2
    assert len(outputs) == 1
    assert outputs[0].returncode == 0
    assert outputs[0].stdout.strip() == str(tmp_path)
    assert messages[-1].content.strip() == "pwd complete."
    assert len(server.requests) == 2


def test_run_turn_mock_single_whoami(tmp_path: Path) -> None:
    """Mock single1 trigger executes whoami and returns the completion text."""
    session = Session(
        model="gpt-test",
        instructions="",
        yolo=True,
        client=OpenAI(api_key="test", base_url="http://example.invalid"),
        cwd=tmp_path,
    )

    with MockResponsesServer() as server:
        session.client = OpenAI(api_key="test", base_url=server.base_url, timeout=5.0)
        events = list(run_turn(session=session, user_text="mock:single1"))

    messages = [event for event in events if isinstance(event, Message)]
    outputs = [event for event in events if isinstance(event, BashOutput)]
    assert len(messages) == 2
    assert len(outputs) == 1
    assert outputs[0].returncode == 0
    assert outputs[0].stdout.strip()
    assert messages[-1].content.strip() == "whoami complete."
    assert len(server.requests) == 2


def test_run_turn_denies_bash_request(tmp_path: Path) -> None:
    """Stops the loop when approval is denied."""
    responses = ["```bash\npwd\n```"]
    session = Session(
        model="gpt-test",
        instructions="",
        client=OpenAI(api_key="test", base_url="http://example.invalid"),
        cwd=tmp_path,
    )

    with MockResponsesServer(responses) as server:
        session.client = OpenAI(api_key="test", base_url=server.base_url, timeout=5.0)
        gen = run_turn(session=session, user_text="run pwd")
        first = next(gen)
        assert isinstance(first, Message)
        second = next(gen)
        assert isinstance(second, BashRequest)
        try:
            gen.send(False)
        except StopIteration:
            pass

    assert not any(isinstance(item, BashOutput) for item in session.context)


def test_run_turn_yolo_executes_command(tmp_path: Path) -> None:
    """Runs commands without approval when yolo is enabled."""
    responses = ["```bash\npwd\n```", "done"]
    session = Session(
        model="gpt-test",
        instructions="",
        yolo=True,
        client=OpenAI(api_key="test", base_url="http://example.invalid"),
        cwd=tmp_path,
    )

    with MockResponsesServer(responses) as server:
        session.client = OpenAI(api_key="test", base_url=server.base_url, timeout=5.0)
        events = list(run_turn(session=session, user_text="run pwd"))

    assert [type(event) for event in events] == [Message, BashOutput, Message]
    assert len(server.requests) == 2


def test_run_turn_multiturn_with_mock_multi(tmp_path: Path) -> None:
    """Validates multi-turn semantics: correct commands, outputs, and final text."""
    marker = tmp_path / "marker.txt"
    marker.write_text("hello", encoding="utf-8")
    session = Session(
        model="gpt-test",
        instructions="",
        yolo=True,
        client=OpenAI(api_key="test", base_url="http://example.invalid"),
        cwd=tmp_path,
    )

    with MockResponsesServer() as server:
        session.client = OpenAI(api_key="test", base_url=server.base_url, timeout=5.0)
        events = list(run_turn(session=session, user_text="mock:multi"))

    messages = [event for event in events if isinstance(event, Message)]
    outputs = [event for event in events if isinstance(event, BashOutput)]
    assert len(messages) == 3
    assert len(outputs) == 2
    assert outputs[0].returncode == 0
    assert outputs[1].returncode == 0
    assert outputs[0].stdout.strip() == str(tmp_path)
    assert marker.name in outputs[1].stdout
    assert messages[-1].content.strip() == "inspect complete."
    assert len(server.requests) == 3
    assert any(isinstance(item, BashOutput) for item in session.context)


def test_instructions_as_system_prompt(tmp_path: Path) -> None:
    """Places instructions as the first system input when enabled."""
    responses = ["ok"]
    session = Session(
        model="gpt-test",
        instructions="system instructions",
        instructions_as_system=True,
        client=OpenAI(api_key="test", base_url="http://example.invalid"),
        cwd=tmp_path,
    )

    with MockResponsesServer(responses) as server:
        session.client = OpenAI(api_key="test", base_url=server.base_url, timeout=5.0)
        list(run_turn(session=session, user_text="hi"))

    first_item = server.requests[0]["input"][0]
    assert first_item["role"] == "system"
    assert first_item["content"] == "system instructions"
