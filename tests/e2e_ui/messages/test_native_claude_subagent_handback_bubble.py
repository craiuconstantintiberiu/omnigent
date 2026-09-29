"""UI journey: a Claude Code subagent hand-back must not render as a user prompt.

Claude Code can deliver a background subagent's final report to its parent
through the ``SubagentHandback`` contract. It queues the report for the parent
as a prompt marked ``isMeta``, wrapped in ``<agent-message from="…">`` with a
``[Subagent hand-back]`` header. The person never typed that text, so the
Omnigent chat view must not show it as one of their own messages.

The contract is gated on Claude Code's auto permission mode and the remote
``tengu_lively_waffle`` feature flag. This build of Claude Code exposes no env
override for the flag, so the fixture stands it in through the GrowthBook cache
in Claude Code's global config file and lets that cache be read while
non-essential traffic is disabled. The mock model scripts every request the
journey makes: the auto-mode classifier approves, the parent spawns a
``general-purpose`` subagent, and the subagent hands back its report.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from playwright.sync_api import Page, expect

from tests.e2e_ui.conftest import (
    _REPO_ROOT,
    _create_native_claude_session,
    _ensure_runner_online,
    _prepared_repro_environment,
    _server_state,
    _temp_omnigent_mock_config,
    configure_mock_llm,
    mock_llm_saw_user_text,
    reset_mock_llm,
    set_fallback_mock_llm,
)

from .test_message_render_parity import (
    _ASSISTANT,
    _USER,
    _WORKING,
    _ensure_chat_view,
    _item_text,
    _ordered_message_items,
    _send,
)
from .test_native_claude_render_parity import _open_terminal_view, _wait_terminal_connected

# Auto mode refuses older models ("does not support auto mode"); the mock
# serves any model name, so pick one Claude Code accepts.
_AUTO_MODE_MODEL = "claude-sonnet-4-6"
_LAUNCH_ARGS = ["--permission-mode", "auto", "--model", _AUTO_MODE_MODEL]

# The auto-mode classifier's two stages quote these instructions in their
# user turn; ``<block>no</block>`` approves the action under review.
_CLASSIFIER_STAGE1 = "Your ENTIRE response MUST begin with <block>"
_CLASSIFIER_STAGE2 = "Use <thinking> before responding with <block>"
_CLASSIFIER_ALLOW = "<block>no</block>"
# Session-title generation quotes the person's prompt too; keep it off the
# parent's queue.
_TITLE_PROMPT = "Write the title in the predominant language"

_HANDBACK_HEADER = "[Subagent hand-back]"

# Claude Code boot + terminal attach + eight scripted model round-trips.
_TERMINAL_READY_TIMEOUT_MS = 120_000
_JOURNEY_TIMEOUT_MS = 150_000


def _claude_config_files() -> tuple[Path, Path]:
    """Return the global config and user settings files the runner's ``claude`` reads.

    Mirrors Claude Code's own resolution: ``.claude.json`` and ``settings.json``
    inside ``CLAUDE_CONFIG_DIR`` when it is set, otherwise ``~/.claude.json``
    and ``~/.claude/settings.json``. The prepared reproduction environment
    isolates that directory next to its config home; a fixture-owned runner
    inherits this process's environment.
    """
    if _prepared_repro_environment()["OMNIGENT_REPRO_SERVER_URL"]:
        state_path = _REPO_ROOT / ".omnigent" / "repro-env" / "environment.json"
        state = json.loads(state_path.read_text())
        config_dir: Path | None = Path(state["config_home"]).parent / "claude-config"
    else:
        configured = os.environ.get("CLAUDE_CONFIG_DIR")
        config_dir = Path(configured).expanduser() if configured else None
    if config_dir is not None:
        return config_dir / ".claude.json", config_dir / "settings.json"
    return Path.home() / ".claude.json", Path.home() / ".claude" / "settings.json"


@contextlib.contextmanager
def _merged_json(path: Path, patch: dict[str, Any]) -> Iterator[None]:
    original = path.read_text() if path.exists() else None
    data: dict[str, Any] = json.loads(original) if original else {}
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    try:
        yield
    finally:
        if original is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(original)


@contextlib.contextmanager
def _handback_contract_enabled(global_config: Path, user_settings: Path) -> Iterator[None]:
    """Turn on Claude Code's subagent hand-back contract for the launches that follow."""
    flag_cache = _merged_json(
        global_config,
        {"cachedGrowthBookFeatures": {"tengu_lively_waffle": True}},
    )
    # The runner disables non-essential traffic, which also switches the flag
    # cache off unless this opt-in is set.
    cache_opt_in = _merged_json(
        user_settings,
        {"env": {"CLAUDE_CODE_GB_DISK_CACHE_WHEN_TELEMETRY_OFF": "1"}},
    )
    with flag_cache, cache_opt_in:
        yield


@pytest.fixture
def native_claude_handback_session(
    live_server: str,
    mock_llm_server_url: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[str, str]]:
    """A mock-backed Claude Code session launched in auto mode with hand-back enabled."""
    respawned = _ensure_runner_online(live_server, tmp_path_factory)
    runner_id = str(_server_state["runner_id"])
    use_mock = not _server_state.get("workflow_owned")
    provider: Any = (
        _temp_omnigent_mock_config(mock_llm_server_url, "claude")
        if use_mock
        else contextlib.nullcontext()
    )
    with provider, _handback_contract_enabled(*_claude_config_files()):
        session_id = _create_native_claude_session(
            live_server, runner_id, terminal_launch_args=_LAUNCH_ARGS
        )
        try:
            yield (live_server, session_id)
        finally:
            httpx.delete(f"{live_server}/v1/sessions/{session_id}", timeout=10.0)
            if respawned is not None:
                respawned.terminate()
                try:
                    respawned.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    respawned.kill()
                    respawned.wait(timeout=5)


def _script_handback_journey(mock_url: str, nonce: str) -> list[str]:
    """Queue every model reply the journey needs; return the parent's reply markers in order."""
    reset_mock_llm(mock_url)
    parent_replies = [f"ast-{nonce}-{index}" for index in range(4)]
    for stage, match in (("s1", _CLASSIFIER_STAGE1), ("s2", _CLASSIFIER_STAGE2)):
        configure_mock_llm(
            mock_url,
            [{"text": _CLASSIFIER_ALLOW}] * 40,
            key=f"classifier-{stage}-{nonce}",
            match=match,
        )
    configure_mock_llm(
        mock_url, [{"text": "Hand-back journey"}] * 5, key=f"title-{nonce}", match=_TITLE_PROMPT
    )
    configure_mock_llm(
        mock_url,
        [
            {
                "tool_calls": [
                    {
                        "name": "Agent",
                        "arguments": json.dumps(
                            {
                                "description": "summarise the request",
                                "prompt": f"worker-{nonce} summarise the request",
                                "subagent_type": "general-purpose",
                            }
                        ),
                    }
                ]
            },
            *({"text": token} for token in parent_replies),
        ],
        key=f"parent-{nonce}",
        match=f"delegate-{nonce}",
    )
    configure_mock_llm(
        mock_url,
        [
            {
                "tool_calls": [
                    {
                        "name": "SubagentHandback",
                        "arguments": json.dumps(
                            {"message": f"REPORT-{nonce}: the delegated summary is complete."}
                        ),
                    }
                ]
            },
            *({"text": f"worker-done-{nonce}"} for _ in range(3)),
        ],
        key=f"child-{nonce}",
        match=f"worker-{nonce}",
    )
    set_fallback_mock_llm(mock_url, _AUTO_MODE_MODEL, "fallback-reply")
    return parent_replies


@pytest.mark.nightly
@pytest.mark.timeout(420)
def test_subagent_handback_is_not_a_user_prompt(
    page: Page,
    native_claude_handback_session: tuple[str, str],
    mock_llm_server_url: str,
) -> None:
    """A subagent's ``<agent-message>`` hand-back never renders as a person's message."""
    base_url, session_id = native_claude_handback_session
    nonce = uuid.uuid4().hex[:8]
    parent_replies = _script_handback_journey(mock_llm_server_url, nonce)

    page.goto(f"{base_url}/c/{session_id}")
    _open_terminal_view(page)
    _wait_terminal_connected(page)
    _ensure_chat_view(page)

    prompt = f"delegate-{nonce} please delegate a summary of this request to a subagent"
    _send(page, prompt)
    expect(page.locator(_USER, has_text=f"delegate-{nonce}")).to_have_count(1, timeout=30_000)
    # The parent answers three times: after launching the subagent, after the
    # hand-back arrives, and after the background-task notification.
    expect(page.locator(_ASSISTANT, has_text=parent_replies[2]).first).to_be_visible(
        timeout=_JOURNEY_TIMEOUT_MS
    )
    expect(page.locator(_WORKING)).to_have_count(0, timeout=_TERMINAL_READY_TIMEOUT_MS)
    # Only a report the parent model actually received can be misattributed;
    # the transcript is deliberately silent about it, so ask the mock.
    assert mock_llm_saw_user_text(mock_llm_server_url, _HANDBACK_HEADER, f"REPORT-{nonce}"), (
        "the subagent's report never reached the parent model as a hand-back"
    )

    expect(page.locator(_USER, has_text=_HANDBACK_HEADER)).to_have_count(0)
    expect(page.locator(_USER)).to_have_count(1)
    user_texts = [
        _item_text(item)
        for item in _ordered_message_items(base_url, session_id)
        if item.get("role") == "user"
    ]
    offending = [text for text in user_texts if _HANDBACK_HEADER in text]
    assert not offending, f"subagent hand-back stored as user message(s): {offending}"
