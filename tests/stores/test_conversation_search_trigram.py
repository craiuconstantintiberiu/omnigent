from __future__ import annotations

import pytest
from sqlalchemy import event, text

from omnigent.db.utils import (
    _TRIGRAM_FTS_TABLE,
    ensure_fts_table,
    supports_trigram_fts,
)
from omnigent.entities import MessageData, NewConversationItem
from omnigent.stores.conversation_store.sqlalchemy_store import (
    SqlAlchemyConversationStore,
)


def _require_trigram(store: SqlAlchemyConversationStore) -> None:
    if not supports_trigram_fts(store._conv_engine):
        pytest.skip("SQLite trigram tokenizer is unavailable")


def _append(store: SqlAlchemyConversationStore, conversation_id: str, *messages: str) -> None:
    store.append(
        conversation_id,
        [
            NewConversationItem(
                type="message",
                response_id=f"resp_{index}",
                data=MessageData(
                    role="user",
                    content=[{"type": "input_text", "text": message}],
                ),
            )
            for index, message in enumerate(messages)
        ],
    )


def _search_ids(store: SqlAlchemyConversationStore, query: str) -> set[str]:
    return {conversation.id for conversation in store.list_conversations(search_query=query).data}


def test_trigram_search_preserves_substring_matching(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    _require_trigram(conversation_store)
    match = conversation_store.create_conversation(title="general")
    other = conversation_store.create_conversation(title="other")
    _append(conversation_store, match.id, 'An OutOfMemoryError reported code FOO-123 and "halted"')
    _append(conversation_store, other.id, "unrelated text")

    statements: list[str] = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(conversation_store._conv_engine, "before_cursor_execute", capture)
    try:
        assert _search_ids(conversation_store, "outofmemory") == {match.id}
    finally:
        event.remove(conversation_store._conv_engine, "before_cursor_execute", capture)

    assert any(
        f"FROM {_TRIGRAM_FTS_TABLE} WHERE workspace_id" in statement for statement in statements
    )
    assert _search_ids(conversation_store, "OUTOFMEMORY") == {match.id}
    assert _search_ids(conversation_store, 'FOO-123 and "halted"') == {match.id}


def test_short_search_uses_substring_fallback(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    _require_trigram(conversation_store)
    match = conversation_store.create_conversation(title="general")
    _append(conversation_store, match.id, "OOM while starting")

    assert _search_ids(conversation_store, "OO") == {match.id}


async def test_bulk_writes_and_deletes_update_trigram_index(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    _require_trigram(conversation_store)
    removed = conversation_store.create_conversation()
    removed_child = conversation_store.create_conversation(
        kind="sub_agent",
        title="child",
        parent_conversation_id=removed.id,
    )
    kept = conversation_store.create_conversation()
    _append(conversation_store, removed.id, "bulkneedle one", "bulkneedle two")
    _append(conversation_store, removed_child.id, "bulkneedle child")
    _append(conversation_store, kept.id, "bulkneedle three")

    with conversation_store._conv_session("test_trigram_index") as session:
        rows = session.execute(
            text(
                f"SELECT item_id FROM {_TRIGRAM_FTS_TABLE} "
                "WHERE workspace_id = 0 AND search_text MATCH :query"
            ),
            {"query": '"bulkneedle"'},
        ).all()
    assert len(rows) == 4
    assert all(isinstance(row.item_id, bytes) for row in rows)

    assert await conversation_store.delete_conversation(removed.id)

    with conversation_store._conv_session("test_trigram_index") as session:
        rows = session.execute(
            text(
                f"SELECT conversation_id FROM {_TRIGRAM_FTS_TABLE} "
                "WHERE workspace_id = 0 AND search_text MATCH :query"
            ),
            {"query": '"bulkneedle"'},
        ).all()
    assert rows == [(bytes.fromhex(kept.id),)]


def test_existing_items_are_backfilled(
    conversation_store: SqlAlchemyConversationStore,
) -> None:
    _require_trigram(conversation_store)
    conversation = conversation_store.create_conversation()
    _append(conversation_store, conversation.id, "backfillneedle")

    with conversation_store._conv_engine.begin() as connection:
        connection.execute(text(f"DROP TABLE {_TRIGRAM_FTS_TABLE}"))
    ensure_fts_table(conversation_store._conv_engine)

    assert _search_ids(conversation_store, "backfillneedle") == {conversation.id}
