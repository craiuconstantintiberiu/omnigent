"""Items and child-session routes."""

from __future__ import annotations

import asyncio
import hashlib

from fastapi import (
    APIRouter,
    Query,
    Request,
    status,
)
from fastapi.responses import Response

from omnigent.runtime.policies.approval import _ELICITATION_MODE
from omnigent.server._elicitation_registry import (
    _harness_elicitation_owners,
    _harness_elicitation_registry,
    _harness_parked_elicitations,
    _harness_pre_resolved_elicitations,
    _ParkedHarnessElicitation,
    _PreResolvedHarnessElicitation,
)
from omnigent.server.auth import (
    LEVEL_READ,
    AuthProvider,
)
from omnigent.server.routes._auth_helpers import (
    get_user_id as _get_user_id,
)
from omnigent.server.routes._auth_helpers import (
    require_access_and_level as _require_access_and_level,
)
from omnigent.server.routes._errors import session_not_found as _session_not_found
from omnigent.server.routes._sessions.common import *
from omnigent.server.routes._sessions.common import (
    get_server_runner_router,
    set_server_runner_router,
)
from omnigent.server.routes._sessions.helpers import *
from omnigent.server.routes._sessions.orchestration import *
from omnigent.server.schemas import (
    ChildSessionList,
    PaginatedList,
)
from omnigent.stores import AgentStore, ConversationStore
from omnigent.stores.permission_store import PermissionStore

# Cache-Control for the immutable (scroll-back) vs. live-tail item pages.
# Committed items never mutate, so a page fully below the tail is immutable
# and served with no revalidation; the tail may grow, so it is revalidated
# against the ETag. Both are ``private`` — item bodies are per-session data
# that a shared cache must never store.
_SESSION_ITEMS_IMMUTABLE_MAX_AGE_S = 86400
_SESSION_ITEMS_CACHE_IMMUTABLE = (
    f"private, max-age={_SESSION_ITEMS_IMMUTABLE_MAX_AGE_S}, immutable"
)
_SESSION_ITEMS_CACHE_TAIL = "private, no-cache"


def _session_items_cache_control(is_historical: bool) -> str:
    """
    Return the ``Cache-Control`` value for a session-items page.

    :param is_historical: ``True`` when the page is bounded strictly below
        the live tail (a scroll-back page that can never change), ``False``
        for the growing tail.
    :returns: The immutable policy for a historical page, else the
        revalidate-always tail policy.
    """
    return _SESSION_ITEMS_CACHE_IMMUTABLE if is_historical else _SESSION_ITEMS_CACHE_TAIL


def _session_items_etag(
    first_id: str | None,
    last_id: str | None,
    count: int,
    has_more: bool,
) -> str:
    """
    Build a validator ETag for a page of session items.

    Items are append-only, so a page's content is fully determined by its
    boundary ids, its length, and whether more rows follow — no field of an
    existing item can change without a new id appearing. The tag is weak
    (``W/``) because it validates semantic page equivalence, not a
    byte-exact body (gzip vs identity encodings share the same page).

    :param first_id: Id of the first item in the page, or ``None`` when empty.
    :param last_id: Id of the last item in the page, or ``None`` when empty.
    :param count: Number of items in the page.
    :param has_more: Whether more items exist beyond the page.
    :returns: A weak ETag string, e.g. ``'W/"a1b2c3d4"'``.
    """
    digest = hashlib.sha256(f"{first_id}:{last_id}:{count}:{has_more}".encode()).hexdigest()[:16]
    return f'W/"{digest}"'


def _if_none_match_matches(header_value: str | None, etag: str) -> bool:
    """
    Return whether an ``If-None-Match`` header matches *etag*.

    Handles a comma-separated list of tags and normalizes the weak
    prefix so a weak validator compares equal regardless of how the
    client echoes it. ``*`` matches any current representation.

    :param header_value: Raw ``If-None-Match`` request header, or ``None``.
    :param etag: The ETag computed for the current page.
    :returns: ``True`` when the client's cached copy is still current.
    """
    if not header_value:
        return False

    def _normalize(tag: str) -> str:
        return tag.strip().removeprefix("W/")

    if header_value.strip() == "*":
        return True
    wanted = _normalize(etag)
    return any(_normalize(candidate) == wanted for candidate in header_value.split(","))


def register_items_routes(
    router: APIRouter,
    *,
    conversation_store: ConversationStore,
    agent_store: AgentStore,
    auth_provider: AuthProvider | None = None,
    permission_store: PermissionStore | None = None,
) -> None:
    """Register the items routes on router."""

    @router.get(
        "/sessions/{session_id}/items",
        response_model=None,
        responses={200: {"model": PaginatedList}},
    )
    async def list_session_items(
        request: Request,
        response: Response,
        session_id: str,
        limit: int = Query(default=100, ge=1, le=1000),
        after: str | None = Query(default=None),
        before: str | None = Query(default=None),
        order: str = Query(default="asc", pattern="^(asc|desc)$"),
    ) -> PaginatedList | Response:
        """
        List items in a session with cursor-based pagination.

        Delegates to the conversation items store — session_id is
        the conversation_id. Same pagination contract as
        ``GET /v1/conversations/{id}/items``.

        Conversation items are append-only and never rewritten in place, so
        this endpoint is HTTP-cacheable. A page bounded strictly below the
        live tail (a scroll-back request — one carrying an ``after`` /
        ``before`` cursor with no more rows beyond it) can never change, so
        it is served ``Cache-Control: immutable`` — the browser re-serves it
        with no network round-trip at all. The live tail (the cursorless
        open fetch, or any page with ``has_more``) can grow when the agent
        commits a turn, so it is served ``no-cache`` with a validator
        ``ETag``: the browser revalidates via ``If-None-Match`` and gets an
        empty ``304`` when nothing changed, saving the body transfer.

        :param session_id: Session/conversation identifier,
            e.g. ``"conv_abc123"``.
        :param limit: Maximum number of items to return
            (1-1000, default 100).
        :param after: Cursor — return items after this item ID,
            e.g. ``"msg_abc123"``.
        :param before: Cursor — return items before this item ID.
        :param order: Sort order, ``"asc"`` (chronological,
            default) or ``"desc"``.
        :returns: A :class:`PaginatedList` of conversation items, or a bare
            ``304`` :class:`Response` when the client's ``If-None-Match``
            matches the current page.
        :raises OmnigentError: 404 if no session exists.
        """
        user_id = _get_user_id(request, auth_provider)
        access = await _require_access_and_level(
            user_id, session_id, LEVEL_READ, permission_store, conversation_store
        )
        if access.conversation is None:
            conv = await asyncio.to_thread(conversation_store.get_conversation, session_id)
            if conv is None:
                raise _session_not_found()
        page = await asyncio.to_thread(
            conversation_store.list_items,
            session_id,
            limit=limit,
            after=after,
            before=before,
            order=order,
        )
        # A cursor-bounded page with nothing older/newer beyond it sits
        # entirely below the live tail — since items never mutate, it is
        # immutable and safe to cache with no revalidation. Everything else
        # is (or touches) the growing tail: revalidate against the ETag.
        is_historical = (after is not None or before is not None) and not page.has_more
        etag = _session_items_etag(page.first_id, page.last_id, len(page.data), page.has_more)
        cache_control = _session_items_cache_control(is_historical)
        if _if_none_match_matches(request.headers.get("if-none-match"), etag):
            not_modified = Response(status_code=status.HTTP_304_NOT_MODIFIED)
            not_modified.headers["ETag"] = etag
            not_modified.headers["Cache-Control"] = cache_control
            return not_modified
        response.headers["ETag"] = etag
        response.headers["Cache-Control"] = cache_control
        data = [m.to_api_dict() for m in page.data]
        return PaginatedList(
            data=data,
            first_id=page.first_id,
            last_id=page.last_id,
            has_more=page.has_more,
        )

    # ── GET /sessions/{session_id}/child_sessions ────────────────

    @router.get(
        "/sessions/{session_id}/child_sessions",
        response_model=None,
        responses={200: {"model": ChildSessionList}},
    )
    async def list_child_sessions(
        request: Request,
        session_id: str,
        limit: int = Query(default=20, ge=1, le=1000),
        after: str | None = Query(default=None),
        before: str | None = Query(default=None),
        order: str = Query(default="desc", pattern="^(asc|desc)$"),
        tool: str | None = Query(default=None),
        session_name: str | None = Query(default=None),
    ) -> PaginatedList:
        """
        List sub-agent (child) sessions under a parent session.

        Returns a page of :class:`ChildSessionSummary` objects
        derived from child conversations (``kind="sub_agent"``,
        ``parent_conversation_id=session_id``) plus each child's
        latest task. Powers the web / REPL debug surfaces' "child
        sessions" panel without parsing parent
        ``function_call_output`` JSON handles. Pagination contract
        matches :func:`list_session_items` so existing client code
        can reuse the same cursor logic.

        :param request: Inbound HTTP request; carries the caller
            identity used to authorize READ on the parent session.
        :param session_id: Parent session/conversation identifier,
            e.g. ``"conv_abc123"``.
        :param limit: Maximum number of children to return
            (1-1000, default 20 — sub-agent fan-out is typically
            sparse compared to conversation items).
        :param after: Cursor — return children whose id appears
            after this one in sort order,
            e.g. ``"conv_child123"``.
        :param before: Cursor — return children before this one.
        :param order: Sort direction, ``"desc"`` (newest-first,
            default) or ``"asc"``. Sort column is ``created_at``.
        :param tool: When set, only return children whose title
            starts with this agent type (the segment before the
            ``":"``). Combined with ``session_name`` to form the
            exact title ``"{tool}:{session_name}"`` for server-side
            filtering.
        :param session_name: When set alongside ``tool``, only
            return children whose title matches
            ``"{tool}:{session_name}"`` exactly.
        :returns: A :class:`PaginatedList` of
            :class:`ChildSessionSummary` objects.
        :raises OmnigentError: 403 if the caller lacks READ on
            ``session_id``; 404 if no session exists there.
        """
        user_id = _get_user_id(request, auth_provider)
        # Require READ on the parent before listing its children (no cross-user enumeration).
        access = await _require_access_and_level(
            user_id, session_id, LEVEL_READ, permission_store, conversation_store
        )
        parent = access.conversation
        if parent is None:
            parent = await asyncio.to_thread(conversation_store.get_conversation, session_id)
        if parent is None:
            raise _session_not_found()
        title_filter: str | None = None
        if tool and session_name:
            title_filter = f"{tool}:{session_name}"
        page = await asyncio.to_thread(
            conversation_store.list_conversations,
            limit=limit,
            after=after,
            before=before,
            kind="sub_agent",
            parent_conversation_id=session_id,
            order=order,
            sort_by="created_at",
            title=title_filter,
        )
        data = await _child_session_summaries_from_conversations(
            page.data,
            session_id,
            conversation_store,
        )
        return PaginatedList(
            data=data,
            first_id=page.first_id,
            last_id=page.last_id,
            has_more=page.has_more,
        )
