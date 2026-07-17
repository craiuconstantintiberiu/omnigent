"""add projects table and conversation_metadata.project_id

Revision ID: b1c2d3e4f5a6
Revises: d7f1a2b3c4e5
Create Date: 2026-07-16 00:00:00.000000

Promotes "projects" from the implicit ``omni_project`` conversation label to a
first-class entity (see ``designs/PROJECTS_PRD.md``). Adds:

- the ``projects`` table — a user-defined, owner-private container that groups
  sessions and exists independently of its members (so it can be empty); and
- ``omnigent_conversation_metadata.project_id`` — nullable session→project
  membership (``NULL`` = unfiled), plus an index backing "list sessions in
  project X" and per-project counts.

Both are additive. There are no foreign-key constraints (schema Rule R032): the
``project_id`` relationship is enforced by the application, not the database.
The new column is nullable, so existing sessions stay unfiled — no backfill
runs here. Migrating existing ``omni_project`` labels onto ``project_id`` is a
separate, later step.

Adding the column uses batch mode (``op.batch_alter_table``) because this repo
runs Alembic with ``render_as_batch=True`` so SQLite — which cannot
``ALTER TABLE ... ADD COLUMN`` with every option in place — rebuilds via the
copy-and-swap batch path.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from omnigent.db.db_models import Uuid16

revision: str = "b1c2d3e4f5a6"
down_revision: str | None = "d7f1a2b3c4e5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create ``projects`` and add ``project_id`` to conversation metadata."""
    op.create_table(
        "projects",
        sa.Column("workspace_id", sa.BigInteger(), nullable=False, server_default="0"),
        # UUID PK stored as 16 raw bytes (Uuid16), read back as bare hex.
        sa.Column("id", Uuid16(), nullable=False),
        sa.Column("name", sa.String(256), nullable=False),
        # Owner stamped on the row (projects have no ACL, Rule R032 / PRD §9).
        # NULL in single-user / OSS mode.
        sa.Column("owner_user_id", sa.String(128), nullable=True),
        sa.Column("created_at", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("workspace_id", "id"),
    )
    op.create_index(
        "ix_projects_owner_user_id",
        "projects",
        ["workspace_id", "owner_user_id", "id"],
        unique=False,
    )
    # Not unique: NULL owners (single-user mode) would otherwise collide on
    # name; per-owner name uniqueness is enforced in the store.
    op.create_index(
        "ix_projects_name",
        "projects",
        ["workspace_id", "owner_user_id", "name", "id"],
        unique=False,
    )

    with op.batch_alter_table("omnigent_conversation_metadata") as batch_op:
        # Relates to projects.id; no DB FK (Rule R032). NULL = unfiled.
        batch_op.add_column(sa.Column("project_id", Uuid16(), nullable=True))
        batch_op.create_index(
            "ix_conversation_metadata_project_id",
            ["workspace_id", "project_id", "id"],
            unique=False,
        )


def downgrade() -> None:
    """Drop ``project_id`` from conversation metadata and the ``projects`` table."""
    with op.batch_alter_table("omnigent_conversation_metadata") as batch_op:
        batch_op.drop_index("ix_conversation_metadata_project_id")
        batch_op.drop_column("project_id")

    op.drop_index("ix_projects_name", table_name="projects")
    op.drop_index("ix_projects_owner_user_id", table_name="projects")
    op.drop_table("projects")
