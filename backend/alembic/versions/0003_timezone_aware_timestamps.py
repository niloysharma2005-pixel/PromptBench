"""Make timestamp columns timezone-aware

Revision ID: 0003_timezone_aware_timestamps
Revises: 0002_add_slug_and_turn_uq
Create Date: 2026-11-14 18:10:00

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '0003_timezone_aware_timestamps'
down_revision: Union[str, None] = '0002_add_slug_and_turn_uq'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # Convert timestamps from 'timestamp without time zone' to 'timestamp with time zone'
    op.alter_column(
        'users', 'created_at',
        type_=sa.DateTime(timezone=True),
        existing_type=sa.DateTime(),
        existing_server_default=sa.text('now()'),
        existing_nullable=True
    )
    op.alter_column(
        'questions', 'created_at',
        type_=sa.DateTime(timezone=True),
        existing_type=sa.DateTime(),
        existing_server_default=sa.text('now()'),
        existing_nullable=True
    )
    op.alter_column(
        'submissions', 'started_at',
        type_=sa.DateTime(timezone=True),
        existing_type=sa.DateTime(),
        existing_server_default=sa.text('now()'),
        existing_nullable=True
    )
    op.alter_column(
        'submissions', 'submitted_at',
        type_=sa.DateTime(timezone=True),
        existing_type=sa.DateTime(),
        existing_nullable=True
    )
    op.alter_column(
        'transcript_turns', 'created_at',
        type_=sa.DateTime(timezone=True),
        existing_type=sa.DateTime(),
        existing_server_default=sa.text('now()'),
        existing_nullable=True
    )
    op.alter_column(
        'evaluations', 'created_at',
        type_=sa.DateTime(timezone=True),
        existing_type=sa.DateTime(),
        existing_server_default=sa.text('now()'),
        existing_nullable=True
    )

def downgrade() -> None:
    op.alter_column(
        'evaluations', 'created_at',
        type_=sa.DateTime(),
        existing_type=sa.DateTime(timezone=True),
        existing_server_default=sa.text('now()'),
        existing_nullable=True
    )
    op.alter_column(
        'transcript_turns', 'created_at',
        type_=sa.DateTime(),
        existing_type=sa.DateTime(timezone=True),
        existing_server_default=sa.text('now()'),
        existing_nullable=True
    )
    op.alter_column(
        'submissions', 'submitted_at',
        type_=sa.DateTime(),
        existing_type=sa.DateTime(timezone=True),
        existing_nullable=True
    )
    op.alter_column(
        'submissions', 'started_at',
        type_=sa.DateTime(),
        existing_type=sa.DateTime(timezone=True),
        existing_server_default=sa.text('now()'),
        existing_nullable=True
    )
    op.alter_column(
        'questions', 'created_at',
        type_=sa.DateTime(),
        existing_type=sa.DateTime(timezone=True),
        existing_server_default=sa.text('now()'),
        existing_nullable=True
    )
    op.alter_column(
        'users', 'created_at',
        type_=sa.DateTime(),
        existing_type=sa.DateTime(timezone=True),
        existing_server_default=sa.text('now()'),
        existing_nullable=True
    )
