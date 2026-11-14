"""Add slug to questions and unique constraint on transcript_turns(submission_id, turn_index)

Revision ID: 0002_add_slug_and_turn_uq
Revises: 0001_baseline
Create Date: 2026-11-14 18:05:00

"""
from typing import Sequence, Union
from alembic import op, context
import sqlalchemy as sa

revision: str = '0002_add_slug_and_turn_uq'
down_revision: Union[str, None] = '0001_baseline'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # 1. Pre-migration check for duplicate (submission_id, turn_index) rows
    bind = op.get_bind()
    if bind is not None and not context.is_offline_mode():
        duplicates = bind.execute(sa.text("""
            SELECT submission_id, turn_index, COUNT(*)
            FROM transcript_turns
            GROUP BY submission_id, turn_index
            HAVING COUNT(*) > 1
        """)).fetchall()
        if duplicates:
            dup_details = ", ".join(
                f"(submission_id={r[0]}, turn_index={r[1]}, count={r[2]})"
                for r in duplicates
            )
            raise RuntimeError(
                f"Cannot create unique constraint 'uq_submission_turn_index'. "
                f"Found duplicate transcript turns: {dup_details}"
            )

    # 2. Add slug column to questions and unique index
    op.add_column('questions', sa.Column('slug', sa.String(), nullable=True))
    op.create_index(op.f('ix_questions_slug'), 'questions', ['slug'], unique=True)

    # 3. Add unique constraint to transcript_turns
    op.create_unique_constraint('uq_submission_turn_index', 'transcript_turns', ['submission_id', 'turn_index'])

def downgrade() -> None:
    op.drop_constraint('uq_submission_turn_index', 'transcript_turns', type_='unique')
    op.drop_index(op.f('ix_questions_slug'), table_name='questions')
    op.drop_column('questions', 'slug')
