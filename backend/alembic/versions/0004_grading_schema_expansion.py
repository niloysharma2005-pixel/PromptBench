"""Grading schema expansion for questions and evaluations

Revision ID: 0004_grading_schema_expansion
Revises: 0003_timezone_aware_timestamps
Create Date: 2026-11-14 18:40:00

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '0004_grading_schema_expansion'
down_revision: Union[str, None] = '0003_timezone_aware_timestamps'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # 1. Questions table expansions
    op.add_column('questions', sa.Column('summary', sa.Text(), nullable=True))
    op.add_column('questions', sa.Column('planted_flaws', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column('questions', sa.Column('key_findings', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column('questions', sa.Column('red_herrings', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column('questions', sa.Column('must_include', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column('questions', sa.Column('reveal_text', sa.Text(), nullable=True))

    # 2. Evaluations table expansions
    op.add_column('evaluations', sa.Column('model', sa.String(), nullable=True))
    op.add_column('evaluations', sa.Column('prompt_version', sa.String(), nullable=True))
    op.add_column('evaluations', sa.Column('judge_runs', sa.Integer(), nullable=True))
    op.add_column('evaluations', sa.Column('score_variance', sa.Float(), nullable=True))
    op.add_column('evaluations', sa.Column('dimension_reasoning_json', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column('evaluations', sa.Column('flaw_results_json', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.add_column('evaluations', sa.Column('raw_runs_json', postgresql.JSONB(astext_type=sa.Text()), nullable=True))

def downgrade() -> None:
    op.drop_column('evaluations', 'raw_runs_json')
    op.drop_column('evaluations', 'flaw_results_json')
    op.drop_column('evaluations', 'dimension_reasoning_json')
    op.drop_column('evaluations', 'score_variance')
    op.drop_column('evaluations', 'judge_runs')
    op.drop_column('evaluations', 'prompt_version')
    op.drop_column('evaluations', 'model')

    op.drop_column('questions', 'reveal_text')
    op.drop_column('questions', 'must_include')
    op.drop_column('questions', 'red_herrings')
    op.drop_column('questions', 'key_findings')
    op.drop_column('questions', 'planted_flaws')
    op.drop_column('questions', 'summary')
