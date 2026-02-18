"""Pydantic schemas for support ticket data validation."""

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class SupportTicket(BaseModel):
    """Schema for a support ticket."""

    ticket_id: str
    created_at: datetime
    updated_at: datetime
    customer_id: str
    customer_tier: str
    organization_id: str
    product: str
    product_version: str
    product_module: str
    category: str  # Target variable
    subcategory: str  # Target variable
    priority: str
    severity: str
    channel: str
    subject: str
    description: str
    error_logs: Optional[str] = None
    stack_trace: Optional[str] = None
    customer_sentiment: str
    previous_tickets: int
    resolution: str
    resolution_code: str
    resolved_at: datetime
    resolution_time_hours: float
    resolution_attempts: int
    agent_id: str
    agent_experience_months: int
    agent_specialization: str
    agent_actions: list[str]
    escalated: bool
    escalation_reason: Optional[str] = None
    transferred_count: int
    satisfaction_score: int = Field(ge=1, le=5)
    feedback_text: Optional[str] = None
    resolution_helpful: bool
    tags: list[str]
    related_tickets: list[str]
    kb_articles_viewed: list[str]
    kb_articles_helpful: list[str]
    environment: str
    account_age_days: int
    account_monthly_value: float
    similar_issues_last_30_days: int
    product_version_age_days: int
    known_issue: bool
    bug_report_filed: bool
    resolution_template_used: Optional[str] = None
    auto_suggested_solutions: list[str]
    auto_suggestion_accepted: bool
    ticket_text_length: int
    response_count: int
    attachments_count: int
    contains_error_code: bool
    contains_stack_trace: bool
    business_impact: str
    affected_users: int
    weekend_ticket: bool
    after_hours: bool
    language: str
    region: str

    class Config:
        """Pydantic config."""

        json_encoders = {datetime: lambda v: v.isoformat()}


class TicketStats(BaseModel):
    """Statistics about a ticket dataset."""

    total_tickets: int
    date_range: tuple[datetime, datetime]
    categories: dict[str, int]
    subcategories: dict[str, int]
    products: dict[str, int]
    customer_tiers: dict[str, int]
    avg_resolution_time_hours: float
    avg_satisfaction_score: float
    missing_values: dict[str, int]


class DataSplit(BaseModel):
    """Information about train/val/test split."""

    train_size: int
    val_size: int
    test_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    stratify_column: str
    random_seed: int
