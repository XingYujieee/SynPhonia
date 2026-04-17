"""Standalone knowledge base module for activity-based record storage."""

from knowledge_base.schemas import ActivityIngestRecord, InputValidationError
from knowledge_base.service import KnowledgeBaseService

__all__ = ["ActivityIngestRecord", "InputValidationError", "KnowledgeBaseService"]
