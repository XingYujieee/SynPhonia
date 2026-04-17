"""Knowledge base subpackage for activity-based record storage.

Importable as:
    from lite_synphonia.knowledge_base import KnowledgeBaseService, ActivityIngestRecord

Or, from within the project root directory:
    from knowledge_base import KnowledgeBaseService, ActivityIngestRecord

CLI entry point (from the project root directory):
    python -m knowledge_base <command> [options]
"""

from .schemas import ActivityIngestRecord
from .service import KnowledgeBaseService

__all__ = ["ActivityIngestRecord", "KnowledgeBaseService"]
