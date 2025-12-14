"""smart_customer_service package.

Homework goals (from README.md):
- Multi-turn customer service dialog
- Tool calling (order query, refund, invoice plugin)
- Hot reload (model + plugins), with old sessions unaffected
- Health endpoint
- Automated tests
"""

from .app import app  # FastAPI app

__all__ = ["app"]


