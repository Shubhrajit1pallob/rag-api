from datetime import datetime, timedelta, timezone
from typing import Protocol


class QueryBudget(Protocol):
    def try_consume(self, limit: int) -> bool:
        """Count one query against today's (UTC) budget.

        Returns False, without counting, when `limit` queries were already used today.
        """
        ...


def seconds_until_utc_midnight(now: datetime | None = None) -> int:
    now = now or datetime.now(timezone.utc)
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((tomorrow - now).total_seconds()))
