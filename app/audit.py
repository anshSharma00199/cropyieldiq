from app.models import AuditLog


def audit(db, actor_id, action: str, detail: dict | None = None, ip: str = ""):
    db.add(AuditLog(actor_id=actor_id, action=action, detail=detail or {}, ip=ip))
    db.commit()
