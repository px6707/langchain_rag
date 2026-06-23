import smtplib
from email.message import EmailMessage

from langchain_core.tools import tool

from app.config import settings


def _validate_email_fields(
    to_email: str,
    subject: str,
    body: str,
    smtp_user: str,
    smtp_password: str,
) -> str | None:
    missing = [
        name
        for name, value in {
            "to_email": to_email,
            "subject": subject,
            "body": body,
            "smtp_user": smtp_user,
            "smtp_password": smtp_password,
        }.items()
        if not str(value).strip()
    ]
    if missing:
        return f"缺少必填字段: {', '.join(missing)}"
    return None


@tool
def send_email(
    to_email: str,
    subject: str,
    body: str,
    smtp_user: str,
    smtp_password: str,
) -> str:
    """发送电子邮件。

    根据用户的描述生成 subject 与 body。smtp_user、smtp_password 若用户未提供则传空字符串，
    将在人工确认环节由用户填写后再发送。

    Args:
        to_email: 收件人邮箱
        subject: 邮件主题
        body: 邮件正文
        smtp_user: 发件邮箱账号
        smtp_password: 发件邮箱密码或授权码
    """
    error = _validate_email_fields(to_email, subject, body, smtp_user, smtp_password)
    if error:
        return error

    message = EmailMessage()
    message["From"] = smtp_user.strip()
    message["To"] = to_email.strip()
    message["Subject"] = subject.strip()
    message.set_content(body.strip())

    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as server:
            if settings.smtp_use_tls:
                server.starttls()
            server.login(smtp_user.strip(), smtp_password)
            server.send_message(message)
    except smtplib.SMTPException as exc:
        return f"邮件发送失败: {exc}"
    except OSError as exc:
        return f"无法连接 SMTP 服务器 {settings.smtp_host}:{settings.smtp_port}: {exc}"

    return f"邮件已成功发送至 {to_email.strip()}"
