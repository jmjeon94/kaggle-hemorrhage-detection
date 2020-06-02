import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_mail(subject, contents):
    # 세션생성, 로그인
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()  # TLS 사용시 필요
    s.login('jmjeon3155@gmail.com', 'simyiexludrwsxwn')

    # 제목, 본문 작성
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg.attach(MIMEText(contents, 'plain'))

    # 메일 전송
    s.sendmail("jmjeon3155@gmail.com", "jmjeon3155@gmail.com", msg.as_string())
    s.quit()