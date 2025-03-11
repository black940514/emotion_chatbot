from langchain_core.tools import tool
import smtplib
from email.mime.text import MIMEText
from api_key import email_id, email_pwd
from ics import Calendar, Event

@tool
def send_email(subject, content):
    """Use this tool to send an email."""
    msg = MIMEText(content)
    msg['Subject'] = subject
    msg['From'] = email_id+"@gmail.com"
    msg['To'] = email_id+"@gmail.com"
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login(email_id, email_pwd)
    s.sendmail(email_id+"@gmail.com", email_id+"@gmail.com", msg.as_string())
    s.quit()
    return "Email sent."

@tool
def make_calendar(subject, date):
    """Use this tool to make a ics calendar file for schedule managing."""
    cal = Calendar()
    event = Event()
    event.name = subject
    event.begin = date
    cal.events.add(event)
    filename = 'my.ics'
    with open(filename, 'w') as my_file:
        my_file.writelines(cal.serialize_iter())
    return "ics file generated: " + filename