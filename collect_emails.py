import imaplib
import email
import mailparser
from email.message import EmailMessage

ORG_EMAIL   = "@gmail.com"
FROM_EMAIL  = "mymail" + ORG_EMAIL
FROM_PWD    = "12345678"
SMTP_SERVER = "imap.gmail.com"
SMTP_PORT   = 993

TALPINET_MAILS_FOLDER_PATH = "talpinet_email_text\\"
TALPINET_EMAIL_MESSAGE_FOLDER = "talpinet_email_messages\\"

def getTalpinetMails():
    mail = imaplib.IMAP4_SSL(SMTP_SERVER)
    mail.login(FROM_EMAIL, FROM_PWD)
    mail.select('inbox')
    type, data = mail.search(None, 'ALL')
    mail_ids = data[0]
    id_list = mail_ids.split()
    numoftalpinetemails = 0
    for idx, id in reversed(list(enumerate(id_list))):
        typ, data = mail.fetch(id, '(RFC822)')  # i is the email id
        body = data[0][1]

        try:
            raw_email_string = body.decode('utf-8')
            email_message = email.message_from_string(raw_email_string)
            if '[talpinet]' in email_message['Subject']:
                numoftalpinetemails += 1

                #save raw full email message format to open later
                save_msg_string = TALPINET_EMAIL_MESSAGE_FOLDER + "msg_" + str(idx) + ".msg"
                with open(save_msg_string,'wb') as f:
                    f.write(bytes(data[0][1]))

                # save text format for data validation
                for part in email_message.walk():
                    if part.get_content_type() == "text/plain":  # ignore attachments/html
                        body = part.get_payload(decode=True)
                        # pprint.pprint(part.get_payload(decode=True))
                        save_string = TALPINET_MAILS_FOLDER_PATH + str("dump" + str(idx) + ".txt")
                        # location on disk
                        myfile = open(save_string, 'a', encoding='UTF-8')
                        myfile.write(body.decode('utf-8'))
                        subject = email_message['Subject']

                        myfile.write(subject)
                        # body is again a byte literal
                        myfile.close()
                    else:
                        continue

        except:
            print("couldnt parse mail...")

        print("idx = " + str(idx) + " num of T = " + str(numoftalpinetemails))
        #print("idx = " + str(idx) + " num of T = " + str(numoftalpinetemails), end="\r", flush=True)

    print(idx)
    print(numoftalpinetemails)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    getTalpinetMails()
    print("Finished!")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
