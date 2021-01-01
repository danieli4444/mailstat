import mailparser
import os

#TALPINET_MAILS_FOLDER_PATH = "talpinet_email_text\\"
TALPINET_EMAIL_MESSAGE_FOLDER = "talpinet_email_messages\\"

file_list = os.listdir(TALPINET_EMAIL_MESSAGE_FOLDER)
file_path = TALPINET_EMAIL_MESSAGE_FOLDER + file_list[5]

def retreive_mail():
    mail = mailparser.parse_from_file(TALPINET_EMAIL_MESSAGE_FOLDER + 'msg_13901.msg')
    print(mail.From)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    retreive_mail()
    print("Finished!")
