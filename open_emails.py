import mailparser
import os

#PLAIN_TEXT_MAIL_PATH = "talpinet_email_text\\"
EMAIL_FOLDER_PATH = "talpinet_email_messages\\"

file_list = os.listdir(EMAIL_FOLDER_PATH)
file_path = EMAIL_FOLDER_PATH + file_list[5]

def retreive_mail():
    mail = mailparser.parse_from_file(EMAIL_FOLDER_PATH + 'msg_13901.msg')
    print(mail.From)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    retreive_mail()
    print("Finished!")
