#https://api.telegram.org/bot486080396:AAHIIVtOmv3Y1WJpXDrfjCMf75oOfQkAKqQ/getupdates
#смотри все сообщения
import datetime
import time
import json 
import requests

a = datetime.datetime.now()

TOKEN = "615082449:AAFoYG-h5pPGIoNnRgRmkRxNXtqVVGVMYL0"
URL = "https://api.telegram.org/bot{}/".format(TOKEN)
chat = "420227621" # брат


time.sleep(1)

def get_url(url):
    response = requests.get(url)
    content = response.content.decode("utf8")
    return content


def send_message(text):
    try:
        url = URL + "sendMessage?text={}&chat_id={}".format(text, chat)
        get_url(url)
    except Exception as e:
        print(e)


def exit_exe():
    b = datetime.datetime.now()
    text = "Запущенный вами процесс закончил выполняеться"+"\n"+"Время работы процесса: "+str(b-a)
    print(str(text))
    send_message(str(text))

if __name__ == '__main__':
    send_message("HUI")
