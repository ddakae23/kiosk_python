import pandas as pd
import speech_recognition as sr
import pyttsx3
"""
Python 3.8.0 base code

pip install pyaudio
추가로 하셔야 합니다.
"""
# Chatbot 데이터 읽기
chatbot_data = pd.read_excel("./chatbot_data.xlsx", engine="openpyxl")

# Rule을 dictionary 형태로 저장
chat_dic = {}
row = 0
for rule in chatbot_data['rule']:
    chat_dic[row] = rule.split('|')
    row += 1
temp_order = {}
menu = ["따뜻한 아메리카노","따뜻한 라떼","아이스 아메리카노","아이스 라떼","아이스티"]

def chat(request):
    for k, v in chat_dic.items():
        index = -1
        for word in v:
            try:
                if index == -1:
                    index = request.index(word)
                else:
                    if index < request.index(word, index):
                        index = request.index(word, index)
                    else:
                        index = -1
                        break
            except ValueError:
                index = -1
                break
        if index > -1:
            return (chatbot_data['response'][k],k)
    return ('무슨 말인지 모르겠어요',-1)


def recognize():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("말하세요...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
def recognize_and_chat():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("말하세요...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        print("음성 인식 중...")
        text = recognizer.recognize_google(audio, language="ko-KR")
        print("인식된 텍스트:", text)

        # 음성 인식 결과를 chat 함수에 전달하여 처리
        response,idx = chat(text)
        print("단골이 :", response)

        # pyttsx3를 사용하여 음성으로 출력
        engine = pyttsx3.init()
        engine.say(response)
        engine.runAndWait()
        return idx
    except sr.UnknownValueError:
        print("음성을 인식할 수 없습니다.")
        return -1
    except sr.RequestError as e:
        print(f"음성 인식 서비스에 오류가 있습니다: {e}")
        return -1

def customer(idx):
    order = []
    if idx in temp_order:
        temp_hello = "이전에 오신 손님이군요 전에 주문하신 메뉴 그대로 주문해드릴까요"
        order = temp_order[idx]
    else:
        temp_hello = "처음 오신 손님이군요 주문해 주시겠습니까"
        temp_order[idx]=[]
    engine = pyttsx3.init()
    engine.say(temp_hello)
    engine.runAndWait()
    index = -1
    while True:
        user_input = input('텍스트 또는 음성으로 대화를 시작하세요. (exit로 종료): ')

        if user_input.lower() == 'exit':
            break
        elif user_input.lower() == '1':
            # 음성으로 전환
            index=recognize_and_chat()
        else:
            # 텍스트 입력을 chat 함수에 전달하여 처리
            response,index= chat(user_input)

        if 0<=index<=4:
            if order:
                order[0]=menu[index]
            else:
                order.append(menu[index])
        elif 5<=index<=9 and order:
            temp_order[idx]=order
            return 1











if __name__ == "__main__":
    while True:
        user_input = input('텍스트 또는 음성으로 대화를 시작하세요. (exit로 종료): ')

        if user_input.lower() == 'exit':
            break
        elif user_input.lower() == '1':
            # 음성으로 전환
            recognize_and_chat()
        else:
            # 텍스트 입력을 chat 함수에 전달하여 처리
            response = chat(user_input)
            print("단골이 :", response)
