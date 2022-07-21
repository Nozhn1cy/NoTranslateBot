import telebot
from translate import translate_sentence

token = 'token'
bot = telebot.TeleBot(token)

@bot.message_handler(commands=['start'])
def start_message(message):
    mess = f'Привет <b>{message.from_user.first_name}</b> ✌'
    bot.send_message(message.chat.id, mess, parse_mode='html')

@bot.message_handler(content_types='text')
def message_reply(message):
    answer, _ = translate_sentence(message.text)
    print(answer)
    bot.send_message(message.chat.id, answer, parse_mode='html')

bot.polling(none_stop=True)
