import logging
from aiogram import Bot, Dispatcher, types
from aiogram.types import ContentType
from aiogram.utils import executor
import aiohttp
import aiofiles
import os
import requests
from io import BytesIO
from PIL import Image
import argparse
from gan import run_gan
from dotenv import load_dotenv

load_dotenv()
# Разбираем аргументы командной строки
parser = argparse.ArgumentParser(description="Telegram bot for Monet-style image transformation")
parser.add_argument("--api-token", type=str, help="Telegram API token", default=os.getenv("API_TOKEN"))
args = parser.parse_args()

API_TOKEN = args.api_token

# Включаем логирование для отладки
logging.basicConfig(level=logging.INFO)

# Инициализация бота и диспетчера
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)


# Хендлер команды /start
@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.reply("Привет! Я могу преобразовать изображение в стиле Моне. Просто отправь мне картинку!")


def image2bytes(image):
    bio = BytesIO()
    bio.name = 'image_monet.jpeg'
    image.save(bio, 'JPEG')
    bio.seek(0)

    return bio

# Хендлер для получения изображений
@dp.message_handler(content_types=[ContentType.PHOTO])
async def handle_photo(message: types.Message):
    user_id = message.from_user.id
    try:
        # Скачиваем изображение
        photo = message.photo[-1]  # Берем изображение наивысшего разрешения
        img_url = await message.photo[-1].get_url()
        response = requests.get(img_url, timeout=5)
        img = Image.open(BytesIO(response.content))
        
        pred_img = run_gan(img)

        await bot.send_photo(user_id, photo = image2bytes(pred_img))
        # Отправляем изображение на сервер через POST-запрос

        
    except Exception as e:
        logging.error(f"Ошибка: {e}")
        await message.reply("Произошла ошибка при обработке изображения.")

# Хендлер на случай получения некорректного типа сообщения
@dp.message_handler()
async def handle_text(message: types.Message):
    await message.reply("Пожалуйста, отправьте мне картинку для обработки!")

if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
