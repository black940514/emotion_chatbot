import base64
from PIL import Image
from io import BytesIO
from openai import OpenAI
import gradio as gr
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()
# 환경 변수 가져오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def encode_image(image):
    buffered = BytesIO()
    w,h = image.size
    if w>512 or h>512:
        scale = 512 / max(w,h)
    else:
        scale = 1.0
    resize_im = image.resize((int(w*scale),int(h*scale)))
    resize_im.save(buffered, format="JPEG")
    print(resize_im.size)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def caption_image(image, text_user):
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {
                "role": "user",
                "content":[
                    {
                        "type": "text",
                        "text":  text_user
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(image)}"
                        }
                    }
                ]
            }
        ]
    )
    return response.choices[0].message.content


    