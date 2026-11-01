from pathlib import Path
import sys
import os
cd = os.getcwd()
sys.path.append(f'{cd}')

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from schemas import ApartmentInput

BASE_DIR = Path(__file__).resolve().parent

MODEL_API_URL = "http://model_server_api:8001/predict"
app = FastAPI(title="Оценка стоимости квартиры")

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)

templates = Jinja2Templates(
    directory=BASE_DIR / "templates"
)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Функция для обработки get запрос на главную страницу сайта.
    """
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.post("/api/predict")
async def predict(apartment: ApartmentInput):
    """
    Функция для обработки post запроса от клиента для дальнейшего взамодействия с API модели.
    """
    payload = apartment.model_dump()

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                MODEL_API_URL,
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

        if "predicted_price" not in result:
            return {
                "error": "API модели вернуло неожиданный формат ответа"
            }

        return {
            "predicted_price": result["predicted_price"]
        }

    except httpx.TimeoutException:
        return {
            "error": "Модель не успела рассчитать стоимость"
        }

    except httpx.HTTPStatusError:
        return {
            "error": "API модели вернуло ошибку"
        }

    except httpx.RequestError:
        return {
            "error": "Не удалось подключиться к API модели"
        }
