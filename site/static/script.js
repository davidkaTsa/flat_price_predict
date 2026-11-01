const form = document.getElementById("apartment-form");
const button = document.getElementById("submit-button");
const result = document.getElementById("result");
const priceElement = document.getElementById("predicted-price");
const errorElement = document.getElementById("error-message");

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    result.hidden = true;
    errorElement.hidden = true;

    const data = new FormData(form);

    const payload = {
        total_area: Number(data.get("total_area")),
        kitchen_area: Number(data.get("kitchen_area")),
        living_area: Number(data.get("living_area")),
        floor: data.get("floor"),
        ceiling_height: Number(data.get("ceiling_height")),
        bathroom_count: Number(data.get("bathroom_count")),
        elevator_count: Number(data.get("elevator_count")),
        property_type: data.get("property_type"),
        building_type: data.get("building_type"),
        address: data.get("address").trim(),
        separate_bathroom: data.has("separate_bathroom"),
        freight_elevator: data.has("freight_elevator"),
    };

    if (
        payload.kitchen_area > payload.total_area ||
        payload.living_area > payload.total_area
    ) {
        errorElement.textContent =
            "Площади кухни и жилой зоны не должны превышать общую площадь.";
        errorElement.hidden = false;
        return;
    }

    button.disabled = true;
    button.textContent = "Рассчитываем стоимость...";

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(payload),
        });

        const answer = await response.json();

        if (!response.ok) {
            throw new Error(
                answer.detail || answer.error || "Ошибка запроса"
            );
        }

        if (answer.error) {
            throw new Error(answer.error);
        }

        const price = Number(answer.predicted_price);

        if (!Number.isFinite(price) || price < 0) {
            throw new Error("Модель вернула некорректную стоимость");
        }

        priceElement.textContent = new Intl.NumberFormat("ru-RU", {
            style: "decimal",
            maximumFractionDigits: 0,
        }).format(price);

        result.hidden = false;
        result.scrollIntoView({
            behavior: "smooth",
            block: "nearest",
        });

    } catch (error) {
        errorElement.textContent =
            error.message || "Не удалось рассчитать стоимость";
        errorElement.hidden = false;
    } finally {
        button.disabled = false;
        button.textContent = "Рассчитать стоимость";
    }
});