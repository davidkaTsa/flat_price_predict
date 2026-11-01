from pydantic import BaseModel, Field


class ApartmentInput(BaseModel):
    """
    Схема параметров для post запроса
    """
    total_area: float = Field(gt=0, description="Общая площадь")
    kitchen_area: float = Field(ge=0, description="Площадь кухни")
    floor: str = Field(min_length=1, description="Этаж")
    property_type: str
    ceiling_height: float = Field(gt=0, description="Высота потолков")
    bathroom_count: int = Field(ge=0, description="Количество санузлов")
    elevator_count: int = Field(ge=0, description="Количество лифтов")
    building_type: str
    living_area: float = Field(ge=0, description="Жилая площадь")
    address: str = Field(min_length=1, description="Адрес")

    separate_bathroom: bool = False
    freight_elevator: bool = False