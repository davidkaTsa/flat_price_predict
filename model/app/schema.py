from pydantic import BaseModel


class ClientData(BaseModel):
    """
    Схема параметров для post запроса
    """
    total_area: float
    kitchen_area: float
    floor: str
    property_type: str
    ceiling_height: float
    bathroom_count: int
    elevator_count: int
    building_type: str
    living_area: float
    address: str

    separate_bathroom: bool
    freight_elevator: bool
