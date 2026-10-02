from datetime import date, time
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProviderModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TransportMode(StrEnum):
    TRAIN = "train"
    FLIGHT = "flight"
    BUS = "bus"


class TransportSearchRequest(ProviderModel):
    origin: str = Field(min_length=1)
    destination: str = Field(min_length=1)
    departure_date: date
    traveler_count: int = Field(ge=1)
    mode: TransportMode | None = None


class TransportOption(ProviderModel):
    provider: str = Field(min_length=1)
    option_id: str = Field(min_length=1)
    mode: TransportMode
    origin: str = Field(min_length=1)
    destination: str = Field(min_length=1)
    departure_date: date
    departure_time: time
    arrival_time: time
    duration_minutes: int = Field(gt=0)
    total_price: float = Field(ge=0)


class TransportSearchResult(ProviderModel):
    provider: str = Field(min_length=1)
    options: list[TransportOption]


class HotelSearchRequest(ProviderModel):
    city: str = Field(min_length=1)
    check_in: date
    check_out: date
    rooms: int = Field(ge=1)

    @model_validator(mode="after")
    def check_dates(self) -> "HotelSearchRequest":
        if self.check_out <= self.check_in:
            raise ValueError("check_out must be after check_in")
        return self


class HotelOption(ProviderModel):
    provider: str = Field(min_length=1)
    option_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    city: str = Field(min_length=1)
    price_per_night: float = Field(ge=0)
    rating: float = Field(ge=0, le=5)
    available_rooms: int = Field(ge=0)


class HotelSearchResult(ProviderModel):
    provider: str = Field(min_length=1)
    options: list[HotelOption]


class AttractionSearchRequest(ProviderModel):
    city: str = Field(min_length=1)
    visit_date: date


class AttractionOption(ProviderModel):
    provider: str = Field(min_length=1)
    option_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    city: str = Field(min_length=1)
    visit_date: date
    duration_minutes: int = Field(gt=0)
    ticket_price: float = Field(ge=0)


class AttractionSearchResult(ProviderModel):
    provider: str = Field(min_length=1)
    options: list[AttractionOption]


class TransportProvider(Protocol):
    async def search(self, request: TransportSearchRequest) -> TransportSearchResult:
        ...


class HotelProvider(Protocol):
    async def search(self, request: HotelSearchRequest) -> HotelSearchResult:
        ...


class AttractionProvider(Protocol):
    async def search(self, request: AttractionSearchRequest) -> AttractionSearchResult:
        ...
