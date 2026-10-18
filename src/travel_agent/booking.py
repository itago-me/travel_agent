"""Schemas for simulated high-risk booking tools."""

from pydantic import BaseModel, ConfigDict, Field


class BookingModel(BaseModel):
    """Reject unknown fields so the Agent cannot smuggle extra arguments."""

    model_config = ConfigDict(extra="forbid")


class CreateMockBookingRequest(BookingModel):
    consultation_id: str = Field(min_length=1)
    itinerary_version_id: str = Field(min_length=1)
    total_amount: float = Field(gt=0)
    idempotency_key: str = Field(min_length=1, max_length=200)


class CreateMockBookingResult(BookingModel):
    booking_id: str
    consultation_id: str
    itinerary_version_id: str
    total_amount: float
    status: str
    idempotency_key: str


class CancelMockBookingRequest(BookingModel):
    consultation_id: str = Field(min_length=1)
    booking_id: str = Field(min_length=1)
    reason: str = Field(min_length=1, max_length=500)
    idempotency_key: str = Field(min_length=1, max_length=200)


class CancelMockBookingResult(BookingModel):
    booking_id: str
    consultation_id: str
    status: str
    cancellation_reason: str
    idempotency_key: str
