from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from .application import create_consultation_service
from .domain import Consultation


class CreateConsultationRequest(BaseModel):
    consultant_id: str = Field(min_length=1)
    customer_name: str = Field(min_length=1)
    initial_message: str = Field(min_length=1)


class ConsultationResponse(BaseModel):
    consultation_id: str
    thread_id: str
    consultant_id: str
    customer_name: str
    status: str
    messages: list[dict[str, str]]
    trip_requirements: dict[str, Any]

    @classmethod
    def from_domain(cls, consultation: Consultation) -> "ConsultationResponse":
        return cls(
            consultation_id=consultation.id,
            thread_id=consultation.thread_id,
            consultant_id=consultation.consultant_id,
            customer_name=consultation.customer_name,
            status=consultation.status,
            messages=[message.as_dict() for message in consultation.messages],
            trip_requirements=consultation.trip_requirements,
        )


def create_app(database_path: str | Path = "data/travel_agent.db") -> FastAPI:
    app = FastAPI(title="Travel Agent", version="0.1.0")
    service = create_consultation_service(database_path)

    @app.post(
        "/consultations",
        response_model=ConsultationResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def create_consultation(request: CreateConsultationRequest) -> ConsultationResponse:
        consultation = service.create_consultation(
            request.consultant_id,
            request.customer_name,
            request.initial_message,
        )
        return ConsultationResponse.from_domain(consultation)

    @app.get("/consultations/{consultation_id}", response_model=ConsultationResponse)
    def get_consultation(consultation_id: str) -> ConsultationResponse:
        try:
            consultation = service.get_consultation(consultation_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error.args[0])) from error
        return ConsultationResponse.from_domain(consultation)

    return app
