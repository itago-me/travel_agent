import inspect

from travel_agent.providers import (
    AttractionProvider,
    HotelProvider,
    TransportProvider,
)


def test_provider_protocols_expose_async_search_contracts():
    assert inspect.iscoroutinefunction(TransportProvider.search)
    assert inspect.iscoroutinefunction(HotelProvider.search)
    assert inspect.iscoroutinefunction(AttractionProvider.search)
