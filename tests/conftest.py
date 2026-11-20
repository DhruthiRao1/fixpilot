import json
from pathlib import Path

import pytest

from fixpilot.graph import FixPilot
from fixpilot.llm import MockLLM
from fixpilot.tools.ticketing import MockTicketClient

FIXTURES = Path(__file__).resolve().parents[1] / "evals" / "fixtures"


@pytest.fixture
def screen():
    def load(name: str) -> bytes:
        return (FIXTURES / f"{name}.json").read_bytes()
    return load


@pytest.fixture
def tickets():
    return MockTicketClient()


@pytest.fixture
def pilot(tickets):
    return FixPilot(llm=MockLLM(), tickets=tickets)


def as_image(data: dict) -> bytes:
    return json.dumps(data).encode()
