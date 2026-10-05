from datetime import datetime
from decimal import Decimal

from app.calculation import Calculation
from app.calculator_memento import CalculatorMemento


def test_memento_to_dict_serializes_history_and_timestamp():
    timestamp = datetime(2024, 1, 2, 3, 4, 5)
    calculation = Calculation(
        operation="Addition",
        operand1=Decimal("2"),
        operand2=Decimal("3"),
        timestamp=timestamp,
    )
    memento = CalculatorMemento(history=[calculation], timestamp=timestamp)

    assert memento.to_dict() == {
        "history": [calculation.to_dict()],
        "timestamp": timestamp.isoformat(),
    }


def test_memento_from_dict_restores_history_and_timestamp():
    timestamp = datetime(2024, 1, 2, 3, 4, 5)
    data = {
        "history": [{
            "operation": "Addition",
            "operand1": "2",
            "operand2": "3",
            "result": "5",
            "timestamp": timestamp.isoformat(),
        }],
        "timestamp": timestamp.isoformat(),
    }

    memento = CalculatorMemento.from_dict(data)

    assert len(memento.history) == 1
    calculation = memento.history[0]
    assert calculation.operation == "Addition"
    assert calculation.operand1 == Decimal("2")
    assert calculation.operand2 == Decimal("3")
    assert calculation.result == Decimal("5")
    assert calculation.timestamp == timestamp
    assert memento.timestamp == timestamp
