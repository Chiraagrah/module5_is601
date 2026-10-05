import datetime
from pathlib import Path
import pandas as pd
import pytest
from unittest.mock import Mock, patch, PropertyMock
from decimal import Decimal
from tempfile import TemporaryDirectory
from app.calculator import Calculator
from app.calculator_repl import calculator_repl
from app.calculator_config import CalculatorConfig
from app.exceptions import OperationError, ValidationError
from app.history import LoggingObserver, AutoSaveObserver
from app.operations import OperationFactory

# Fixture to initialize Calculator with a temporary directory for file paths
@pytest.fixture
def calculator():
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        config = CalculatorConfig(base_dir=temp_path)

        # Patch properties to use the temporary directory paths
        with patch.object(CalculatorConfig, 'log_dir', new_callable=PropertyMock) as mock_log_dir, \
             patch.object(CalculatorConfig, 'log_file', new_callable=PropertyMock) as mock_log_file, \
             patch.object(CalculatorConfig, 'history_dir', new_callable=PropertyMock) as mock_history_dir, \
             patch.object(CalculatorConfig, 'history_file', new_callable=PropertyMock) as mock_history_file:
            
            # Set return values to use paths within the temporary directory
            mock_log_dir.return_value = temp_path / "logs"
            mock_log_file.return_value = temp_path / "logs/calculator.log"
            mock_history_dir.return_value = temp_path / "history"
            mock_history_file.return_value = temp_path / "history/calculator_history.csv"
            
            # Return an instance of Calculator with the mocked config
            yield Calculator(config=config)

# Test Calculator Initialization

def test_calculator_initialization(calculator):
    assert calculator.history == []
    assert calculator.undo_stack == []
    assert calculator.redo_stack == []
    assert calculator.operation_strategy is None

# Test Logging Setup

@patch('app.calculator.logging.info')
def test_logging_setup(logging_info_mock):
    with patch.object(CalculatorConfig, 'log_dir', new_callable=PropertyMock) as mock_log_dir, \
         patch.object(CalculatorConfig, 'log_file', new_callable=PropertyMock) as mock_log_file:
        mock_log_dir.return_value = Path('/tmp/logs')
        mock_log_file.return_value = Path('/tmp/logs/calculator.log')
        
        # Instantiate calculator to trigger logging
        calculator = Calculator(CalculatorConfig())
        logging_info_mock.assert_any_call("Calculator initialized with configuration")

def test_initialization_warns_when_history_loading_fails(tmp_path):
    config = CalculatorConfig(base_dir=tmp_path)

    with patch.object(
        Calculator, "load_history", side_effect=OSError("corrupt history")
    ), patch("app.calculator.logging.warning") as warning:
        Calculator(config=config)

    warning.assert_called_once_with(
        "Could not load existing history: corrupt history"
    )

def test_setup_logging_prints_and_reraises_directory_error(tmp_path, capsys):
    calculator = Calculator.__new__(Calculator)
    calculator.config = Mock(
        log_dir=tmp_path,
        log_file=tmp_path / "calculator.log",
    )

    with patch("app.calculator.os.makedirs", side_effect=PermissionError("denied")):
        with pytest.raises(PermissionError, match="denied"):
            calculator._setup_logging()

    assert "Error setting up logging: denied" in capsys.readouterr().out

# Test Adding and Removing Observers

def test_add_observer(calculator):
    observer = LoggingObserver()
    calculator.add_observer(observer)
    assert observer in calculator.observers

def test_remove_observer(calculator):
    observer = LoggingObserver()
    calculator.add_observer(observer)
    calculator.remove_observer(observer)
    assert observer not in calculator.observers

# Test Setting Operations

def test_set_operation(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    assert calculator.operation_strategy == operation

# Test Performing Operations

def test_perform_operation_addition(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    result = calculator.perform_operation(2, 3)
    assert result == Decimal('5')

def test_perform_operation_validation_error(calculator):
    calculator.set_operation(OperationFactory.create_operation('add'))
    with pytest.raises(ValidationError):
        calculator.perform_operation('invalid', 3)

def test_perform_operation_operation_error(calculator):
    with pytest.raises(OperationError, match="No operation set"):
        calculator.perform_operation(2, 3)

def test_perform_operation_wraps_unexpected_operation_errors(calculator):
    operation = Mock()
    operation.execute.side_effect = RuntimeError("unexpected failure")
    calculator.set_operation(operation)

    with patch("app.calculator.logging.error") as error:
        with pytest.raises(
            OperationError, match="Operation failed: unexpected failure"
        ):
            calculator.perform_operation(2, 3)

    error.assert_called_once_with("Operation failed: unexpected failure")

def test_perform_operation_keeps_only_the_most_recent_history(calculator):
    calculator.config.max_history_size = 1
    calculator.set_operation(OperationFactory.create_operation("add"))

    calculator.perform_operation(2, 3)
    calculator.perform_operation(4, 5)

    assert len(calculator.history) == 1
    assert calculator.history[0].operand1 == Decimal("4")
    assert calculator.history[0].operand2 == Decimal("5")
    assert calculator.history[0].result == Decimal("9")

# Test Undo/Redo Functionality

def test_undo(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    calculator.undo()
    assert calculator.history == []

def test_redo(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    calculator.undo()
    calculator.redo()
    assert len(calculator.history) == 1

def test_undo_returns_false_when_history_is_empty(calculator):
    assert calculator.undo() is False

def test_redo_returns_false_when_redo_stack_is_empty(calculator):
    assert calculator.redo() is False

# Test History Management

@patch('app.calculator.pd.DataFrame.to_csv')
def test_save_history(mock_to_csv, calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    calculator.save_history()
    mock_to_csv.assert_called_once()

def test_save_empty_history_logs_success(calculator):
    with patch("app.calculator.pd.DataFrame.to_csv") as to_csv, patch(
        "app.calculator.logging.info"
    ) as info:
        calculator.save_history()

    to_csv.assert_called_once_with(calculator.config.history_file, index=False)
    info.assert_called_once_with("Empty history saved")

def test_save_history_wraps_write_errors(calculator):
    with patch(
        "app.calculator.pd.DataFrame.to_csv",
        side_effect=OSError("disk full"),
    ), patch("app.calculator.logging.error") as error:
        with pytest.raises(OperationError, match="Failed to save history: disk full"):
            calculator.save_history()

    error.assert_called_once_with("Failed to save history: disk full")

@patch('app.calculator.pd.read_csv')
@patch('app.calculator.Path.exists', return_value=True)
def test_load_history(mock_exists, mock_read_csv, calculator):
    # Mock CSV data to match the expected format in from_dict
    mock_read_csv.return_value = pd.DataFrame({
        'operation': ['Addition'],
        'operand1': ['2'],
        'operand2': ['3'],
        'result': ['5'],
        'timestamp': [datetime.datetime.now().isoformat()]
    })
    
    # Test the load_history functionality
    try:
        calculator.load_history()
        # Verify history length after loading
        assert len(calculator.history) == 1
        # Verify the loaded values
        assert calculator.history[0].operation == "Addition"
        assert calculator.history[0].operand1 == Decimal("2")
        assert calculator.history[0].operand2 == Decimal("3")
        assert calculator.history[0].result == Decimal("5")
    except OperationError:
        pytest.fail("Loading history failed due to OperationError")
        
def test_load_empty_history_logs_success(calculator):
    with patch("app.calculator.Path.exists", return_value=True), patch(
        "app.calculator.pd.read_csv",
        return_value=pd.DataFrame(),
    ), patch("app.calculator.logging.info") as info:
        calculator.load_history()

    assert calculator.history == []
    info.assert_called_once_with("Loaded empty history file")

def test_load_history_wraps_read_errors(calculator):
    with patch("app.calculator.Path.exists", return_value=True), patch(
        "app.calculator.pd.read_csv",
        side_effect=OSError("invalid CSV"),
    ), patch("app.calculator.logging.error") as error:
        with pytest.raises(OperationError, match="Failed to load history: invalid CSV"):
            calculator.load_history()

    error.assert_called_once_with("Failed to load history: invalid CSV")

def test_get_history_dataframe_returns_string_fields_and_timestamp(calculator):
    calculator.set_operation(OperationFactory.create_operation("add"))
    calculator.perform_operation(2, 3)
    calculation = calculator.history[0]

    dataframe = calculator.get_history_dataframe()

    assert dataframe.to_dict(orient="records") == [{
        "operation": "Addition",
        "operand1": "2",
        "operand2": "3",
        "result": "5",
        "timestamp": calculation.timestamp,
    }]

def test_show_history_formats_each_calculation(calculator):
    calculator.set_operation(OperationFactory.create_operation("add"))
    calculator.perform_operation(2, 3)

    assert calculator.show_history() == ["Addition(2, 3) = 5"]

            
# Test Clearing History

def test_clear_history(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    calculator.clear_history()
    assert calculator.history == []
    assert calculator.undo_stack == []
    assert calculator.redo_stack == []

# Test REPL Commands (using patches for input/output handling)

@patch('builtins.input', side_effect=['exit'])
@patch('builtins.print')
def test_calculator_repl_exit(mock_print, mock_input):
    with patch('app.calculator.Calculator.save_history') as mock_save_history:
        calculator_repl()
        mock_save_history.assert_called_once()
        mock_print.assert_any_call("History saved successfully.")
        mock_print.assert_any_call("Goodbye!")

@patch('builtins.input', side_effect=['help', 'exit'])
@patch('builtins.print')
def test_calculator_repl_help(mock_print, mock_input):
    calculator_repl()
    mock_print.assert_any_call("\nAvailable commands:")

@patch('builtins.input', side_effect=['add', '2', '3', 'exit'])
@patch('builtins.print')
def test_calculator_repl_addition(mock_print, mock_input):
    calculator_repl()
    mock_print.assert_any_call("\nResult: 5")


@pytest.fixture
def repl_mocks():
    with patch("app.calculator_repl.Calculator") as calculator_class, patch(
        "builtins.print"
    ) as print_mock:
        yield calculator_class.return_value, print_mock


def test_calculator_repl_exit_reports_save_failure(repl_mocks):
    calculator, print_mock = repl_mocks
    calculator.save_history.side_effect = OperationError("disk unavailable")

    with patch("builtins.input", side_effect=["exit"]):
        calculator_repl()

    print_mock.assert_any_call(
        "Warning: Could not save history: disk unavailable"
    )
    print_mock.assert_any_call("Goodbye!")


def test_calculator_repl_history_empty(repl_mocks):
    calculator, print_mock = repl_mocks
    calculator.show_history.return_value = []

    with patch("builtins.input", side_effect=["history", "exit"]):
        calculator_repl()

    print_mock.assert_any_call("No calculations in history")


def test_calculator_repl_history_displays_numbered_entries(repl_mocks):
    calculator, print_mock = repl_mocks
    calculator.show_history.return_value = ["Addition(2, 3) = 5", "Power(2, 3) = 8"]

    with patch("builtins.input", side_effect=["history", "exit"]):
        calculator_repl()

    print_mock.assert_any_call("\nCalculation History:")
    print_mock.assert_any_call("1. Addition(2, 3) = 5")
    print_mock.assert_any_call("2. Power(2, 3) = 8")


def test_calculator_repl_clear_history(repl_mocks):
    calculator, print_mock = repl_mocks

    with patch("builtins.input", side_effect=["clear", "exit"]):
        calculator_repl()

    calculator.clear_history.assert_called_once_with()
    print_mock.assert_any_call("History cleared")


@pytest.mark.parametrize(
    ("undo_result", "message"),
    [(True, "Operation undone"), (False, "Nothing to undo")],
)
def test_calculator_repl_undo_result(repl_mocks, undo_result, message):
    calculator, print_mock = repl_mocks
    calculator.undo.return_value = undo_result

    with patch("builtins.input", side_effect=["undo", "exit"]):
        calculator_repl()

    calculator.undo.assert_called_once_with()
    print_mock.assert_any_call(message)


@pytest.mark.parametrize(
    ("redo_result", "message"),
    [(True, "Operation redone"), (False, "Nothing to redo")],
)
def test_calculator_repl_redo_result(repl_mocks, redo_result, message):
    calculator, print_mock = repl_mocks
    calculator.redo.return_value = redo_result

    with patch("builtins.input", side_effect=["redo", "exit"]):
        calculator_repl()

    calculator.redo.assert_called_once_with()
    print_mock.assert_any_call(message)


@pytest.mark.parametrize(
    ("command", "method", "success_message", "error_message"),
    [
        ("save", "save_history", "History saved successfully", "Error saving history"),
        ("load", "load_history", "History loaded successfully", "Error loading history"),
    ],
)
def test_calculator_repl_save_and_load_commands(
    repl_mocks, command, method, success_message, error_message
):
    calculator, print_mock = repl_mocks
    with patch("builtins.input", side_effect=[command, EOFError]):
        calculator_repl()
    getattr(calculator, method).assert_called_once_with()
    print_mock.assert_any_call(success_message)

    getattr(calculator, method).reset_mock()
    print_mock.reset_mock()
    getattr(calculator, method).side_effect = OperationError("storage failure")
    with patch("builtins.input", side_effect=[command, EOFError]):
        calculator_repl()
    getattr(calculator, method).assert_called_once_with()
    print_mock.assert_any_call(f"{error_message}: storage failure")


@pytest.mark.parametrize(
    ("inputs", "expected_operand_calls"),
    [
        (["add", "cancel", "exit"], 1),
        (["add", "2", "cancel", "exit"], 2),
    ],
)
def test_calculator_repl_can_cancel_entering_either_operand(
    repl_mocks, inputs, expected_operand_calls
):
    calculator, print_mock = repl_mocks
    with patch("builtins.input", side_effect=inputs) as input_mock:
        calculator_repl()

    assert input_mock.call_count == expected_operand_calls + 2
    calculator.set_operation.assert_not_called()
    print_mock.assert_any_call("Operation cancelled")


def test_calculator_repl_reports_known_operation_errors(repl_mocks):
    calculator, print_mock = repl_mocks
    calculator.perform_operation.side_effect = ValidationError("invalid number")

    with patch("builtins.input", side_effect=["add", "bad", "3", "exit"]):
        calculator_repl()

    print_mock.assert_any_call("Error: invalid number")


def test_calculator_repl_reports_unexpected_operation_errors(repl_mocks):
    calculator, print_mock = repl_mocks
    calculator.perform_operation.side_effect = RuntimeError("unexpected failure")

    with patch("builtins.input", side_effect=["add", "2", "3", "exit"]):
        calculator_repl()

    print_mock.assert_any_call("Unexpected error: unexpected failure")


def test_calculator_repl_reports_unknown_command(repl_mocks):
    _, print_mock = repl_mocks

    with patch("builtins.input", side_effect=["mystery", "exit"]):
        calculator_repl()

    print_mock.assert_any_call(
        "Unknown command: 'mystery'. Type 'help' for available commands."
    )


def test_calculator_repl_continues_after_keyboard_interrupt(repl_mocks):
    _, print_mock = repl_mocks

    with patch("builtins.input", side_effect=[KeyboardInterrupt, "exit"]):
        calculator_repl()

    print_mock.assert_any_call("\nOperation cancelled")
    print_mock.assert_any_call("Goodbye!")


def test_calculator_repl_exits_on_eof(repl_mocks):
    calculator, print_mock = repl_mocks

    with patch("builtins.input", side_effect=EOFError):
        calculator_repl()

    print_mock.assert_any_call("\nInput terminated. Exiting...")
    calculator.save_history.assert_not_called()


def test_calculator_repl_continues_after_other_command_errors(repl_mocks):
    _, print_mock = repl_mocks

    with patch("builtins.input", side_effect=[RuntimeError("input failed"), "exit"]):
        calculator_repl()

    print_mock.assert_any_call("Error: input failed")
    print_mock.assert_any_call("Goodbye!")


def test_calculator_repl_logs_and_reraises_initialization_errors():
    with patch(
        "app.calculator_repl.Calculator",
        side_effect=RuntimeError("startup failed"),
    ), patch("builtins.print") as print_mock, patch(
        "app.calculator_repl.logging.error"
    ) as error:
        with pytest.raises(RuntimeError, match="startup failed"):
            calculator_repl()

    print_mock.assert_any_call("Fatal error: startup failed")
    error.assert_called_once_with("Fatal error in calculator REPL: startup failed")
