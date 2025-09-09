from typing import Callable, Tuple


def add(left: float, right: float) -> float:
    """Return the sum of two numbers."""
    return left + right


def subtract(left: float, right: float) -> float:
    """Return the difference of two numbers (left - right)."""
    return left - right


def multiply(left: float, right: float) -> float:
    """Return the product of two numbers."""
    return left * right


def divide(left: float, right: float) -> float:
    """Return the true division result (left / right). Raises ZeroDivisionError for right == 0."""
    if right == 0:
        raise ZeroDivisionError("Cannot divide by zero.")
    return left / right


def floor_divide(left: float, right: float) -> float:
    """Return the floor division result (left // right). Raises ZeroDivisionError for right == 0."""
    if right == 0:
        raise ZeroDivisionError("Cannot floor-divide by zero.")
    return left // right


def modulus(left: float, right: float) -> float:
    """Return the modulus (left % right). Raises ZeroDivisionError for right == 0."""
    if right == 0:
        raise ZeroDivisionError("Cannot take modulus by zero.")
    return left % right


def power(left: float, right: float) -> float:
    """Return left raised to the power of right."""
    return left ** right


def read_number(prompt: str) -> float:
    """Prompt until the user provides a valid float. Returns the parsed number."""
    while True:
        raw = input(prompt).strip()
        try:
            return float(raw)
        except ValueError:
            print("Invalid number. Please enter a numeric value (e.g., 12, -3.5, 1e3).")


def read_operands() -> Tuple[float, float]:
    """Read and return two operands from the user."""
    left = read_number("Enter first number: ")
    right = read_number("Enter second number: ")
    return left, right


def print_menu() -> None:
    print("\n=== Python Calculator ===")
    print("1) Addition (+)")
    print("2) Subtraction (-)")
    print("3) Multiplication (*)")
    print("4) Division (/)")
    print("5) Floor Division (//)")
    print("6) Modulus (%)")
    print("7) Power (**)")
    print("q) Quit")


def get_operation(choice: str) -> Tuple[str, Callable[[float, float], float]]:
    """Map a menu choice to an operator symbol and function. Raises KeyError for invalid choice."""
    operations: dict[str, Tuple[str, Callable[[float, float], float]]] = {
        "1": ("+", add),
        "2": ("-", subtract),
        "3": ("*", multiply),
        "4": ("/", divide),
        "5": ("//", floor_divide),
        "6": ("%", modulus),
        "7": ("**", power),
    }
    return operations[choice]


def main() -> None:
    while True:
        print_menu()
        choice = input("Choose an option (1-7) or 'q' to quit: ").strip().lower()

        if choice in {"q", "quit", "exit"}:
            print("Goodbye!")
            break

        try:
            symbol, operation = get_operation(choice)
        except KeyError:
            print("Invalid choice. Please select 1-7 or 'q' to quit.")
            continue

        left, right = read_operands()
        try:
            result = operation(left, right)
        except ZeroDivisionError as error:
            print(f"Error: {error}")
            continue

        print(f"Result: {left} {symbol} {right} = {result}")


if __name__ == "__main__":
    main()


