import streamlit as st

# Streamlit Calculator App
def main():
    st.title("Simple Calculator")
    st.write("Perform basic arithmetic operations")

    # User input for numbers
    num1 = st.number_input("Enter the first number", step=1.0)
    num2 = st.number_input("Enter the second number", step=1.0)

    # User input for operation
    operation = st.selectbox("Select an operation", ["Addition (+)", "Subtraction (-)", "Multiplication (*)", "Division (/)"])

    # Perform the calculation
    result = None
    if st.button("Calculate"):
        if operation == "Addition (+)":
            result = num1 + num2
        elif operation == "Subtraction (-)":
            result = num1 - num2
        elif operation == "Multiplication (*)":
            result = num1 * num2
        elif operation == "Division (/)":
            if num2 != 0:
                result = num1 / num2
            else:
                st.error("Error: Division by zero is not allowed.")

        if result is not None:
            st.success(f"The result is: {result}")

if __name__ == "__main__":
    main()
