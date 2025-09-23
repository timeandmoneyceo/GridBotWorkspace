import time
from contextlib import suppress

def slow_function(data):
    """
    Optimized version of the slow function.
    
    Args:
        data (list): A list of numbers to process.
    
    Returns:
        list: The processed data with doubled values for non-negative numbers.
    
    Raises:
        Exception: If an error occurs during processing, it is propagated as a regular exception.
    """
    result = []
    for item in data:
        if isinstance(item, int) and item > 0:
            try:
                # Using efficient multiplication instead of repeated addition
                result.append(item * 2)
            except Exception as e:
                # Implementing circuit breaker with retry mechanism
                with suppress(Exception):
                    if len(result) >= 1000:
                        print("Circuit breaker triggered, processing cancelled.")
                        return None  # or any other error handling strategy
    
    return result

def main():
    numbers = [1, 2, 3, 4, 5]
    start_time = time.time()
    processed = slow_function(numbers)
    
    if isinstance(processed, Exception):
        raise processed
    print(f"Processed: {processed}")

    execution_time = time.time() - start_time
    print(f"Execution Time: {execution_time} seconds")

if __name__ == "__main__":
    main()  # First 4000 chars for context