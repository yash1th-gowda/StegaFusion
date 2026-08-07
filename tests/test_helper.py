"""
------------------------------------------------------------
StegaFusion Helper Module Test
------------------------------------------------------------
Tests helper functions.
------------------------------------------------------------
"""

from utils.helper import (
    create_project_directories,
    current_timestamp,
    print_header,
    print_success,
)


def main():

    print_header("Testing Helper Module")

    create_project_directories()

    print_success("Project directories created successfully.")

    print(f"Current Timestamp : {current_timestamp()}")

    print_success("Helper module test completed.")


if __name__ == "__main__":
    main()