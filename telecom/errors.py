class AppError(Exception):
    def __init__(self, message: str, code: str = "application_error", status: int = 503):
        super().__init__(message)
        self.code = code
        self.status = status


class CustomerNotFound(AppError):
    def __init__(self, customer_id: str):
        super().__init__(f"Fictional customer {customer_id} was not found.", "customer_not_found", 404)
