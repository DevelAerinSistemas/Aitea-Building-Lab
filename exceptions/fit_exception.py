
class InsufficientDataError(Exception):
    def __init__(self, msg="There is not enough data"):
        self.msg = msg
        super().__init__(self.msg)
    
    def __str__(self):
        return super().__str__()
