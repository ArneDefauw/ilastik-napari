class IlastikException(Exception):

    def __init__(self, *args):
        super().__init__(*args)

    def get_error_message_box(self):
        return " ".join(self.args)

class InvalidPrefixError(IlastikException):

    def __init__(self,*args):
        super().__init__(*args)

class InvalidAnnotationsArray(IlastikException):

    def __init__(self, *args):
        super().__init__(*args)

class TooManyRectangles(IlastikException):

    def __init__(self, *args):
        super().__init__(*args)

class InvalidDepth(IlastikException):

    def __init__(self, *args):
        super().__init__(*args)

class SameLayerException(IlastikException):

    def __init__(self, *args):
        super().__init__(*args)

class BoxOutOfBoundsException(IlastikException):

    def __init__(self, *args):
        super().__init__(*args)

class EmptyFilterListError(IlastikException):

    def __init__(self,*args):
        super().__init__(*args)

class IncompatibleFeatures(IlastikException):

    def __init__(self, *args):
        super().__init__(*args)

class NoModelFound(IlastikException):

    def __init__(self, *args):
        super().__init__(*args)
