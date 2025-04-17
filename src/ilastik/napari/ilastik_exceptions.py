class IlastikException(Exception):

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_error_message_box(self):
        return "Something went wrong, check log."

class InvalidPrefixError(IlastikException):

    def __init__(self,*args):
        super().__init__(*args)

    @classmethod
    def get_error_message_box(self):
        return "Invalid prefix."
class InvalidAnnotationsArray(IlastikException):

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_error_message_box(self):
        return "less than two annotations have been passed. You must have two or more labels to run."

class TooManyRectangles(IlastikException):

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_error_message_box(self):
        return "Too many rectangles has been passed in the shapes layer. Please pass one rectangle"

class DepthTooLarge(IlastikException):

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_error_message_box(self):
        return "Given Depth is too large. It needs to be smaller than the image size."

class SameLayerException(IlastikException):

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_error_message_box(self):
        return "You have passed two of the same layer. Please select or make anothor one."


class BoxOutOfBoundsException(IlastikException):

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_error_message_box(self):
        return "The given rectangle is out of bounds. Please make sure at least a part of the box is in the image."

class EmptyFilterListError(IlastikException):

    def __init__(self,*args):
        super().__init__(*args)

    @classmethod
    def get_error_message_box(self):
        return "No filters has been passed"

class IncompatibleFeatures(IlastikException):

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_error_message_box(self):
        return f"The selected features ar not compatible.{super}"
