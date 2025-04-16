class InvalidPrefixError(Exception):

    def __init__(self,*args):
        super().__init__(*args)

class InvalidAnnotationsArray(Exception):

    def __init__(self, *args):
        super().__init__(*args)

class TooManyRectangles(Exception):

    def __init__(self, *args):
        super().__init__(*args)

class DepthTooLarge(Exception):

    def __init__(self, *args):
        super().__init__(*args)
