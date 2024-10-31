
#I put this bit of code here to avoid circular imports
class Model:
    name: str

    def __init__(self):
        pass

    def get_alpha(self) -> float:
        raise NotImplementedError

    def get_mu(self) -> float:
        raise NotImplementedError

    def time(self, task, nb_proc: int) -> float:
        raise NotImplementedError

    def p_max(self, task, p: int) -> int:
        raise NotImplementedError