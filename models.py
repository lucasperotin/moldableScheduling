from math import sqrt, floor, ceil
from task import Task
from model import Model


class AmdahlModel(Model):
    name = "Amdahl"

    def get_alpha(self) -> float:
        return (sqrt(2) + 1 + sqrt(2 * sqrt(2) - 1)) / 2

    def get_mu(self) -> float:
        return (1 - sqrt(8 * sqrt(2) - 11)) / 2

    def time(self, task: Task, nb_proc: int) -> float:
        return task.get_w() * ((1 - task.get_d()) / nb_proc + task.get_d())

    def p_max(self, task: Task, p: int) -> int:
        return p


class GeneralModel(Model):
    name = "General"

    def get_alpha(self) -> float:
        return 2

    def get_mu(self) -> float:
        return (33 - sqrt(738)) / 27

    def time(self, task: Task, nb_proc: int) -> float:
        w, d, p, c = task.get_w(), task.get_d(), task.get_p(), task.get_c()
        return w * ((1 - d) / min(nb_proc, p) + d + c * (nb_proc - 1))

    def p_max(self, task: Task, p: int) -> int:
        w, d, c = task.get_w(), task.get_d(), task.get_c()
        s = sqrt((1-d) / c)
        if task.get_execution_time(floor(s), self) <= task.get_execution_time(ceil(s), self):
            p_tild = floor(s)
        else:
            p_tild = ceil(s)
        return round(min(p, task.get_p(), p_tild))


class RooflineModel(Model):
    name = "Roofline"

    def get_alpha(self) -> float:
        return 1

    def get_mu(self) -> float:
        return (3 - sqrt(5)) / 2

    def time(self, task: Task, nb_proc: int) -> float:
        w, d, p, c = task.get_w(), task.get_d(), task.get_p(), task.get_c()
        return w / min(nb_proc, p)

    def p_max(self, task: Task, p: int) -> int:
        return min(ceil(task.get_p()), p)


# class Power0Model:
#     name = "Power0"

#     def get_alpha(self) -> float:
#         return 1.88  # TODO: get exact values

#     def get_mu(self) -> float:
#         return 1/4.54

#     def time(self, task: Task, nb_proc: int) -> float:
#         w, c = task.get_w(), task.get_c()
#         return w *(1/ nb_proc + c)
    
#     def p_max(self, task: Task, p: int) -> int:
#         return p


class Power25Model(Model):
    name = "Power25"

    def get_alpha(self) -> float:
        return 1.6  # TODO: get exact values

    def get_mu(self) -> float:
        return 1/3.6

    def time(self, task: Task, nb_proc: int) -> float:
        w, c = task.get_w(), task.get_c()
        return w *(1/ nb_proc + c * (nb_proc)**(0.25))

    def p_max(self, task: Task, p: int) -> int:  # TODO
        w, c = task.get_w(),task.get_c()
        s = (4/c)**(1/1.25)
        if task.get_execution_time(floor(s), self) <= task.get_execution_time(ceil(s), self):
            p_tild = floor(s)
        else:
            p_tild = ceil(s)

        return round(min(p, p_tild))


class Power50Model(Model):
    name = "Power50"

    def get_alpha(self) -> float:
        return 1.37  # TODO: get exact values

    def get_mu(self) -> float:
        return 1/3.5

    def time(self, task: Task, nb_proc: int) -> float:
        w, c = task.get_w(), task.get_c()
        return w *(1/ nb_proc + c * (nb_proc)**(0.5))

    def p_max(self, task: Task, p: int) -> int:  # TODO
        w, c = task.get_w(),task.get_c()
        s = (2/c)**(1/1.5)
        if task.get_execution_time(floor(s), self) <= task.get_execution_time(ceil(s), self):
            p_tild = floor(s)
        else:
            p_tild = ceil(s)

        return round(min(p, p_tild))


class Power75Model(Model):
    name = "Power75"

    def get_alpha(self) -> float:
        return 1.41  # TODO: get exact values

    def get_mu(self) -> float:
        return 1/3.48

    def time(self, task: Task, nb_proc: int) -> float:
        w, c = task.get_w(), task.get_c()
        return w *(1/ nb_proc + c * (nb_proc)**(0.75))

    def p_max(self, task: Task, p: int) -> int:  # TODO
        w, c = task.get_w(),task.get_c()
        s = (1/0.75/c)**(1/1.75)
        if task.get_execution_time(floor(s), self) <= task.get_execution_time(ceil(s), self):
            p_tild = floor(s)
        else:
            p_tild = ceil(s)

        return round(min(p, p_tild))



class CommunicationModel(Model):
    name = "Communication"

    def get_alpha(self) -> float:
        return 4 / 3

    def get_mu(self) -> float:
        return (23 - sqrt(313)) / 18

    def time(self, task: Task, nb_proc: int) -> float:
        w, c = task.get_w(), task.get_c()
        return w *(1/ nb_proc + c * (nb_proc - 1))

    def p_max(self, task: Task, p: int) -> int:
        s = sqrt(1/ task.get_c())
        if task.get_execution_time(floor(s), self) <= task.get_execution_time(ceil(s), self):
            p_tild = floor(s)
        else:
            p_tild = ceil(s)

        return round(min(p, p_tild))



class Power100Model(Model):
    name = "Power100"

    def get_alpha(self) -> float:
        return 1.45  # TODO: get exact values

    def get_mu(self) -> float:
        return 1/3.44

    def time(self, task: Task, nb_proc: int) -> float:
        w, c = task.get_w(), task.get_c()
        return w *(1/ nb_proc + c * nb_proc)

    def p_max(self, task: Task, p: int) -> int:
        w, c = task.get_w(), task.get_c()
        s = sqrt(1/c)
        if task.get_execution_time(floor(s), self) <= task.get_execution_time(ceil(s), self):
            p_tild = floor(s)
        else:
            p_tild = ceil(s)

        return round(min(p, p_tild))
