class Keisan(object):
    def add_number_plus_double(self, num1, num2):
        if type(num1) is not int or type(num2) is not int:
            raise ValueError

        # 足し算
        result = num1 + num2
        # 2倍する
        result *= 2
        return result
