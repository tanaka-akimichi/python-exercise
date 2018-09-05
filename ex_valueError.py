start_number = input("Please Enter Start Number(start with 0): ")

try:
    num = int(start_number)
    print("num={}".format(num))
except ValueError:
    print("Please specify an integer.")