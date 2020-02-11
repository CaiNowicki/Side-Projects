import datetime


class Person():
    def __init__(self, name, age, birthdate):
        self.name = name
        self.age = age
        self.birthdate = birthdate
        if (age < 0) | (age > 120):
            age = 21

    def birthstone(self):
        datetime_birthday = datetime.datetime.strptime(self.birthdate, '%m-%d-%Y')
        month = datetime_birthday.month
        birthstones = {1: 'garnet', 2: 'amethyst', 3: 'aquamarine', 4: 'diamond', 5: 'emerald',
                       6: 'pearl', 7: 'ruby', 8: 'peridot', 9: 'sapphire', 10: 'opal',
                       11: 'topaz', 12: 'tanzanite'}
        self.birthstone = birthstones[month]

    def greets(self):
        print(f'Hello {self.name}! My name is {self.__class__.__name__}, nice to meet you')

    def had_birthday(self):
        self.age = self.age + 1

    def present(self):
        print(f'Happy birthday! I wanted to get you a {self.birthstone}'
              f' ring to celebrate, but that was too expensive.')


class Worker(Person):
    def company(self, company):
        self.company = company

    def job_title(self, job_title):
        self.job_title = job_title

    def personal_title(self, personal_title):
        self.personal_title = personal_title

    def college_degree(self, college_degree):
        self.college_degree = college_degree

    def hire_date(self, hire_date):
        self.hire_date = hire_date

    def years_with_company(self):
        years_with_company = (datetime.datetime.today() - datetime.datetime.strptime(
            self.hire_date, '%m-%d-%Y'))
        self.years_with_company = round(years_with_company.days / 365.25)

    def age_hired(self):
        self.age_hired = self.age - self.years_with_company

    def position(self):
        print(f'{self.name} has been a {self.job_title} with {self.company} '
              f'for about {self.years_with_company} years.')

    def greets(self):
        print(f'Hello, {self.personal_title} {self.name}!')


def sumMult35(n):
    # This function finds checks to see if all numbers between 1 and n are a
    # multiple of 3 or five and then adds those ones up.
    # After that it returns that total.
    t = 0
    for i in range(1, n):
        if i % 3 == 0:  # is it a multiple of 3?
            if i % 5 == 0:  # is it a multiple of 5?
                t = t + i  # if passes, add it to total
    return t



import pandas as pd
import numpy as np

test_df = pd.DataFrame(np.random.randint(0,100,size=(15, 4)), columns=list('ABCD'))

from sklearn.model_selection import train_test_split
train, test = train_test_split(test_df)