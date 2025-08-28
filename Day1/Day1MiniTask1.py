# Aim: getting all passed students.
# Get mean value then compare exam's score with mean value.
# If it is equal or greater than mean value then student has passed.

import pandas as pd

df = pd.read_csv("C:/Users/Cihangir/Downloads/students_updated.csv")

average_score = df['not'].mean()
passed_exams = df.loc[df['not'] >= average_score]
passed_students = passed_exams['isim'].drop_duplicates()
print("Geçen öğrenciler:\n", passed_students)