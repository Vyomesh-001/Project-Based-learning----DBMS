import pandas as pd

def generate_data():
    """
    Generates a simple CSV dataset of inefficient and efficient queries.
    """
    data = {
        'inefficient_query': [
            "SELECT * FROM students",
            "SELECT * FROM courses WHERE department = 'CS'",
            "SELECT id, name, email FROM users WHERE last_login < '2022-01-01'",
            "select * from grades where student_id = 123",
            "select name from professors"
        ],
        'efficient_query': [
            "SELECT student_id, name, course FROM students LIMIT 1000",
            "SELECT course_id, course_name FROM courses WHERE department = 'CS'",
            "SELECT id, name FROM users WHERE last_login < '2022-01-01'",
            "select grade from grades where student_id = 123",
            "select name from professors limit 500"
        ]
    }
    
    df = pd.DataFrame(data)
    df.to_csv('data/query_dataset.csv', index=False)
    print("Dataset created successfully at 'data/query_dataset.csv'")

if __name__ == '__main__':
    generate_data()