import sqlparse

def optimize_with_rules(sql_query):
    """
    Applies a set of predefined rules to optimize a given SQL query.
    """
    optimized_queries = []
    
    if 'select *' in sql_query.lower():
        assumed_columns = "student_id, name, course, registration_date"
        optimized_q = sql_query.lower().replace('select *', f'SELECT {assumed_columns}')
        optimized_queries.append({
            "rule": "Avoid SELECT *",
            "original_query": sql_query,
            "optimized_query": optimized_q,
            "explanation": "Replaced 'SELECT *' with specific column names to reduce data retrieval."
        })
    
    current_query = optimized_queries[-1]["optimized_query"] if optimized_queries else sql_query
    
    if not 'limit' in current_query.lower():
        optimized_q_limit = current_query.strip().rstrip(';') + " LIMIT 100;"
        optimized_queries.append({
            "rule": "Add LIMIT Clause",
            "original_query": current_query,
            "optimized_query": optimized_q_limit,
            "explanation": "Added 'LIMIT 100' to prevent unintentionally large data fetches."
        })

    if not optimized_queries:
        return None 
    
    return optimized_queries[-1]

if __name__ == '__main__':
    
    test_query = "SELECT * FROM students WHERE course = 'Computer Science';"
    optimization = optimize_with_rules(test_query)
    if optimization:
        print(f"Original: {test_query}")
        print(f"Optimized: {optimization['optimized_query']}")
        print(f"Reason: {optimization['explanation']}")