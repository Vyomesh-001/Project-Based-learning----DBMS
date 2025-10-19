import os
from rule_based_optimizer import optimize_with_rules
from ml_optimizer import optimize_with_ml
def process_sql_file(file_path):
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return

    queries = [q.strip() for q in content.split(';') if q.strip() and not q.strip().startswith('--')]
    
    print(f"Found {len(queries)} queries in {os.path.basename(file_path)}.\n")

    for i, query in enumerate(queries):
        print(f"--- Optimizing Query #{i+1} ---")
        print(f"Original: {query}\n")

        # 1. Apply Rule-Based Optimizer
        print("Applying Rule-Based Optimizer...")
        rule_optimization = optimize_with_rules(query)
        if rule_optimization:
            print(f"   Optimized: {rule_optimization['optimized_query']}")
            print(f"   Reason: {rule_optimization['explanation']}\n")
        else:
            print("   No applicable rules found.\n")

        # 2. Apply ML-Based Optimizer
        print("Applying Machine Learning Optimizer...")
        ml_optimization = optimize_with_ml(query)
        if ml_optimization:
            print(f"   Optimized: {ml_optimization['optimized_query']}")
            print(f"   Reason: {ml_optimization['explanation']}\n")
        else:
            print("   Could not perform ML optimization.\n")
        
        print("-" * 30 + "\n")


if __name__ == '__main__':
    # The path to the file uploaded by the user
    user_file = 'uploads/sample_queries.sql'
    process_sql_file(user_file)