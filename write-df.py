import pandas as pd 

fb = pd.DataFrame({"score" : [],
                "feedback" : []})

fb.to_csv('feedback.csv')