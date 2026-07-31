Here is the compiled list of all questions asked during the session, matched directly with their corresponding answers and guidance:



\---



\### \*\*1. Model Selection \& Comparison\*\*



\* \*\*Question:\*\* My model trained on historical data is giving $R^2$ of 83% using Random Forest Model, but when I used XGBoost with hyperparameter tuning it gave $R^2$ of 88%. Please tell me with which model I can proceed?

\* \*\*Answer:\*\* \*\*You can proceed with XGBoost. The PDF lists Random Forest and Ridge as examples, not as the only allowed models. If XGBoost gives a better $R^2$ after proper evaluation, use that and report the comparison (including Random Forest) in your write-up.\*\* \*(Hafsa Imtiaz)\*



\---



\### \*\*2. Frontend/Backend \& Serverless Architecture Requirements\*\*



\* \*\*Question:\*\* The project description mentions a serverless stack. I have already made the data pipeline, training pipeline, and cloud storage serverless. Does the frontend and backend stack also have to be serverless, or is the serverless requirement mainly for the backend, data, and ML infrastructure?

\* \*\*Answer:\*\* \*\*Serverless mainly applies to the data/ML pipelines, storage, and automation, not a special frontend/backend setup. For the app, Streamlit, Gradio, or a stack like React is fine.\*\* \*(Hafsa Imtiaz)\*



\---



\### \*\*3. UI Model Comparison Requirements\*\*



\* \*\*Question:\*\* Do we have to show which model performs better in UI too and give comparisons with other models?

\* \*\*Answer:\*\* \*(Note: Not explicitly answered separately in the transcript, but covered under general instructions)\* \*\*The comparison (including Random Forest vs XGBoost) should be reported in your write-up report.\*\* \*(Hafsa Imtiaz)\*



\---



\### \*\*4. Performance Expectations at the 72-Hour Mark ($R^2$, MAE, RMSE)\*\*



\* \*\*Question:\*\* What MAE, RMSE, and $R^2$ should we aim for at the 72-hour mark? Someone in the Discord channel said 0.7 $R^2$, but at 72 hours it's kind of impossible to get. I have good RMSE and MAE scores, but $R^2$ is around 33% at the 72-hour mark. \*(Another student noted their ceiling was around 0.44 for 72 hours).\*

\* \*\*Answer:\*\* \*\*0.7 will be difficult for 72hrs, so aim for the highest you can reach using your models and training setup. As long as the predictions are realistic, it works.\*\* \*(Hafsa Imtiaz)\*



\---



\### \*\*5. Windows Dependencies \& Environment Variable Handling for Hopsworks\*\*



\* \*\*Question:\*\* I initially faced Hopsworks dependency/Delta/HDFS issues on Windows, so I moved the Hopsworks part to GitHub Codespaces, where I successfully uploaded 17.5k historical records. But now, if I'm switching back to local VS Code, running Hopsworks will give an error. I'm thinking of using an environment variable (`Local VS Code → Hopsworks = false`, `Codespaces → Hopsworks = true`). Is this a good approach or have I done something wrong?

\* \*\*Answer:\*\* \*\*That’s a good approach; you haven’t done anything wrong. Use the env flag so Hopsworks runs in Codespaces and is off locally. Just keep a working local fallback for when it’s off, and note this in your report.\*\* \*(Hafsa Imtiaz)\*



\---



\### \*\*6. HR Email Address\*\*



\* \*\*Question:\*\* Please give the HR email.

\* \*\*Answer:\*\* \*(No answer was provided in the chat transcript).\*

