@echo off
echo =====================================================================
echo  Activating Virtual Environment and Launching Supply Chain Predictor
echo =====================================================================
call ..\venv\Scripts\activate.bat
python -m streamlit run app.py
pause
