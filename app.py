from flask import Flask, render_template, request
from src.pipelines.predict_pipeline import Customdata, PredictPipeline


application = Flask(__name__)
app = application


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')

    try:
        # Map form fields to `Customdata` expected names with safe defaults
        gender = request.form.get('gender') or 'female'
        race_ethnicity = request.form.get('race_ethnicity') or request.form.get('ethnicity') or 'group A'
        parental_level_of_education = request.form.get('parental_level_of_education') or request.form.get('parental_level') or "some high school"
        lunch = request.form.get('lunch') or 'standard'
        test_preparation_course = request.form.get('test_preparation_course') or request.form.get('test_prep') or 'none'

        # Safely parse numeric inputs (fall back to 0 when missing)
        try:
            reading_score = float(request.form.get('reading_score') or 0)
        except ValueError:
            reading_score = 0.0
        try:
            writing_score = float(request.form.get('writing_score') or 0)
        except ValueError:
            writing_score = 0.0

        data = Customdata(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score,
        )

        pred_df = data.get_data_as_dataframe()

        pipeline = PredictPipeline()
        result = pipeline.predict(pred_df)
        value = result[0] if hasattr(result, '__len__') else result
        return render_template('home.html', result=value)

    except Exception as e:
        # Render template with error message for debugging in development
        return render_template('home.html', error=str(e))


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)